dofile('init.lua')
dofile('build_recpool_net.lua')
dofile('train_recpool_net.lua')
dofile('display_recpool_net.lua')
dofile('set_recpool_net_structural_params.lua')


cmd = torch.CmdLine()
cmd:text()
cmd:text('Reconstruction pooling network')
cmd:text()
cmd:text('Options')
cmd:option('-log_directory', 'recpool_results', 'directory in which to save experiments')
cmd:option('-load_file','', 'file from which to load experiments')
cmd:option('-num_layers','1', 'number of reconstruction pooling layers in the network')
-- connection_diagram generates figures showing the decoders of the units with the strongest explaining-away connections to and from selected units, a random selection of encoders and decoders, and the most categorical units of each class compared to the average of the input data
-- receptive_fields generates figures showing the 'spike-triggered average' receptive fields and the optimal decoding dictionary (shrink_dictionary) for each unit
-- reconstruction_connections generates figures showing the progressive reconstruction of a few digits 
-- full_diagnostic generates the scatterplots
cmd:option('-run_type','quick_train', 'train slowly over the entire training set (except for the held-out validation elements)') -- (full, quick) x (train, test, diagnostic), connection_diagram, receptive_fields, reconstruction_connections, reconstruction_temporal, energy_landscape; full_diagnostic generates other figures; increase epoch size to generate final figures!!!
cmd:option('-data_set','train', 'data set on which to perform experiment experiments')
cmd:option('-layer_size','200', 'size of sparse coding layer')
cmd:option('-selected_dataset','mnist', 'dataset on which to train (mnist, cifar, or spiral_2d)')

-- set parameters, both from the command line and with fixed values
local L1_scaling = 1 -- CIFAR: 2 works with windows, but seems to be too much with the entire dataset; 1 is too small for the entire dataset; 1.5 - 50% of units are untrained after 30 epochs, 25% are untrained after 50 epochs and many trained units are still distributed high-frequency; 1.25 - 10% of units are untrained after 50 epochs and many trained units are still disbtributed high-frequency

local desired_minibatch_size = 10 -- 0 does pure matrix-vector SGD, >=1 does matrix-matrix minibatch SGD
local desired_test_minibatch_size = 50
-- use 0.5e-3 for spiral_2d dataset
local quick_train_learning_rate = 5e-3 --20e-3 --10e-3 --2e-3 --math.max(1, desired_minibatch_size) * 2e-3 --25e-3 --(1/6)*2e-3 --2e-3 --5e-3
local full_train_learning_rate = 5e-3 --5e-3 --5e-3 --math.max(1, desired_minibatch_size) * 2e-3 --10e-3
local RESET_CLASSIFICATION_DICTIONARY = false
local parameter_save_interval = 50 --20 --50
local classification_scale_factor = 0 -- DEBUG ONLY!!! 1 --0.3 --1

local optimization_algorithm = 'SGD' -- 'SGD', 'ASGD'
local desired_learning_rate_decay = 5e-7
if optimization_algorithm == 'ASGD' then
   desired_learning_rate_decay = 20e-7 --10e-7 --5e-7
   print('using ASGD learning rate decay ' .. desired_learning_rate_decay)
end
local always_track_criteria_outputs = true -- slows things down a little, but gives extra diagnostic information
local num_epochs_no_classification = 100
local delay_entropy_regularization = true -- only apply the entropy loss function after num_epochs_no_classification
local force_initial_learning_rate_decay = false -- force the initial learning rate decay to be equivalent to that after default_pretraining_num_epochs; this happens by default if num_epochs_no_classification <= 0, but must be ensure manually if we're restarting a previously pretrained network with a new entropy-based or weighted-L1 regularizer, lest the pretrained structure be lost due to large initial parameter updates
local num_epochs_gentle_pretraining = -1 -- negative values disable; positive values scale up the learning rate by fast_pretraining_scale_factor after the specified number of epochs
local fast_pretraining_scale_factor = 2
local num_classification_epochs_before_averaging_SGD = 300
local default_pretraining_num_epochs = 100
local num_epochs = 1001
local use_multiplicative_filter = false -- do dropout with nn.MultiplicativeFilter?

local divide_L2_invariance_based_on_L0_invariance = false

if delay_entropy_regularization and classification_scale_factor ~= 0 then
   error('Both delaying entropy regularization and using the classification loss function; make sure that regularization should be delayed')
end

-- extract the command line parameters
local params = cmd:parse(arg)
if params.run_type == 'reconstruction_temporal' then
   params.run_type = 'reconstruction_connections'
   params.use_temporal_reconstruction = true
end
-- choose the dataset
local data_set_spec
if params.selected_dataset == 'mnist' then
   require 'mnist'
   data_set_spec = mnist_spec
elseif params.selected_dataset == 'cifar' then
   require 'cifar'
   data_set_spec = cifar_spec
elseif params.selected_dataset == 'berkeley' then
   require 'cifar'
   data_set_spec = berkeley_spec
elseif params.selected_dataset == 'spiral_2d' then
   require 'spiral_2d'
   data_set_spec = spiral_2d
else
   error('Did not recognize selected_dataset ' .. params.selected_dataset .. '; available datasets are mnist, cifar, and spiral_2d')
end

local restrict_to_window, desired_window_shifts, window_shift_increment, desired_whitened_output_window = data_set_spec:window_params()

local quick_train_epoch_size = math.ceil(0.2 * data_set_spec:train_set_size()) -- 10000 (out of 50000)
local plot_receptive_fields_epoch_size = math.ceil(0.2 * data_set_spec:train_set_size()) --10000 (out of 50000)
local full_diagnostic_epoch_size --this is set separately for diagnostic runs and reconstructions below

if (params.run_type == 'full_diagnostic') or (params.run_type == 'quick_diagnostic') then
   full_diagnostic_epoch_size, desired_window_shifts = data_set_spec:diagnostic_params()
elseif params.run_type == 'reconstruction_connections' then
   full_diagnostic_epoch_size, desired_window_shifts, window_shift_increment = data_set_spec:reconstruction_params()
elseif params.run_type == 'invariance' then
   desired_test_minibatch_size = 101 -- must be odd, since window shifts are symmetric around the center index
   full_diagnostic_epoch_size, desired_window_shifts, window_shift_increment = data_set_spec:invariance_params(desired_test_minibatch_size)
end
   

params.num_layers = tonumber(params.num_layers)
params.fe_layer_size = tonumber(params.layer_size) --200 --400 --200
params.p_layer_size = 50 --200 --50

if (params.run_type == 'quick_test') or (params.run_type == 'full_test') or (params.run_type == 'quick_diagnostic') or (params.run_type == 'full_diagnostic') or 
   (params.run_type == 'receptive_fields') or (params.run_type == 'connection_diagram') or (params.run_type == 'reconstruction_connections') or (params.run_type == 'energy_landscape') then
   num_epochs_no_classification = 0
   num_epochs = 1
elseif (params.run_type == 'invariance') then
   num_epochs_no_classification = 0
   if divide_L2_invariance_based_on_L0_invariance then
      num_epochs = 2 -- one epoch to determine which units are invariant to single-pixel shifts, one epoch to calculate average hidden unit differences on these units
   else
      num_epochs = 1
   end
end


-- recpool_config_prefs are num_ista_iterations, shrink_style, disable_pooling, use_squared_weight_matrix, normalize_each_layer, repair_interval
local recpool_config_prefs = {}
recpool_config_prefs.num_ista_iterations = 10 --15 --14 
recpool_config_prefs.num_loss_function_ista_iterations = 1 --5
--recpool_config_prefs.shrink_style = 'ParameterizedShrink'
recpool_config_prefs.shrink_style = 'FixedShrink'
--recpool_config_prefs.shrink_style = 'SoftPlus' --'FixedShrink' --'ParameterizedShrink'
recpool_config_prefs.disable_pooling = true
recpool_config_prefs.disable_pooling_losses = false
recpool_config_prefs.use_squared_weight_matrix = true
recpool_config_prefs.normalize_each_layer = false -- THIS IS NOT YET IMPLEMENTED!!!
recpool_config_prefs.randomize_pooling_dictionary = true
recpool_config_prefs.repair_interval = 5 --((desired_minibatch_size <= 1) and 5) or 1
recpool_config_prefs.manually_maintain_explaining_away_diagonal = true
recpool_config_prefs.use_multiplicative_filter = use_multiplicative_filter -- do dropout with nn.MultiplicativeFilter?


-- seed the random number generator
--torch.manualSeed(46393475) -- init random number generator.  Obviously, this should be taken from the clock when doing an actual run
torch.seed()


-- create the dataset
local data, data_inline_test
-- 'recpool_net' option ensures that the returned table contains elements data and labels, for which the __index method is overloaded.  
local data_set_options = {train_or_test = params.data_set, maxLoad = 0, alternative_access_method = 'recpool_net', offset = nil, restrict_to_window = restrict_to_window, 
			  desired_window_shifts = desired_window_shifts, window_shift_increment = window_shift_increment, desired_whitened_output_window = desired_whitened_output_window}
if params.data_set == 'train' then
   data_set_options.maxLoad = (((params.run_type == 'full_train') or (params.run_type == 'full_test')) and data_set_spec:train_set_size()) or 
      ((params.run_type == 'quick_diagnostic') and desired_test_minibatch_size) or
      (((params.run_type == 'full_diagnostic') or (params.run_type == 'reconstruction_connections') or (params.run_type == 'invariance')) and full_diagnostic_epoch_size) or
      ((params.run_type == 'receptive_fields') and plot_receptive_fields_epoch_size) or
      quick_train_epoch_size
   if params.run_type == 'full_train' then
      -- also load the validation set for inline testing
      data_inline_test = data_set_spec:loadDataSet({train_or_test = 'train', maxLoad = data_set_spec:validation_set_size(), alternative_access_method = 'recpool_net', 
						    offset = data_set_spec:validation_set_offset(), restrict_to_window = restrict_to_window, desired_window_shifts = desired_window_shifts, 
						    window_shift_increment = window_shift_increment, desired_whitened_output_window = desired_whitened_output_window})
      --data_inline_test:normalizeByColor()
      --data_inline_test:useGrayscale()
      --data_inline_test:normalizeL2() -- use for 12/7 CIFAR
      data_inline_test:normalizeStandard() -- use otherwise
   end
elseif params.data_set == 'test' then
   data_set_options.maxLoad = (((params.run_type == 'full_train') or (params.run_type == 'full_test') or (params.run_type == 'full_diagnostic') or (params.run_type == 'receptive_fields')) and data_set_spec:test_set_size()) or ((params.run_type == 'reconstruction_connections') and 1000) or 5000
elseif params.data_set == 'validation' then
   data_set_options.maxLoad = (((params.run_type == 'full_train') or (params.run_type == 'full_test') or (params.run_type == 'full_diagnostic') or (params.run_type == 'receptive_fields')) and data_set_spec:validation_set_size()) or ((params.run_type == 'reconstruction_connections') and 1000) or ((params.run_type == 'quick_diagnostic') and desired_test_minibatch_size) or 5000
   data_set_options.train_or_test = 'train' -- validation set is just the end of the train set; 'test' is already set correctly in the original definition of data_set_options
   data_set_options.offset = data_set_spec:train_set_size()
else
   error('unrecognized data set: ' .. params.data_set)
end

data = data_set_spec:loadDataSet(data_set_options) 


--Indexing labels returns an index, rather than a tensor
--data:normalizeByColor()
--data:useGrayscale()
--data:normalizeL2() -- normalize each example to have L2 norm equal to 1 -- use for 12/7 CIFAR
data:normalizeStandard() -- use otherwise


if params.run_type == 'invariance' then
   print('data set size is ' .. data:nExample() .. ', minibatch size is ' .. desired_test_minibatch_size .. ', ratio is ' .. data:nExample() / desired_test_minibatch_size)
end

-- the code required to set the structural parameters of a network is messy, so it's been moved to a separate file
local layer_size, layered_lambdas, layered_lagrange_multiplier_targets, layered_lagrange_multiplier_learning_rate_scaling_factors = 
   set_recpool_net_structural_params(recpool_config_prefs, data, params, L1_scaling)

--layered_lambdas[1].ista_L2_reconstruction_lambda = 0 -- DEBUG ONLY!!! FOR THE LOVE OF GOD!!!
--layered_lambdas[1].ista_L1_lambda = 0 -- DEBUG ONLY!!! FOR THE LOVE OF GOD!!!


local model = build_recpool_net(layer_size, layered_lambdas, classification_scale_factor, layered_lagrange_multiplier_targets, layered_lagrange_multiplier_learning_rate_scaling_factors, recpool_config_prefs, data) -- last argument is num_ista_iterations

-- option array for RecPoolTrainer
-- must use data:nExample() rather than data_set_spec:train_set_size since the latter does not include the number of views of windowed datasets.  This may be dangerous if we try to train on something other than the training dataset!
local default_pretraining_minibatches = default_pretraining_num_epochs * data:nExample() / math.max(1, desired_minibatch_size) 
opt = {log_directory = params.log_directory, -- subdirectory in which to save/log experiments
   visualize = false, -- visualize input data and weights during training
   plot = false, -- live plot
   num_classification_categories = data:nClass(),
   optimization = optimization_algorithm, -- optimization method: SGD | ASGD | CG | LBFGS
   init_eval_counter = (((num_epochs_no_classification <= 0) or force_initial_learning_rate_decay) and default_pretraining_minibatches) or 0, 
   learning_rate = ((params.run_type == 'full_train') and full_train_learning_rate) or 
      ((params.run_type == 'quick_train') and quick_train_learning_rate) or 
      (((params.run_type == 'full_test') or (params.run_type == 'quick_test') or (params.run_type == 'full_diagnostic') or (params.run_type == 'quick_diagnostic') or (params.run_type == 'receptive_fields') or (params.run_type == 'connection_diagram') or (params.run_type == 'reconstruction_connections') or (params.run_type == 'energy_landscape') or (params.run_type == 'invariance')) and 0) or 0, --1e-3, -- learning rate at t=0
   batch_size = desired_minibatch_size, -- mini-batch size (0 = pure stochastic)
   test_batch_size = desired_test_minibatch_size,
   learning_rate_decay = desired_learning_rate_decay * math.max(1, desired_minibatch_size), -- learning rate decay is performed based upon the number of calls to SGD.  When using minibatches, we must increase the decay in proportion to the minibatch size to maintain parity based upon the number of datapoints examined
   weight_decay = 0, --2e-3, --1e-3, --0, -- weight decay (SGD only)
   L3_weight_decay = 0, --1e-3
   L1_weight_decay = 0, --1e-4, --1e-5, -- L1 weight decay (SGD only)
   momentum = 0.5, -- momentum (SGD only) --WAS 0!!!
   t0 = (((num_epochs_no_classification <= 0) and default_pretraining_minibatches) or 
	 num_epochs_no_classification * (data:nExample() / math.max(1, desired_minibatch_size))) + 
      num_classification_epochs_before_averaging_SGD * (data:nExample() / math.max(1, desired_minibatch_size)), -- start averaging at t0 (ASGD only), measured in ASGD calls
   max_iter = 2, -- maximum nb of iterations for CG and LBFGS
   plot_temporal_reconstructions = params.use_temporal_reconstruction
}

local trainer_run_type = nil
if (params.run_type == 'full_diagnostic') or (params.run_type == 'reconstruction_connections') or (params.run_type == 'invariance') then
   trainer_run_type = 'display' -- always plot the same elements, in the same order; forcibly disable training 
elseif (params.run_type == 'full_test') or (params.run_type == 'quick_test') or (params.run_type == 'quick_diagnostic') or (params.run_type == 'receptive_fields') or (params.run_type == 'connection_diagram') or (params.run_type == 'energy_landscape') then
   trainer_run_type = 'validation' -- forcibly disable training
end

print('Using opt.learning_rate = ' .. opt.learning_rate)


local track_criteria_outputs = always_track_criteria_outputs or not((params.run_type == 'full_train') or (params.run_type == 'full_test'))
local receptive_field_builder = nil
if (params.run_type == 'full_diagnostic') or (params.run_type == 'quick_diagnostic') or (params.run_type == 'receptive_fields') or (params.run_type == 'reconstruction_connections') then
   receptive_field_builder = receptive_field_builder_factory(data:nExample(), data:dataSize(), layer_size[2], 1+recpool_config_prefs.num_ista_iterations, model)
elseif params.run_type == 'invariance' then
   receptive_field_builder = invariance_builder_factory(layer_size[2], desired_test_minibatch_size)
end
local trainer = nn.RecPoolTrainer(model, opt, layered_lambdas, track_criteria_outputs, receptive_field_builder) -- layered_lambdas is required for debugging purposes only


-- load parameters from file if desired
if params.load_file ~= '' then
   local average_saved_params = false
   if average_saved_params then
      local real_params = trainer:get_flattened_parameters()
      local temp_params = torch.Tensor():resizeAs(real_params)
      real_params:zero()
      for j = 6,9 do
	 for i = 0,8,2 do
	    load_parameters(temp_params, (params.load_file .. tostring(j) .. tostring(i) .. '1.bin'))
	    real_params:add(0.2/4, temp_params)
	 end
      end
   else
      load_parameters(trainer:get_flattened_parameters(), params.load_file)
   end
   --model.layers[1].module_list.explaining_away.weight:add(1, torch.diag(torch.ones(model.layers[1].module_list.explaining_away.weight:size(1))))
   --print(torch.diag(model.layers[1].module_list.explaining_away.weight):unfold(1,10,10))
end

if params.run_type == 'invariance' then
   receptive_field_builder:set_encoder_and_decoder(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight, 
						   model.layers[1].module_list.decoding_feature_extraction_dictionary.weight)
end

os.execute('rm -rf ' .. opt.log_directory)
os.execute('mkdir -p ' .. opt.log_directory)

-- confirm that shrink parameters are properly shared
if shrink_style == 'ParameterizedShrink' then
   for i = 1,#model.layers do
      print('for layer ' .. i)
      local shrink = model.layers[i].module_list.shrink
      local shrink_copies = model.layers[i].module_list.shrink_copies
      for j = 1,#shrink_copies do
	 print('checking sharing for shrink copy ' .. j)
	 if shrink.shrink_val:storage() ~= shrink_copies[j].shrink_val:storage() then
	    print('Before training, shrink storage is not the same as the shrink copy ' .. j .. ' storage!!!')
	    io.read()
	 end
      end
   end
end


if params.run_type == 'connection_diagram' then
   save_filter(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight:t():narrow(2,90,40), 'selected_encoding_fe_dict', opt.log_directory, 20)
   save_filter(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:narrow(2,90,40), 'selected_decoding_fe_dict', opt.log_directory, 20)

   plot_most_categorical_filters(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight,
				 model.layers[1].module_list.decoding_feature_extraction_dictionary.weight,
				 model.module_list.classification_dictionary.weight,
				 data, opt)

   plot_explaining_away_connections(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight,
				    model.layers[1].module_list.decoding_feature_extraction_dictionary.weight, 
				    model.layers[1].module_list.explaining_away.weight, opt)
   plot_explaining_away_connections(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight,
				    model.layers[1].module_list.decoding_feature_extraction_dictionary.weight, 
				    model.layers[1].module_list.explaining_away.weight, opt, 'restrict to positive')


   plot_explaining_away_connections(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight,
				    model.layers[1].module_list.decoding_feature_extraction_dictionary.weight, 
				    model.layers[1].module_list.explaining_away.weight, opt, {'destination', 'any', 'part'},
				    model.module_list.classification_dictionary.weight, 1, 20) --6, 3 for ICLR paper

   plot_explaining_away_connections(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight,
				    model.layers[1].module_list.decoding_feature_extraction_dictionary.weight, 
				    model.layers[1].module_list.explaining_away.weight, opt, {'destination', 'part', 'categorical'},
				    model.module_list.classification_dictionary.weight, 1, 20) --16, 3 for ICLR paper
   
   
   local sort_by_class = nil -- normally true
   plot_explaining_away_connections(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight,
				    model.layers[1].module_list.decoding_feature_extraction_dictionary.weight, 
				    model.layers[1].module_list.explaining_away.weight, opt, {'source', 'part', 'categorical', sort_by_class},
				    model.module_list.classification_dictionary.weight, 1, 20) --16, 3 for ICLR paper
   plot_explaining_away_connections(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight,
				    model.layers[1].module_list.decoding_feature_extraction_dictionary.weight, 
				    model.layers[1].module_list.explaining_away.weight, opt, {'source', 'categorical', 'categorical', sort_by_class},
				    model.module_list.classification_dictionary.weight, 1, 10) --16, 3 for ICLR paper
   return
elseif params.run_type == 'energy_landscape' then
      model:reset_classification_lambda(0)
      plot_energy_landscape_2d(model, data, 100, opt)
      return
end



-- consider increasing learning rate when classification loss is disabled; otherwise, new features in the feature_extraction_dictionaries are discovered very slowly
model:reset_classification_lambda(0) -- SPARSIFYING LAMBDAS SHOULD REALLY BE TURNED UP WHEN THE CLASSIFICATION CRITERION IS DISABLED
if delay_entropy_regularization then
   print('Turning entropy regularization off')
   model:reset_entropy_scale_factor(0)
end

for i = 1,num_epochs_no_classification do
   if ((i % parameter_save_interval == 1) or (parameter_save_interval == 1)) and (i >= 1) then -- make sure to save the initial paramters, before any training occurs, to allow comparisons later
      save_parameters(trainer:get_output_flattened_parameters(), opt.log_directory, i) -- defined in display_recpool_net
   end

   if (num_epochs_gentle_pretraining >= 0) and (i == num_epochs_gentle_pretraining) then
      trainer:reset_learning_rate(fast_pretraining_scale_factor * opt.learning_rate)
   end

   trainer:train(data, trainer_run_type)
   print('Effective learning rate decay is ' .. trainer.config.evalCounter * trainer.config.learningRateDecay)
   --print('iterations so far: ' .. trainer.config.evalCounter)
   if (i < 30) or (i % 10 == 1) then
      plot_filters(opt, i, model.filter_list)
   end
end

-- reset lambdas to be closer to pure top-down fine-tuning and continue training
model:reset_classification_lambda(classification_scale_factor) -- 0.2 seems to strike an even balance between reconstruction and classification
if delay_entropy_regularization then
   print('Turning entropy regularization back on')
   model:reset_entropy_scale_factor(1)
end
if ((opt.weight_decay > 0) or (opt.L1_weight_decay > 0)) and (RESET_CLASSIFICATION_DICTIONARY or (num_epochs_no_classification > 0)) then
   model:reset_classification_dictionary() -- ensure that weight decay during the unsupervised pretraining doesn't cause the classification dictionary to grow too small
end
if num_epochs_gentle_pretraining >= 0 then
   trainer:reset_learning_rate(opt.learning_rate)
end
--trainer:reset_learning_rate(5e-3) -- potentially use faster learning rate for the unsupervised pretraining, then revert to a more careful learning rate for supervised training with the classification loss
--trainer.config.evalCounter = 0 -- reset counter for learning rate decay; this maintains consistency between full runs and runs initialized with an unsupervised-pretrained network
local perform_classifier_pretraining = false
local perform_slow_burn_in = false
local num_epochs_classification_pretraining = 20 --5
local num_epochs_classification_slow_burn_in = 0
local slow_burn_in_scale_factor = 0.1 -- 0.05

if perform_classifier_pretraining then
   --trainer:reset_learning_rate(20 * (5000 / data_set_size) * opt.learning_rate) -- Do a few epochs of accelerated training to initialize the classification_dictionary
   --print('resetting learning rate to ' .. 20 * (5000 / data_set_size) * opt.learning_rate)
   print('Doing fast classifier pretraining')
   model:reset_classification_lambda(2*classification_scale_factor)
   model:reset_ista_learning_scale_factor(0) -- turn off learning for all ISTA ConstrainedLinear modules, leaving only the classification dictionary to be trained.  THIS DOES NOT WORK PROPERLY WITH PARAMETERIZED_SHRINK!!!
end

for i = 1+num_epochs_no_classification,num_epochs+num_epochs_no_classification do
   -- Train the entire network with a very low learning rate for a few epochs, to resolve mismatches between the sparse coding layer and the classification layer.  Remember that, since the classification dictionary is now so large, the backpropagated gradients are correspondingly scaled up
   if perform_classifier_pretraining and (i == num_epochs_classification_pretraining + num_epochs_no_classification) then
      --trainer:reset_learning_rate(opt.learning_rate) 

      print('Ending fast classifier pretraining')
      model:reset_classification_lambda(classification_scale_factor)
      model:reset_ista_learning_scale_factor(1) 
      --trainer:reset_learning_rate(slow_burn_in_scale_factor * opt.learning_rate) 
      --print('resetting learning rate to ' .. slow_burn_in_scale_factor * opt.learning_rate)
   elseif perform_slow_burn_in and perform_classifier_pretraining and (i == num_epochs_classification_slow_burn_in + num_epochs_classification_pretraining + num_epochs_no_classification) then
      error('USING SLOW BURN-IN!!!')
      --model:reset_learning_scale_factor(0.2) -- this is largely equivalent to the learning rate decay after 200 epochs, which we removed above
      model:reset_classification_lambda(0.1)
      local final_scale_factor = 5 --0.25 -- 0.15
      trainer:reset_learning_rate(final_scale_factor * opt.learning_rate) 
      print('resetting learning rate to ' .. final_scale_factor * opt.learning_rate)
   end
   
   if ((i % parameter_save_interval == 1) or (parameter_save_interval == 1)) and (i > 1) then
      save_parameters(trainer:get_output_flattened_parameters(), opt.log_directory, i) -- defined in display_recpool_net
   end

   if (i % 2 == 1) and (i > 1) and data_inline_test then
      print('starting test epoch')
      trainer:train(data_inline_test, 'validation') --second argument specifies that this is a test epoch; no learning is performed even though the learning rate is not set to zero
      print('writing performance ' .. trainer.current_performance)
      save_performance_history(trainer.current_performance, opt.log_directory, i) -- defined in display_recpool_net
   end

   trainer:train(data, trainer_run_type)
   print('Effective learning rate decay is ' .. trainer.config.evalCounter * trainer.config.learningRateDecay)
   if (i < 30) or (i % 10 == 1) then
      plot_filters(opt, i, model.filter_list)
   end

   if (receptive_field_builder and (params.run_type == 'receptive_fields')) then 
      receptive_field_builder:plot_receptive_fields(opt, model.layers[1].module_list.encoding_feature_extraction_dictionary.weight, 
						    model.layers[1].module_list.decoding_feature_extraction_dictionary.weight) 
   end
   if (receptive_field_builder and (params.run_type == 'full_diagnostic')) then receptive_field_builder:plot_other_figures(opt) end
   if (receptive_field_builder and (params.run_type == 'quick_diagnostic')) then receptive_field_builder:quick_diagnostic_plots(opt) end
   if (receptive_field_builder and (params.run_type == 'reconstruction_connections')) then receptive_field_builder:plot_reconstruction_connections(opt) end
   if (receptive_field_builder and (params.run_type == 'invariance')) then 
      if divide_L2_invariance_based_on_L0_invariance and (i == 1) then
	 print('resetting filters based on invariance')
	 receptive_field_builder:reset_filters_based_on_invariance(10, 50)
      else
	 print('producing invariance scatterplots')
	 receptive_field_builder:plot_invariance_scatterplot(opt, model.layers[1].module_list.encoding_feature_extraction_dictionary.weight, 
							     model.layers[1].module_list.decoding_feature_extraction_dictionary.weight)  
      end
   end
   
end







