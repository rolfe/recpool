dofile('init.lua')
dofile('build_recpool_net.lua')
dofile('train_recpool_net.lua')
dofile('display_recpool_net.lua')


cmd = torch.CmdLine()
cmd:text()
cmd:text('Reconstruction pooling network')
cmd:text()
cmd:text('Options')
cmd:option('-log_directory', 'recpool_results', 'directory in which to save experiments')
cmd:option('-load_file','', 'file from which to load experiments')
cmd:option('-num_layers','1', 'number of reconstruction pooling layers in the network')
cmd:option('-full_test','quick_train', 'train slowly over the entire training set (except for the held-out validation elements)')
cmd:option('-data_set','train', 'data set on which to perform experiment experiments')

local desired_minibatch_size = 1 -- 0 does pure matrix-vector SGD, >=1 does matrix-matrix minibatch SGD
local quick_train_learning_rate = math.max(1, desired_minibatch_size) * 5e-3 --25e-3 --(1/6)*2e-3 --2e-3 --5e-3
local full_train_learning_rate = math.max(1, desired_minibatch_size) * 2e-3 --10e-3
local quick_train_epoch_size = 5000

local num_epochs_no_classification = 200 --200 --501 --201
local num_epochs = 1000

local fe_layer_size = 200 --400 --200
local p_layer_size = 50 --200 --50

local params = cmd:parse(arg)
local num_layers = tonumber(params.num_layers)

local sl_mag = nil
local rec_mag = nil
local pooling_rec_mag = nil
local pooling_orig_rec_mag = nil
local pooling_shrink_position_L2_mag = nil
local pooling_orig_position_L2_mag = nil
local pooling_sl_mag = nil
local mask_mag = nil

-- recpool_config_prefs are num_ista_iterations, shrink_style, disable_pooling, use_squared_weight_matrix, normalize_each_layer
local recpool_config_prefs = {}
recpool_config_prefs.num_ista_iterations = 5 --5 --5 --3
--recpool_config_prefs.shrink_style = 'ParameterizedShrink'
recpool_config_prefs.shrink_style = 'FixedShrink'
--recpool_config_prefs.shrink_style = 'SoftPlus' --'FixedShrink' --'ParameterizedShrink'
recpool_config_prefs.disable_pooling = true
local disable_pooling_losses = false
recpool_config_prefs.use_squared_weight_matrix = true
recpool_config_prefs.normalize_each_layer = false -- THIS IS NOT YET IMPLEMENTED!!!
recpool_config_prefs.randomize_pooling_dictionary = true

pooling_sl_mag = 0.5e-2 --0.9e-2 --0.5e-2 --0.15e-2 --0.25e-2 --2e-2 --5e-2 -- keep in mind that there are four times as many mask outputs as pooling outputs in the first layer -- also remember that the columns of decoding_pooling_dictionary are normalized to be the square root of the pooling factor.  However, before training, this just ensures that all decoding projections have a magnitude of one
mask_mag = 0.3e-2 --0.2e-2 --0.3e-2 --0.4e-2 --0.5e-2 --0 --0.75e-2 --0.5e-2 --0.75e-2 --8e-2 --4e-2 --2.5e-2 --1e-1 --5e-2

--pooling_sl_mag = 0.1e-2
--mask_mag = 0.35e-2 --0.375e-2 --0.35e-2 --0.325e-2

--pooling_sl_mag = 1.7e-2
--mask_mag = 0

if not(recpool_config_prefs.disable_pooling) and not(disable_pooling_losses) then
   --sl_mag = 0 -- this *SHOULD* be scaled by L1_scaling
   sl_mag = 0.33e-2 --now scaled by L1_scaling = 3    was: 1e-2 -- attempt to duplicate good run on 10/11
   --sl_mag = 0.025e-2 -- used in addition to group sparsity
else
   sl_mag = 3e-2 -- now scaled by L1_scaling = 3 was: 9e-2 --3e-2
end
pooling_rec_mag = 1 --0 --0.5
pooling_orig_rec_mag = 0 --1 --0.05 --1
--pooling_shrink_position_L2_mag = 0.1
--pooling_shrink_position_L2_mag = 0.01 --0.001
--pooling_shrink_position_L2_mag = 1e-3 --1e-4 --4e-3 --1e-3 --0.0001 --0.01 --0.005 --0
pooling_shrink_position_L2_mag = 1e-3 -- further tests with sqrt reconstruction
--pooling_shrink_position_L2_mag = 1e-2 -- straight L2 position, sqrt-sum-of-squares pooling
--pooling_shrink_position_L2_mag = 1e-4 --1e-6 -- straight L2 position, cube-root-sum-of-squares pooling
--pooling_shrink_position_L2_mag = 1e-6 -- straight L2 position, cube-root-sum-of-squares pooling
pooling_orig_position_L2_mag = 0 --0.005 --0.1
--local pooling_reconstruction_scaling = 3 --1.5 --2.5 --1.5 --0.85 --0.5 --0.25 -- straight L2 position, sqrt-sum-of-squares pooling
--local pooling_reconstruction_scaling = 40 --60000 --100000 -- straight L2 position, cube-root-sum-of-squares pooling
--local pooling_reconstruction_scaling = 20000  --200000 --100000 -- straight L2 position, cube-root-sum-of-squares pooling
--local pooling_reconstruction_scaling = 200 --140 --400 --180 --1400 --40 --140
local pooling_reconstruction_scaling = 400 -- further tests with sqrt reconstruction
if disable_pooling_losses then
   pooling_reconstruction_scaling = 0
   pooling_sl_mag = 0
   mask_mag = 0
end
pooling_rec_mag = pooling_reconstruction_scaling * pooling_rec_mag
pooling_orig_rec_mag = pooling_reconstruction_scaling * pooling_orig_rec_mag
pooling_shrink_position_L2_mag = pooling_reconstruction_scaling * pooling_shrink_position_L2_mag
pooling_orig_position_L2_mag = pooling_reconstruction_scaling * pooling_orig_position_L2_mag

-- GROUP SPARSITY TEST
rec_mag = 5 --4 --5 --4
if num_layers == 1 then
   if fe_layer_size == 200 then
      L1_scaling = 3 --4 --2.5 -- with square root L2 position loss
      --L1_scaling = 6 -- with classification loss added in
      --L1_scaling = 3 --2.5 --1.5 -- straight L2 position, sqrt-sum-of-squares pooling
      --L1_scaling = 2 --1.25 --6 --1 -- straight L2 position, cube-root-sum-of-squares pooling
      if p_layer_size == 200 then
	 L1_scaling = 3 * 3/4
      end
   else
      -- SHOULD SCALE INITIAL SPARSITY RATHER THAN L1 SCALING WHEN CHANGING THE NUMBER OF UNITS
      L1_scaling = 3 --3.1 --2 --3/math.sqrt(2) -- for use with 400 FE units
      mask_mag = mask_mag * math.sqrt(200/50) / math.sqrt(fe_layer_size/p_layer_size)
   end
elseif num_layers == 2 then
   -- TRY ONLY ADJUSTING THE LAYER 2 SCALING WHEN ADDING A SECOND LAYER!!!
   L1_scaling = 2.25 --1 --0.25 
else
   error('L1_scaling not specified for num_layers')
end

print('Using group sparsity scaling ' .. L1_scaling)

--L1_scaling = 2
L1_scaling_layer_2 = 0.2*2.25 --0.05 --0.1
pooling_rec_layer_2 = 0.2*1 --0.2 --0.5


--[[
rec_mag = 0
pooling_rec_mag = 0
pooling_orig_rec_mag = 0 
pooling_shrink_position_L2_mag = 0
pooling_orig_position_L2_mag = 0
sl_mag = 0
pooling_sl_mag = 0
mask_mag = 0
local L1_scaling = 0
local L1_scaling_layer_2 = 0
--]]

-- Correct classification of the last few examples are is learned very slowly when we turn up the regularizers, since as the classification improves, the regularization error becomes as large as the classification error, so corrections to the classification trade off against the sparsity and reconstruction quality.  
local lambdas = {ista_L2_reconstruction_lambda = rec_mag, ista_L1_lambda = sl_mag, pooling_L2_shrink_reconstruction_lambda = pooling_rec_mag, pooling_L2_orig_reconstruction_lambda = pooling_orig_rec_mag, pooling_L2_shrink_position_unit_lambda = pooling_shrink_position_L2_mag, pooling_L2_orig_position_unit_lambda = pooling_orig_position_L2_mag, pooling_output_cauchy_lambda = pooling_sl_mag, pooling_mask_cauchy_lambda = mask_mag} -- classification implicitly has a scaling constant of 1

-- FOR THE LOVE OF GOD!!!  DEBUG ONLY!!!  L1_scaling should also scale sl_mag, but this has been removed to reconstruct the good run on 10/11
--local lambdas_1 = {ista_L2_reconstruction_lambda = rec_mag, ista_L1_lambda = sl_mag, pooling_L2_shrink_reconstruction_lambda = pooling_rec_mag, pooling_L2_orig_reconstruction_lambda = pooling_orig_rec_mag, pooling_L2_shrink_position_unit_lambda = pooling_shrink_position_L2_mag, pooling_L2_orig_position_unit_lambda = pooling_orig_position_L2_mag, pooling_output_cauchy_lambda = L1_scaling * pooling_sl_mag, pooling_mask_cauchy_lambda = L1_scaling * mask_mag} -- classification implicitly has a scaling constant of 1
local lambdas_1 = {ista_L2_reconstruction_lambda = rec_mag, ista_L1_lambda = L1_scaling * sl_mag, pooling_L2_shrink_reconstruction_lambda = pooling_rec_mag, pooling_L2_orig_reconstruction_lambda = pooling_orig_rec_mag, pooling_L2_shrink_position_unit_lambda = pooling_shrink_position_L2_mag, pooling_L2_orig_position_unit_lambda = pooling_orig_position_L2_mag, pooling_output_cauchy_lambda = L1_scaling * pooling_sl_mag, pooling_mask_cauchy_lambda = L1_scaling * mask_mag} -- classification implicitly has a scaling constant of 1


-- NOTE THAT POOLING_MASK_CAUCHY_LAMBDA IS MUCH LARGER
local lambdas_2 = {ista_L2_reconstruction_lambda = pooling_rec_layer_2 * rec_mag, ista_L1_lambda = L1_scaling_layer_2 * sl_mag, pooling_L2_shrink_reconstruction_lambda = pooling_rec_layer_2 * pooling_rec_mag, pooling_L2_orig_reconstruction_lambda = pooling_rec_layer_2 * pooling_orig_rec_mag, pooling_L2_shrink_position_unit_lambda = pooling_rec_layer_2 * pooling_shrink_position_L2_mag, pooling_L2_orig_position_unit_lambda = pooling_rec_layer_2 * pooling_orig_position_L2_mag, pooling_output_cauchy_lambda = L1_scaling_layer_2 * pooling_sl_mag, pooling_mask_cauchy_lambda = L1_scaling_layer_2 * mask_mag} -- classification implicitly has a scaling constant of 1


-- targets a multiplied by layer_size to produce the final value, since the L1 loss increases linear with the number of units; it represents the desired value for each unit
local lagrange_multiplier_targets = {feature_extraction_target = 5e-3, pooling_target = 1e-3, mask_target = 0.25e-3} --{feature_extraction_lambda = 5e-2, pooling_lambda = 2e-2, mask_lambda = 1e-2} -- {feature_extraction_lambda = 1e-2, pooling_lambda = 5e-2, mask_lambda = 1e-1} -- {feature_extraction_lambda = 5e-3, pooling_lambda = 1e-1}
local lagrange_multiplier_targets_1 = {feature_extraction_target = 5e-3, pooling_target = 1e-3, mask_target = 0.25e-3}
local lagrange_multiplier_targets_2 = {feature_extraction_target = 5e-3, pooling_target = 1e-3, mask_target = 5e-3}
local lagrange_multiplier_learning_rate_scaling_factors = {feature_extraction_scaling_factor = 1e-1, pooling_scaling_factor = 1e-2, mask_scaling_factor = 1e-4} -- {feature_extraction_scaling_factor = 1e-1, pooling_scaling_factor = 2e-3, mask_scaling_factor = 1e-3}
local lagrange_multiplier_learning_rate_scaling_factors_1 = {feature_extraction_scaling_factor = 1e-1, pooling_scaling_factor = 1e-2, mask_scaling_factor = 0} --1e-4} 
local lagrange_multiplier_learning_rate_scaling_factors_2 = {feature_extraction_scaling_factor = 1e-1, pooling_scaling_factor = 1e-2, mask_scaling_factor = 0} --2e-6} 



for k,v in pairs(lambdas) do
   lambdas[k] = v * 1
end

local layer_size, layered_lambdas, layered_lagrange_multiplier_targets, layered_lagrange_multiplier_learning_rate_scaling_factors
if false and (num_layers == 1) then
   print(lambdas)
   layer_size = {28*28, 200, 50, 10}
   layered_lambdas = {lambdas}
   local this_layer_lagrange_multiplier_targets = {}
   this_layer_lagrange_multiplier_targets.feature_extraction_target = lagrange_multiplier_targets_1.feature_extraction_target * layer_size[2]
   this_layer_lagrange_multiplier_targets.pooling_target = lagrange_multiplier_targets_1.pooling_target * layer_size[3]
   this_layer_lagrange_multiplier_targets.mask_target = lagrange_multiplier_targets_1.mask_target * layer_size[2]
   layered_lagrange_multiplier_targets = {this_lagrange_multiplier_targets}
   layered_lagrange_multiplier_learning_rate_scaling_factors = {lagrange_multiplier_learning_rate_scaling_factors_1}
else
   print(lambdas_1, lambdas_2)
   layer_size = {28*28}
   layered_lambdas = {}
   layered_lagrange_multiplier_targets = {}
   layered_lagrange_multiplier_learning_rate_scaling_factors = {}
   for i = 1,num_layers do
      if i == 1 then
	 table.insert(layer_size, fe_layer_size) --200)
	 table.insert(layer_size, p_layer_size)
	 table.insert(layered_lambdas, lambdas_1)
      else
	 table.insert(layer_size, 100) --100
	 table.insert(layer_size, 25) --25
	 table.insert(layered_lambdas, lambdas_2)
      end
      
      local this_layer_lagrange_multiplier_targets = {}
      if i == 1 then
	 this_layer_lagrange_multiplier_targets.feature_extraction_target = lagrange_multiplier_targets_1.feature_extraction_target * layer_size[#layer_size - 1]
	 this_layer_lagrange_multiplier_targets.pooling_target = lagrange_multiplier_targets_1.pooling_target * layer_size[#layer_size]
	 this_layer_lagrange_multiplier_targets.mask_target = lagrange_multiplier_targets_1.mask_target * layer_size[#layer_size - 1] 
	 table.insert(layered_lagrange_multiplier_learning_rate_scaling_factors, lagrange_multiplier_learning_rate_scaling_factors_1)
      else
	 this_layer_lagrange_multiplier_targets.feature_extraction_target = lagrange_multiplier_targets_2.feature_extraction_target * layer_size[#layer_size - 1]
	 this_layer_lagrange_multiplier_targets.pooling_target = lagrange_multiplier_targets_2.pooling_target * layer_size[#layer_size]
	 this_layer_lagrange_multiplier_targets.mask_target = lagrange_multiplier_targets_2.mask_target * layer_size[#layer_size - 1] 
	 table.insert(layered_lagrange_multiplier_learning_rate_scaling_factors, lagrange_multiplier_learning_rate_scaling_factors_2)
      end
      table.insert(layered_lagrange_multiplier_targets, this_layer_lagrange_multiplier_targets)

   end
   table.insert(layer_size, 10) -- insert the classification output last
end



-- create the dataset
require 'mnist'
local data_set_size, data
if params.data_set == 'train' then
   data_set_size = (((params.full_test == 'full_train') or (params.full_test == 'full_test')) and 50000) or quick_train_epoch_size
   data = mnist.loadTrainSet(data_set_size, 'recpool_net') -- 'recpool_net' option ensures that the returned table contains elements data and labels, for which the __index method is overloaded.  
   --data = mnist.loadTrainSet(data_set_size, 'recpool_net_L2_classification') -- DEBUG ONLY!!! FOR THE LOVE OF GOD!!!
else
   data_set_size = (((params.full_test == 'full_train') or (params.full_test == 'full_test')) and 10000) or 5000
   if params.data_set == 'validation' then
      data = mnist.loadTrainSet(data_set_size, 'recpool_net', 50000)
   elseif params.data_set == 'test' then
      data = mnist.loadTestSet(data_set_size, 'recpool_net') -- 'recpool_net' option ensures that the returned table contains elements data and labels, for which the __index method is overloaded. 
   else
      error('requested data set ' .. params.data_set .. ' was not recognized')
   end
end

--Indexing labels returns an index, rather than a tensor
data:normalizeL2() -- normalize each example to have L2 norm equal to 1




local model = build_recpool_net(layer_size, layered_lambdas, 1, layered_lagrange_multiplier_targets, layered_lagrange_multiplier_learning_rate_scaling_factors, recpool_config_prefs, data) -- last argument is num_ista_iterations

-- option array for RecPoolTrainer
opt = {log_directory = params.log_directory, -- subdirectory in which to save/log experiments
   visualize = false, -- visualize input data and weights during training
   plot = false, -- live plot
   optimization = 'SGD', -- optimization method: SGD | ASGD | CG | LBFGS
   init_eval_counter = ((num_epochs_no_classification <= 1) and (200 * 50000 / math.max(1, desired_minibatch_size))) or 0,
   learning_rate = ((params.full_test == 'full_train') and full_train_learning_rate) or ((params.full_test == 'quick_train') and quick_train_learning_rate) or 
      (((params.full_test == 'full_test') or (params.full_test == 'quick_test')) and 0), --1e-3, -- learning rate at t=0
   batch_size = desired_minibatch_size, -- mini-batch size (0 = pure stochastic)
   learning_rate_decay = 5e-7 * math.max(1, desired_minibatch_size), -- learning rate decay is performed based upon the number of calls to SGD.  When using minibatches, we must increase the decay in proportion to the minibatch size to maintain parity based upon the number of datapoints examined
   weight_decay = 0, -- weight decay (SGD only)
   momentum = 0, -- momentum (SGD only)
   t0 = 1, -- start averaging at t0 (ASGD only), in number (?!?) of epochs -- WHAT DOES THIS MEAN?
   max_iter = 2 -- maximum nb of iterations for CG and LBFGS
}

print('Using opt.learning_rate = ' .. opt.learning_rate)

--torch.manualSeed(23827602) -- init random number generator.  Obviously, this should be taken from the clock when doing an actual run
torch.seed()


local track_criteria_outputs = not((params.full_test == 'full_train') or (params.full_test == 'full_test'))
local trainer = nn.RecPoolTrainer(model, opt, layered_lambdas, track_criteria_outputs) -- layered_lambdas is required for debugging purposes only


-- load parameters from file if desired
if params.load_file ~= '' then
   load_parameters(trainer:get_flattened_parameters(), params.load_file)
   --model:randomize_pooling()
   --model.module_list.classification_dictionary.weight:mul(10)
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


-- consider increasing learning rate when classification loss is disabled; otherwise, new features in the feature_extraction_dictionaries are discovered very slowly
model:reset_classification_lambda(0) -- SPARSIFYING LAMBDAS SHOULD REALLY BE TURNED UP WHEN THE CLASSIFICATION CRITERION IS DISABLED

for i = 1,num_epochs_no_classification do
   if (i % 20 == 1) and (i >= 1) then -- make sure to save the initial paramters, before any training occurs, to allow comparisons later
      save_parameters(trainer:get_flattened_parameters(), opt.log_directory, i) -- defined in display_recpool_net
   end

   trainer:train(data)
   --print('iterations so far: ' .. trainer.config.evalCounter)
   plot_filters(opt, i, model.filter_list, model.filter_enc_dec_list, model.filter_name_list)
   print('Effective learning rate decay is ' .. trainer.config.evalCounter * trainer.config.learningRateDecay)
end

-- reset lambdas to be closer to pure top-down fine-tuning and continue training
model:reset_classification_lambda(1) -- 0.2 seems to strike an even balance between reconstruction and classification
--trainer:reset_learning_rate(1e-3) -- potentially use faster learning rate for the unsupervised pretraining, then revert to a more careful learning rate for supervised training with the classification loss
--trainer.config.evalCounter = 0 -- reset counter for learning rate decay; this maintains consistency between full runs and runs initialized with an unsupervised-pretrained network
local perform_classifier_pretraining = false
local num_epochs_classification_pretraining = 7 -- 12
local num_epochs_classification_slow_burn_in = 10
local slow_burn_in_scale_factor = 0.1 -- 0.05

if perform_classifier_pretraining then
   trainer:reset_learning_rate(20 * (5000 / data_set_size) * opt.learning_rate) -- Do a few epochs of accelerated training to initialize the classification_dictionary
   print('resetting learning rate to ' .. 20 * (5000 / data_set_size) * opt.learning_rate)
   model:reset_learning_scale_factor(0) -- turn off learning for all ConstrainedLinear modules, leaving only the classification dictionary to be trained.  THIS DOES NOT WORK PROPERLY WITH PARAMETERIZED_SHRINK!!!
end

for i = 1+num_epochs_no_classification,num_epochs+num_epochs_no_classification do
   -- Train the entire network with a very low learning rate for a few epochs, to resolve mismatches between the sparse coding layer and the classification layer.  Remember that, since the classification dictionary is now so large, the backpropagated gradients are correspondingly scaled up
   if perform_classifier_pretraining and (i == num_epochs_classification_pretraining + num_epochs_no_classification) then
      --model:reset_learning_scale_factor(0.05) 
      --trainer:reset_learning_rate(opt.learning_rate) 

      model:reset_classification_lambda(0.1)
      model:reset_learning_scale_factor(1) 
      trainer:reset_learning_rate(slow_burn_in_scale_factor * opt.learning_rate) 
      print('resetting learning rate to ' .. slow_burn_in_scale_factor * opt.learning_rate)
   elseif perform_classifier_pretraining and (i == num_epochs_classification_slow_burn_in + num_epochs_classification_pretraining + num_epochs_no_classification) then
      --model:reset_learning_scale_factor(0.2) -- this is largely equivalent to the learning rate decay after 200 epochs, which we removed above
      model:reset_classification_lambda(0.1)
      local final_scale_factor = 5 --0.25 -- 0.15
      trainer:reset_learning_rate(final_scale_factor * opt.learning_rate) 
      print('resetting learning rate to ' .. final_scale_factor * opt.learning_rate)
   end
   
   if (i % 20 == 1) and (i > 1) then
      save_parameters(trainer:get_flattened_parameters(), opt.log_directory, i) -- defined in display_recpool_net
   end

   trainer:train(data)
   plot_filters(opt, i, model.filter_list, model.filter_enc_dec_list, model.filter_name_list)
   print('Effective learning rate decay is ' .. trainer.config.evalCounter * trainer.config.learningRateDecay)
end







