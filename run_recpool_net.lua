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

local params = cmd:parse(arg)

local sl_mag = 5e-2 --1.5e-2 --5e-2 --4e-2 --0.5e-2 --5e-2 --2e-3 --5e-3 -- 1e-2 -- 2e-2 --5e-2 --1e-2 --1e-1 --5e-2 -- sparsifying l1 magnitude (4e-2)
local rec_mag = 4 -- reconstruction L2 magnitude
local pooling_rec_mag = 1 --4 --2 --10e-1 --0 --0.25 --8 -- pooling reconstruction L2 magnitude
local pooling_orig_rec_mag = 0 --2 --10e-1 --0 --4 --8 -- pooling reconstruction L2 magnitude
local pooling_shrink_position_L2_mag = 1 --4 --4 --20e-1 --0 --1.5e-2 --16e-2 --8e-2 --5e-2 --4e-1 -- 1e-1 --5e-2
local pooling_orig_position_L2_mag = 0
local pooling_sl_mag = 1e-2 --0 --0.75e-2 --8e-2 --4e-2 --2.5e-2 --5e-2 --4e-1 -- 1e-1 --5e-2
local mask_mag = 0.1e-2 --0 --0.75e-2 --0.5e-2 --0.75e-2 --8e-2 --4e-2 --2.5e-2 --1e-1 --5e-2


---[[
-- no classification loss
--pooling_sl_mag = 0.4e-2 --0.25e-2 --2e-2 --5e-2
--local mask_mag = 0.5e-2 --0.5e-2 --0 --0.75e-2 --0.5e-2 --0.75e-2 --8e-2 --4e-2 --2.5e-2 --1e-1 --5e-2

-- with classification loss
-- one layers
--pooling_sl_mag = 0.25e-2 --0.25e-2 --2e-2 --5e-2
-- two layers (CURRENTLY USED)
pooling_sl_mag = 0.5e-2 --0.9e-2 --0.5e-2 --0.15e-2 --0.25e-2 --2e-2 --5e-2 -- keep in mind that there are four times as many mask outputs as pooling outputs in the first layer -- also remember that the columns of decoding_pooling_dictionary are normalized to be the square root of the pooling factor.  However, before training, this just ensures that all decoding projections have a magnitude of one
mask_mag = 0.3e-2 --0.2e-2 --0.3e-2 --0.4e-2 --0.5e-2 --0 --0.75e-2 --0.5e-2 --0.75e-2 --8e-2 --4e-2 --2.5e-2 --1e-1 --5e-2

-- experiment to train all feature extraciton filters
--pooling_sl_mag = 0 --0.5e-2 
--mask_mag = 0.4e-2 --0.2e-2 


local pooling_rec_mag = 20 -- this can be scaled up freely, and only affects alpha; for some reason, though, when I make it very large, I get nans
local pooling_shrink_position_L2_mag = 0.1 -- this can be scaled down to reduce alpha, but will simultaneously scale down the pooling reconstruction and position position unit losses.  It should not be too large, since the pooling unit loss can be made small by making the pooling reconstruction anticorrelated with the input
local num_ista_iterations = 5 --3 --7 --3
local L1_scaling = 1.0 --1.5 --0.001 --0.4 --1.5 --1.2 --1.5 
local L1_scaling_layer_2 = 0.3 --0.05
--]]

---[[
--sl_mag = 10e-2 --80e-2 --1e-2 --2e-2 --5e-2
sl_mag = 0
--sl_mag = 4e-2
--pooling_sl_mag = 0
--mask_mag = 0
rec_mag = 2
pooling_rec_mag = 1 --0 --0.5
pooling_orig_rec_mag = 0 --1 --0.05 --1
--pooling_shrink_position_L2_mag = 0.1
--pooling_shrink_position_L2_mag = 0.01 --0.001
pooling_shrink_position_L2_mag = 1e-3 --1e-3 --0.0001 --0.01 --0.005 --0
pooling_orig_position_L2_mag = 0 --0.005 --0.1
--local pooling_reconstruction_scaling = 1.5 --2.5 --1.5 --0.85 --0.5 --0.25
local pooling_reconstruction_scaling = 140
pooling_rec_mag = pooling_reconstruction_scaling * pooling_rec_mag
pooling_orig_rec_mag = pooling_reconstruction_scaling * pooling_orig_rec_mag
pooling_shrink_position_L2_mag = pooling_reconstruction_scaling * pooling_shrink_position_L2_mag
pooling_orig_position_L2_mag = pooling_reconstruction_scaling * pooling_orig_position_L2_mag

-- when not using pooling reconstruction, 4 seems too small; 8 seems too large
L1_scaling = 2 --1.5 --1.0 --0.5 --1.5 --5 --2.5 --2 --0.4 --0.1 --0.75
L1_scaling_layer_2 = 0.125 --0.06 --0.12 --0.03
--]]

---[[
-- GROUP SPARSITY TEST
rec_mag = 5 --4 --5 --4
--L1_scaling = 0.25 --5 --3 --0.5 --1 --7.5 --5.5 --7.5 --6 is not too large; 9 is too large
L1_scaling = 4
--]]

--[[
rec_mag = 0
pooling_rec_mag = 0
pooling_orig_rec_mag = 0 
pooling_shrink_position_L2_mag = 0
sl_mag = 0
pooling_sl_mag = 0
mask_mag = 0
local L1_scaling = 0
local L1_scaling_layer_2 = 0
local num_ista_iterations = 3
--]]

-- Correct classification of the last few examples are is learned very slowly when we turn up the regularizers, since as the classification improves, the regularization error becomes as large as the classification error, so corrections to the classification trade off against the sparsity and reconstruction quality.  
local lambdas = {ista_L2_reconstruction_lambda = rec_mag, ista_L1_lambda = sl_mag, pooling_L2_shrink_reconstruction_lambda = pooling_rec_mag, pooling_L2_orig_reconstruction_lambda = pooling_orig_rec_mag, pooling_L2_shrink_position_unit_lambda = pooling_shrink_position_L2_mag, pooling_L2_orig_position_unit_lambda = pooling_orig_position_L2_mag, pooling_output_cauchy_lambda = pooling_sl_mag, pooling_mask_cauchy_lambda = mask_mag} -- classification implicitly has a scaling constant of 1

-- reduce lambda scaling to 0.15; still too sparse
local lambdas_1 = {ista_L2_reconstruction_lambda = rec_mag, ista_L1_lambda = sl_mag, pooling_L2_shrink_reconstruction_lambda = pooling_rec_mag, pooling_L2_orig_reconstruction_lambda = pooling_orig_rec_mag, pooling_L2_shrink_position_unit_lambda = pooling_shrink_position_L2_mag, pooling_L2_orig_position_unit_lambda = pooling_orig_position_L2_mag, pooling_output_cauchy_lambda = L1_scaling * pooling_sl_mag, pooling_mask_cauchy_lambda = L1_scaling * mask_mag} -- classification implicitly has a scaling constant of 1


-- NOTE THAT POOLING_MASK_CAUCHY_LAMBDA IS MUCH LARGER
local lambdas_2 = {ista_L2_reconstruction_lambda = rec_mag, ista_L1_lambda = L1_scaling_layer_2 * sl_mag, pooling_L2_shrink_reconstruction_lambda = pooling_rec_mag, pooling_L2_orig_reconstruction_lambda = pooling_orig_rec_mag, pooling_L2_shrink_position_unit_lambda = pooling_shrink_position_L2_mag, pooling_L2_orig_position_unit_lambda = pooling_orig_position_L2_mag, pooling_output_cauchy_lambda = L1_scaling_layer_2 * pooling_sl_mag, pooling_mask_cauchy_lambda = L1_scaling_layer_2 * mask_mag} -- classification implicitly has a scaling constant of 1


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
local num_layers = tonumber(params.num_layers)
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
	 table.insert(layer_size, 200)
	 table.insert(layer_size, 50)
	 table.insert(layered_lambdas, lambdas_1)
      else
	 table.insert(layer_size, 100)
	 table.insert(layer_size, 50)
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
   data_set_size = (((params.full_test == 'full_train') or (params.full_test == 'full_test')) and 50000) or 5000
   data = mnist.loadTrainSet(data_set_size, 'recpool_net') -- 'recpool_net' option ensures that the returned table contains elements data and labels, for which the __index method is overloaded.  
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




local model = build_recpool_net(layer_size, layered_lambdas, 1, layered_lagrange_multiplier_targets, layered_lagrange_multiplier_learning_rate_scaling_factors, num_ista_iterations, data) -- last argument is num_ista_iterations

-- option array for RecPoolTrainer
opt = {log_directory = params.log_directory, -- subdirectory in which to save/log experiments
   visualize = false, -- visualize input data and weights during training
   plot = false, -- live plot
   optimization = 'SGD', -- optimization method: SGD | ASGD | CG | LBFGS
   learning_rate = ((params.full_test == 'full_train') and 2e-3) or ((params.full_test == 'quick_train') and 5e-3) or 
      (((params.full_test == 'full_test') or (params.full_test == 'quick_test')) and 0), --1e-3, -- learning rate at t=0
   batch_size = 1, -- mini-batch size (1 = pure stochastic)
   weight_decay = 0, -- weight decay (SGD only)
   momentum = 0, -- momentum (SGD only)
   t0 = 1, -- start averaging at t0 (ASGD only), in number (?!?) of epochs -- WHAT DOES THIS MEAN?
   max_iter = 2 -- maximum nb of iterations for CG and LBFGS
}

print('Using opt.learning_rate = ' .. opt.learning_rate)

torch.manualSeed(10934783) -- init random number generator.  Obviously, this should be taken from the clock when doing an actual run


local trainer = nn.RecPoolTrainer(model, opt, layered_lambdas) -- layered_lambdas is required for debugging purposes only

-- load parameters from file if desired
if params.load_file ~= '' then
   load_parameters(trainer:get_flattened_parameters(), params.load_file)
   --model:randomize_pooling()
end

os.execute('rm -rf ' .. opt.log_directory)
os.execute('mkdir -p ' .. opt.log_directory)

-- confirm that shrink parameters are properly shared
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

-- consider increasing learning rate when classification loss is disabled; otherwise, new features in the feature_extraction_dictionaries are discovered very slowly
model:reset_classification_lambda(0) -- SPARSIFYING LAMBDAS SHOULD REALLY BE TURNED UP WHEN THE CLASSIFICATION CRITERION IS DISABLED
num_epochs_no_classification = 201
for i = 1,num_epochs_no_classification do
   trainer:train(data)
   plot_filters(opt, i, model.filter_list, model.filter_enc_dec_list, model.filter_name_list)

   if (i % 10 == 1) and (i > 1) then
      save_parameters(trainer:get_flattened_parameters(), opt.log_directory, i) -- defined in display_recpool_net
   end
end

-- reset lambdas to be closer to pure top-down fine-tuning and continue training
model:reset_classification_lambda(1)
num_epochs = 500
for i = 1+num_epochs_no_classification,num_epochs+num_epochs_no_classification do
   trainer:train(data)
   plot_filters(opt, i, model.filter_list, model.filter_enc_dec_list, model.filter_name_list)

   if (i % 10 == 1) and (i > 1) then
      save_parameters(trainer:get_flattened_parameters(), opt.log_directory, i) -- defined in display_recpool_net
   end
end







