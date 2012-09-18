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
cmd:option('-num_layers','2', 'number of reconstruction pooling layers in the network')

local params = cmd:parse(arg)


local sl_mag = 5e-2 --1.5e-2 --5e-2 --4e-2 --0.5e-2 --5e-2 --2e-3 --5e-3 -- 1e-2 -- 2e-2 --5e-2 --1e-2 --1e-1 --5e-2 -- sparsifying l1 magnitude (4e-2)
local rec_mag = 4 -- reconstruction L2 magnitude
local pooling_rec_mag = 1 --4 --2 --10e-1 --0 --0.25 --8 -- pooling reconstruction L2 magnitude
local pooling_orig_rec_mag = 0 --2 --10e-1 --0 --4 --8 -- pooling reconstruction L2 magnitude
local pooling_position_L2_mag = 1 --4 --4 --20e-1 --0 --1.5e-2 --16e-2 --8e-2 --5e-2 --4e-1 -- 1e-1 --5e-2
local pooling_sl_mag = 1e-2 --0 --0.75e-2 --8e-2 --4e-2 --2.5e-2 --5e-2 --4e-1 -- 1e-1 --5e-2
local mask_mag = 0.1e-2 --0 --0.75e-2 --0.5e-2 --0.75e-2 --8e-2 --4e-2 --2.5e-2 --1e-1 --5e-2

--[[
pooling_rec_mag = 0
pooling_position_L2_mag = 0
--]]

---[[
sl_mag = 0 --1e-2

-- no classification loss
--pooling_sl_mag = 0.4e-2 --0.25e-2 --2e-2 --5e-2
--local mask_mag = 0.5e-2 --0.5e-2 --0 --0.75e-2 --0.5e-2 --0.75e-2 --8e-2 --4e-2 --2.5e-2 --1e-1 --5e-2

-- with classification loss
-- one layers
--pooling_sl_mag = 0.25e-2 --0.25e-2 --2e-2 --5e-2
-- two layers
pooling_sl_mag = 0.15e-2 --0.25e-2 --2e-2 --5e-2
local mask_mag = 0.4e-2 --0.5e-2 --0 --0.75e-2 --0.5e-2 --0.75e-2 --8e-2 --4e-2 --2.5e-2 --1e-1 --5e-2

local pooling_rec_mag = 20 -- this can be scaled up freely, and only affects alpha; for some reason, though, when I make it very large, I get nans
local pooling_position_L2_mag = 0.1 -- this can be scaled down to reduce alpha, but will simultaneously scale down the pooling reconstruction and position position unit losses.  It should not be too large, since the pooling unit loss can be made small by making the pooling reconstruction anticorrelated with the input
local num_ista_iterations = 3
local L1_scaling = 1.5 -- these shouldn't be used; these parameters are for a single hidden layer
local L1_scaling_layer_2 = 0.05
--]]

--[[
rec_mag = 4 --1
pooling_rec_mag = 1 --0.25
pooling_position_L2_mag = 0.1
sl_mag = 0.5e-2 --1e-2 -- 0
pooling_sl_mag = 1e-1 --5e-2 -- remember that this is effectively scaled down by a factor for 4 relative to sl_mag, since there are fewer pooled elements
mask_mag = 2e-3 --1e-3
local L1_scaling = 1 -- 2 hidden layers -- proper scaling of the L1 losses is CRITICAL for good performance!  If it's too low, training is ridiculously slow!
local L1_scaling_layer_2 = 0.1
--local L1_scaling = 0.1 -- 3 hidden layers -- proper scaling of the L1 losses is CRITICAL for good performance!  If it's too low, training is ridiculously slow!
--local L1_scaling = 0.1 -- 4 hidden layers -- proper scaling of the L1 losses is CRITICAL for good performance!  If it's too low, training is ridiculously slow!
local num_ista_iterations = 3
--]]

--[[
rec_mag = 0
pooling_rec_mag = 0
pooling_orig_rec_mag = 0 
pooling_position_L2_mag = 0
sl_mag = 0
pooling_sl_mag = 0
mask_mag = 0
local L1_scaling = 0
local L1_scaling_layer_2 = 0
local num_ista_iterations = 3
--]]

-- Correct classification of the last few examples are is learned very slowly when we turn up the regularizers, since as the classification improves, the regularization error becomes as large as the classification error, so corrections to the classification trade off against the sparsity and reconstruction quality.  
local lambdas = {ista_L2_reconstruction_lambda = rec_mag, ista_L1_lambda = sl_mag, pooling_L2_shrink_reconstruction_lambda = pooling_rec_mag, pooling_L2_orig_reconstruction_lambda = pooling_orig_rec_mag, pooling_L2_position_unit_lambda = pooling_position_L2_mag, pooling_output_cauchy_lambda = pooling_sl_mag, pooling_mask_cauchy_lambda = mask_mag} -- classification implicitly has a scaling constant of 1

-- reduce lambda scaling to 0.15; still too sparse
local lambdas_1 = {ista_L2_reconstruction_lambda = rec_mag, ista_L1_lambda = L1_scaling * sl_mag, pooling_L2_shrink_reconstruction_lambda = pooling_rec_mag, pooling_L2_orig_reconstruction_lambda = pooling_orig_rec_mag, pooling_L2_position_unit_lambda = pooling_position_L2_mag, pooling_output_cauchy_lambda = L1_scaling * pooling_sl_mag, pooling_mask_cauchy_lambda = L1_scaling * mask_mag} -- classification implicitly has a scaling constant of 1


-- NOTE THAT POOLING_MASK_CAUCHY_LAMBDA IS MUCH LARGER
local lambdas_2 = {ista_L2_reconstruction_lambda = rec_mag, ista_L1_lambda = L1_scaling_layer_2 * sl_mag, pooling_L2_shrink_reconstruction_lambda = pooling_rec_mag, pooling_L2_orig_reconstruction_lambda = pooling_orig_rec_mag, pooling_L2_position_unit_lambda = pooling_position_L2_mag, pooling_output_cauchy_lambda = L1_scaling_layer_2 * pooling_sl_mag, pooling_mask_cauchy_lambda = L1_scaling_layer_2 * mask_mag} -- classification implicitly has a scaling constant of 1


local lagrange_multiplier_targets = {feature_extraction_lambda = 1e-2, pooling_lambda = 2e-2, mask_lambda = 1e-2} --{feature_extraction_lambda = 5e-2, pooling_lambda = 2e-2, mask_lambda = 1e-2} -- {feature_extraction_lambda = 1e-2, pooling_lambda = 5e-2, mask_lambda = 1e-1} -- {feature_extraction_lambda = 5e-3, pooling_lambda = 1e-1}
local lagrange_multiplier_learning_rate_scaling_factors = {feature_extraction_scaling_factor = 1e-1, pooling_scaling_factor = 1e-2, mask_scaling_factor = 1e-2} -- {feature_extraction_scaling_factor = 1e-1, pooling_scaling_factor = 2e-3, mask_scaling_factor = 1e-3}

for k,v in pairs(lambdas) do
   lambdas[k] = v * 1
end

local layer_size, layered_lambdas, layered_lagrange_multiplier_targets, layered_lagrange_multiplier_learning_rate_scaling_factors
local num_layers = tonumber(params.num_layers)
if num_layers == 1 then
   print(lambdas)
   layer_size = {28*28, 200, 50, 10}
   layered_lambdas = {lambdas}
   layered_lagrange_multiplier_targets = {lagrange_multiplier_targets}
   layered_lagrange_multiplier_learning_rate_scaling_factors = {lagrange_multiplier_learning_rate_scaling_factors}
else
   print(lambdas_1, lambdas_2)
   layer_size = {28*28, 200, 50}
   layered_lambdas = {lambdas_1}
   layered_lagrange_multiplier_targets = {lagrange_multiplier_targets}
   layered_lagrange_multiplier_learning_rate_scaling_factors = {lagrange_multiplier_learning_rate_scaling_factors}
   for i = 1,num_layers-1 do
      table.insert(layer_size, 100)
      table.insert(layer_size, 50)
      table.insert(layered_lambdas, lambdas_2)
      table.insert(layered_lagrange_multiplier_targets, lagrange_multiplier_targets)
      table.insert(layered_lagrange_multiplier_learning_rate_scaling_factors, lagrange_multiplier_learning_rate_scaling_factors)
   end
   table.insert(layer_size, 10) -- insert the classification output last
end

local model = build_recpool_net(layer_size, layered_lambdas, layered_lagrange_multiplier_targets, layered_lagrange_multiplier_learning_rate_scaling_factors, num_ista_iterations) -- last argument is num_ista_iterations

-- load parameters from file if desired
if params.load_file ~= '' then
   load_parameters(model, params.load_file)
end

-- option array for RecPoolTrainer
opt = {log_directory = params.log_directory, -- subdirectory in which to save/log experiments
   visualize = false, -- visualize input data and weights during training
   plot = false, -- live plot
   optimization = 'SGD', -- optimization method: SGD | ASGD | CG | LBFGS
   learning_rate = 5e-3, --2e-3, --5e-3, --1e-3, -- learning rate at t=0
   batch_size = 1, -- mini-batch size (1 = pure stochastic)
   weight_decay = 0, -- weight decay (SGD only)
   momentum = 0, -- momentum (SGD only)
   t0 = 1, -- start averaging at t0 (ASGD only), in number (?!?) of epochs -- WHAT DOES THIS MEAN?
   max_iter = 2 -- maximum nb of iterations for CG and LBFGS
}

torch.manualSeed(10934783) -- init random number generator.  Obviously, this should be taken from the clock when doing an actual run

-- create the dataset
require 'mnist'
data = mnist.loadTrainSet(50000, 'recpool_net') -- 'recpool_net' option ensures that the returned table contains elements data and labels, for which the __index method is overloaded.  Indexing labels returns an index, rather than a tensor
data:normalizeL2() -- normalize each example to have L2 norm equal to 1


local trainer = nn.RecPoolTrainer(model, opt, layered_lambdas) -- layered_lambdas is required for debugging purposes only
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


num_epochs = 100
for i = 1,num_epochs do
   trainer:train(data)
   plot_filters(opt, i, model.filter_list, model.filter_enc_dec_list, model.filter_name_list)

   if i % 10 == 1 then
      save_parameters(model, opt.log_directory, i) -- defined in display_recpool_net
   end
end

-- reset lambdas to be closer to pure top-down fine-tuning and continue training







