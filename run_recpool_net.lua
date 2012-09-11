dofile('init.lua')
dofile('build_recpool_net.lua')
dofile('train_recpool_net.lua')
dofile('display_recpool_net.lua')


-- reconstructions seem too good, with too sparse internal representations, to be consistent with the reported decoding dictionaries
-- encoding and decoding pooling dictionaries seem to be almost identical.  Neither obviously map many inputs to a single output

local layer_size = {28*28, 200, 50, 10}
--local layer_size = {28*28, 100, 50, 100, 50, 10}
local target = math.random(layer_size[4])
--[[
local sl_mag = 4e-3 -- sparsifying l1 magnitude (2e-3)
local rec_mag = 1e-1 -- reconstruction L2 magnitude
local mask_mag = 1e-4
--]]

--local sl_mag, rec_mag, pooling_rec_mag, pooling_sl_mag, mask_mag = 0,0,0,0,0

local sl_mag = 5e-2 --1.5e-2 --5e-2 --4e-2 --0.5e-2 --5e-2 --2e-3 --5e-3 -- 1e-2 -- 2e-2 --5e-2 --1e-2 --1e-1 --5e-2 -- sparsifying l1 magnitude (4e-2)
local rec_mag = 4 -- reconstruction L2 magnitude
local pooling_rec_mag = 4 --2 --10e-1 --0 --0.25 --8 -- pooling reconstruction L2 magnitude
local pooling_orig_rec_mag = 0 --2 --10e-1 --0 --4 --8 -- pooling reconstruction L2 magnitude
local pooling_position_L2_mag = 4 --4 --20e-1 --0 --1.5e-2 --16e-2 --8e-2 --5e-2 --4e-1 -- 1e-1 --5e-2
local pooling_sl_mag = 1e-2 --0 --0.75e-2 --8e-2 --4e-2 --2.5e-2 --5e-2 --4e-1 -- 1e-1 --5e-2
local mask_mag = 0.1e-2 --0 --0.75e-2 --0.5e-2 --0.75e-2 --8e-2 --4e-2 --2.5e-2 --1e-1 --5e-2


-- Correct classification of the last few examples are is learned very slowly when we turn up the regularizers, since as the classification improves, the regularization error becomes as large as the classification error, so corrections to the classification trade off against the sparsity and reconstruction quality.  
local lambdas = {ista_L2_reconstruction_lambda = rec_mag, ista_L1_lambda = sl_mag, pooling_L2_shrink_reconstruction_lambda = pooling_rec_mag, pooling_L2_orig_reconstruction_lambda = pooling_orig_rec_mag, pooling_L2_position_unit_lambda = pooling_position_L2_mag, pooling_output_cauchy_lambda = pooling_sl_mag, pooling_mask_cauchy_lambda = mask_mag} -- classification implicitly has a scaling constant of 1

local lagrange_multiplier_targets = {feature_extraction_lambda = 1e-2, pooling_lambda = 2e-2, mask_lambda = 1e-2} -- {feature_extraction_lambda = 1e-2, pooling_lambda = 5e-2, mask_lambda = 1e-1} -- {feature_extraction_lambda = 5e-3, pooling_lambda = 1e-1}
local lagrange_multiplier_learning_rate_scaling_factors = {feature_extraction_scaling_factor = 1e-2, pooling_scaling_factor = 1e-2, mask_scaling_factor = 1e-2} -- {feature_extraction_scaling_factor = 1e-1, pooling_scaling_factor = 2e-3, mask_scaling_factor = 1e-3}

for k,v in pairs(lambdas) do
   lambdas[k] = v * 1
end
print(lambdas)


-- build_recpool_net also returns: criteria_list, encoding_dictionary, decoding_dictionary, encoding_pooling_dictionary, decoding_pooling_dictionary, classification_dictionary, explaining_away, shrink, explaining_away_copies, shrink_copies
local model = build_recpool_net(layer_size, lambdas, lagrange_multiplier_targets, lagrange_multiplier_learning_rate_scaling_factors, 5) -- last argument is num_ista_iterations

-- option array for RecPoolTrainer
opt = {log_directory = 'recpool_results', -- subdirectory in which to save/log experiments
   visualize = false, -- visualize input data and weights during training
   plot = false, -- live plot
   optimization = 'SGD', -- optimization method: SGD | ASGD | CG | LBFGS
   learning_rate = 5e-3, --1e-3, -- learning rate at t=0
   batch_size = 1, -- mini-batch size (1 = pure stochastic)
   weight_decay = 0, -- weight decay (SGD only)
   momentum = 0, -- momentum (SGD only)
   t0 = 1, -- start averaging at t0 (ASGD only), in number (?!?) of epochs -- WHAT DOES THIS MEAN?
   max_iter = 2 -- maximum nb of iterations for CG and LBFGS
}

torch.manualSeed(10934783) -- init random number generator.  Obviously, this should be taken from the clock when doing an actual run

-- create the dataset
require 'mnist'
data = mnist.loadTrainSet(5000, 'recpool_net') -- 'recpool_net' option ensures that the returned table contains elements data and labels, for which the __index method is overloaded.  Indexing labels returns an index, rather than a tensor
data:normalizeL2() -- normalize each example to have L2 norm equal to 1


local trainer = nn.RecPoolTrainer(model, opt)
os.execute('rm -rf ' .. opt.log_directory)
os.execute('mkdir -p ' .. opt.log_directory)

for i = 1,#model.shrink_copies do
   print('checking sharing for shrink copy ' .. i)
   if model.shrink.shrink_val:storage() ~= model.shrink_copies[i].shrink_val:storage() then
      print('Before training, shrink storage is not the same as the shrink copy ' .. i .. ' storage!!!')
      io.read()
   end
end


num_epochs = 100
for i = 1,num_epochs do
   trainer:train(data)
   plot_filters(opt, i, model.filter_list, model.filter_enc_dec_list, model.filter_name_list)
end
