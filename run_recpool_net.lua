dofile('build_recpool_net.lua')
dofile('train_recpool_net.lua')

local layer_size = {28*28, 200, 50, 10}
local target = math.random(layer_size[4])
local lambdas = {ista_L2_reconstruction_lambda = 0.1, ista_L1_lambda = 0.01, pooling_L2_reconstruction_lambda = 0.1, pooling_L2_position_unit_lambda = 0.01, pooling_output_cauchy_lambda = 0.01, pooling_mask_cauchy_lambda = 0.01} -- classification implicitly has a scaling constant of 1

-- build_recpool_net also returns: criteria_list, encoding_dictionary, decoding_dictionary, encoding_pooling_dictionary, decoding_pooling_dictionary, classification_dictionary, explaining_away, shrink, explaining_away_copies, shrink_copies
local model = build_recpool_net(layer_size, lambdas, 5) -- last argument is num_ista_iterations

-- option array for RecPoolTrainer
opt = {save = 'results', -- subdirectory in which to save/log experiments
   visualize = false, -- visualize input data and weights during training
   plot = false, -- live plot
   optimization = 'SGD', -- optimization method: SGD | ASGD | CG | LBFGS
   learning_rate = 1e-3, -- learning rate at t=0
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

local trainer = RecPoolTrainer(model, opt)

num_epochs = 100
for i = 1,num_epochs do
   trainer:train(data)
end
