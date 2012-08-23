require 'torch'
require 'nn'
require 'kex'

DEBUG_shrink = false --true -- don't require that the shrink value be non-negative, to facilitate the comparison of backpropagated gradients to forward-propagated parameter perturbations
DEBUG_L2 = false
DEBUG_L1 = false
DEBUG_OUTPUT = false

-- the input is a table of three elments: the subject of the shrink operation z [1], the transformed input W*x [2], and the untransformed input x [3]
-- the output is a table of three elements: the subject of the shrink operation z [1], the transformed input W*x [2], and the untransformed input x [3]
local function build_ISTA_iteration(base_explaining_away, base_shrink, layer_size)
   local ista_seq = nn.Sequential()
   local explaining_away = nn.Linear(layer_size[2], layer_size[2])
   local shrink = nn.ParameterizedShrink(layer_size[2], true, DEBUG_shrink) -- EFFICIENCY NOTE: when using non-negative units this could be accomplished more efficiently using an unparameterized, one-sided rectification, just like Glorot, Bordes, and Bengio, along with a non-positive bias in the inverse_dictionary.  However, both nn.SoftShrink and the shrinkage utility method implemented in kex are two-sided.
   
   explaining_away:share(base_explaining_away, 'weight', 'bias')
   shrink:share(base_shrink, 'shrink_val')
   
   local explaining_away_parallel = nn.ParallelTable() --subtract out the explained input: S*z
   explaining_away_parallel:add(explaining_away)
   explaining_away_parallel:add(nn.Identity()) -- preserve W*x [2]
   explaining_away_parallel:add(nn.Identity()) -- preserve x [3]
   ista_seq:add(explaining_away_parallel)
   
   local add_in_WX = nn.Sequential()
   add_in_WX:add(nn.SelectTable{1,2})
   add_in_WX:add(nn.CAddTable())
   local combined_add_in_WX = nn.ParallelDistributingTable() -- add the transformed input W*x back in
   combined_add_in_WX:add(add_in_WX)
   combined_add_in_WX:add(nn.SelectTable{2}) -- preserve W*x
   combined_add_in_WX:add(nn.SelectTable{3}) -- preserve x
   ista_seq:add(combined_add_in_WX)

   local shrink_parallel = nn.ParallelTable() --shrink W*x + S*z
   shrink_parallel:add(shrink)
   shrink_parallel:add(nn.Identity()) -- preserve W*x [2]
   shrink_parallel:add(nn.Identity()) -- preserve x [3]
   ista_seq:add(shrink_parallel)
   
   return ista_seq, explaining_away, shrink
end

-- the input is a table of three elments: the subject of the shrink operation z [1], the transformed input W*x [2], and the untransformed input x [3]
-- the output is a table of one element: the subject of the shrink operation z [1]
local function build_L2_reconstruction_loss(L2_lambda, forward_dictionary, criteria_list)
   print('L2_rec_loss: L2_lambda is ', L2_lambda)
   io.read()

   local L2_seq = nn.Sequential()
   L2_seq:add(nn.CopyTable(1,2)) -- now: the subject of the shrink operation z [1, 2], the transformed input W*x [3], and the untransformed input x [4]
   
   local reconstruct_input = nn.ParallelTable() -- reconstruct the input from the shrunk code z
   reconstruct_input:add(nn.Identity()) -- preserve the subject of the shrink operation z [1]
   reconstruct_input:add(forward_dictionary) -- reconstruct the input D*z from the shrunk code z [2]
   reconstruct_input:add(nn.Identity()) -- preserve the transformed input W*x [3]
   reconstruct_input:add(nn.Identity()) -- preserve the untransformed input x [4]
   L2_seq:add(reconstruct_input) -- now: the subject of the shrink operation z [1], the reconstructed input D*z [2], the transformed input W*x [3], and the untransformed input x [4]
   
   local combined_loss = nn.ParallelDistributingTable() -- calculate the MSE between the reconstructed input D*z [2] and the untransformed input x [4]; only pass on the shrunk code z [1]
   combined_loss:add(nn.SelectTable{1}) -- throw away all streams but the shrunk code z [1]; the result is a table with a single entry
   
   local L2_loss_seq = nn.Sequential() -- calculate the MSE between the reconstructed input D*z [2] and the untransformed input x [4]
   L2_loss_seq:add(nn.SelectTable{2,4})
   if DEBUG_L2 then
      local sequential_zero = nn.Sequential()
      local parallel_zero = nn.ParallelTable() 
      parallel_zero:add(nn.ZeroModule())
      parallel_zero:add(nn.ZeroModule())
      sequential_zero:add(parallel_zero)
      sequential_zero:add(nn.SelectTable{1}) -- a SelectTable is necessary to ensure that the module outputs a single nil, which is ignored by the ParallelDistributingTable, rather than an empty table (i.e., a table of nils), which ParallelDistributingTable incorrectly passes onto its output
      L2_loss_seq:add(sequential_zero) 
   else
      local effective_L2_loss_function = nn.L2Cost(L2_lambda)
      L2_loss_seq:add(effective_L2_loss_function) -- using this instead of MSECriterion fed through CriterionTable ensures that the gradient is propagated to both inputs
      L2_loss_seq:add(nn.Ignore()) -- don't pass the L2 loss value onto the rest of the network
      table.insert(criteria_list, effective_L2_loss_function)   
      print('inserting L2 loss into criteria list, resulting in ' .. #criteria_list .. ' entries')
   end
   combined_loss:add(L2_loss_seq)
   L2_seq:add(combined_loss) 
   -- rather than using nn.Ignore on the output of the criterion, we could use a SelectTable{1} without a ParallelDistributingTable, which would output a tensor in the forward direction, and send a nil as gradOutput to the criterion in the backwards direction

   return L2_seq
end


-- the input is a table of one element: the subject of the shrink operation z [1]
-- the output is a table of one element: the subject of the shrink operation z [1]
local function build_L1_loss(L1_lambda, layer_size, criteria_list)
   local L1_seq = nn.Sequential()
   L1_seq:add(nn.CopyTable(1, 2)) -- split into the output to the rest of the chain [1] and the output to the L1 norm [2]
   
   local apply_L1_norm = nn.ParallelDistributingTable()
   apply_L1_norm:add(nn.SelectTable{1}) -- pass the code [1] through unchanged for further processing
   
   local scaled_L1_norm = nn.Sequential() -- scale the code copy [2], calculate its L1 norm, and throw away the output
   scaled_L1_norm:add(nn.SelectTable{2})
   local L1_loss_scaling = nn.Mul(layer_size)
   scaled_L1_norm:add(L1_loss_scaling)
   local L1_loss_function = nn.L1Cost()
   if DEBUG_L1 then
      scaled_L1_norm:add(nn.ZeroModule())
   else
      scaled_L1_norm:add(nn.L1CriterionModule(L1_loss_function)) -- also compute the L1 norm on the code
      scaled_L1_norm:add(nn.Ignore()) -- don't pass the L1 loss value onto the rest of the network
      table.insert(criteria_list, L1_loss_function) -- make sure that we consider the L1_loss_function when evaluating the total loss
      print('inserting L1 loss into criteria list, resulting in ' .. #criteria_list .. ' entries')
   end
   apply_L1_norm:add(scaled_L1_norm) -- since we add scaled_L1_norm without a SelectTable, it receives 
   L1_seq:add(apply_L1_norm)

   L1_seq:add(nn.SelectTable({1}, true)) -- when running updateGradInput, this passes a nil back to the L1Cost, which ignores it away; make sure that the output is a table, rather than a tensor
   
   
   L1_loss_scaling.weight[1] = L1_lambda -- make sure that the scaling factor on the L1 loss is constant
   L1_loss_scaling.accGradParameters = function() end -- disable updating
   L1_loss_scaling.updateParameters = function() end -- disable updating
   L1_loss_scaling.accUpdateParameters = function() end -- disable updating
   
   return L1_seq
end


-- a reconstructing-pooling network.  This is like reconstruction ICA, but with reconstruction applied to both the feature extraction and the pooling, and using shrink operators rather than linear transformations for the feature extraction.  The initial version of this network is built with simple linear transformations, but it can just as easily be used to convolutions
-- use DISABLE_NORMALIZATION when testing parameter updates
function build_recpool_net(layer_size, L2_lambda, L1_lambda, num_ista_iterations, DISABLE_NORMALIZATION)
   print('build: L2/L1_lambda is ', L2_lambda, L1_lambda)
   io.read()
   
   local criteria_list = {} -- list of all criteria comprising the loss function.  These are necessary to run the Jacobian unit test forwards
   DISABLE_NORMALIZATION = DISABLE_NORMALIZATION or false

   local function build_pooling()
      local pool_seq = nn.Sequential()
   end
   
   local model = nn.Sequential()
   local forward_dictionary = nn.Linear(layer_size[2],layer_size[1]) -- this should have zero bias, and columns normalized to unit magnitude
   -- ensure that the dictionary matrix has normalized columns
   local function normalize_columns(m)
      for i=1,m:size(2) do
	 m:select(2,i):div(m:select(2,i):norm()+1e-12)
      end
   end
   if not(DISABLE_NORMALIZATION) then 
      forward_dictionary.unnormed_updateParameters = forward_dictionary.updateParameters
      forward_dictionary.unnormed_accUpdateGradParameters = forward_dictionary.accUpdateGradParameters
      
      function forward_dictionary:updateParameters(learningRate) 
	 self:unnormed_updateParameters(learningRate)
	 normalize_columns(self.weight) 
      end 
      function forward_dictionary:accUpdateGradParameters(input, gradOutput, lr) 
	 self:unnormed_accUpdateGradParameters(input, gradOutput, lr) 
	 normalize_columns(self.weight) 
      end
   end
   normalize_columns(forward_dictionary.weight)-- make sure to normalize the initial parameters as well!

   local inverse_dictionary = nn.Linear(layer_size[1],layer_size[2]) -- this should have zero bias!!!
   local base_explaining_away = nn.Linear(layer_size[2], layer_size[2]) -- this should have zero bias!!!
   local base_shrink = nn.ParameterizedShrink(layer_size[2], true, DEBUG_shrink)
   local explaining_away_copies = {}
   local shrink_copies = {}
   inverse_dictionary.weight:copy(forward_dictionary.weight:t())
   base_explaining_away.weight:copy(torch.mm(inverse_dictionary.weight, forward_dictionary.weight)) -- the step constant should only be applied to explaining_away once, rather than twice
   inverse_dictionary.weight:mul(0.2)
   base_explaining_away.weight:mul(-0.2)
   for i = 1,base_explaining_away.weight:size(1) do -- add the identity matrix into base_explaining_away
      --print('adding to ' .. base_explaining_away.weight[{i,i}] .. ' to get ' .. base_explaining_away.weight[{i,i}] + 1)
      base_explaining_away.weight[{i,i}] = base_explaining_away.weight[{i,i}] + 1
   end
   --print(base_explaining_away.weight)
   forward_dictionary.bias:zero()
   inverse_dictionary.bias:zero()
   base_explaining_away.bias:zero()
   base_shrink.shrink_val:fill(0.000001)
   base_shrink.negative_shrink_val:mul(base_shrink.shrink_val, -1)

   model:add(nn.IdentityTable()) -- wrap the tensor in a table
   model:add(nn.CopyTable(1,2)) -- split into the transformed input h_theta(W*x) [1], and the untransformed input x [2]

   local first_WX = nn.ParallelTable()
   first_WX:add(inverse_dictionary) -- transform the input by the transpose dictionary matrix, W*x
   first_WX:add(nn.Identity())
   model:add(first_WX)

   model:add(nn.CopyTable(1,2)) -- split into the subject of the shrink operation z (initially just W*x) [1], the transformed input W*x [2], and the untransformed input x [3]

   local first_shrink_parallel = nn.ParallelTable() -- shrink z; stream is now z = h(W*x) [1], the transformed input W*x [2], and the untransformed input x [3]
   local first_shrink = nn.ParameterizedShrink(layer_size[2], true, DEBUG_shrink) 
   first_shrink:share(base_shrink, 'shrink_val')
  shrink_copies[#shrink_copies + 1] = first_shrink
   first_shrink_parallel:add(first_shrink)
   first_shrink_parallel:add(nn.Identity())
   first_shrink_parallel:add(nn.Identity())
   model:add(first_shrink_parallel)


   local ista_seq
   for i = 1,num_ista_iterations do
      ista_seq, explaining_away_copies[#explaining_away_copies + 1], shrink_copies[#shrink_copies + 1] = build_ISTA_iteration(base_explaining_away, base_shrink, {layer_size[1], layer_size[2]})
      model:add(ista_seq)
   end
   -- calculate the L2 distance between the reconstruction based on the shrunk code D*z, and the original input x; discard the transformed input W*x [2] and the original input x [3]
   model:add(build_L2_reconstruction_loss(L2_lambda, forward_dictionary, criteria_list)) 
   -- calculate the L1 magnitude of the shrunk code z [1], returning the shrunk code z [1] unchanged
   model:add(build_L1_loss(L1_lambda, layer_size[2], criteria_list))
   if DEBUG_OUTPUT then
      model:add(nn.SelectTable({1})) -- when running updateGradInput, this passes a nil back to the L1Cost, which ignores it away; return the unwrapped output tensor for Jacobian testing
   else
      model:add(nn.ParallelDistributingTable('throw away final output')) -- if no modules are added to a ParallelDistributingTable, it throws away its input; updateGradInput produces a tensor of zeros   
   end


   -- try just using the L2 loss 
   if not(DEBUG_OUTPUT) then
      model.original_updateOutput = model.updateOutput
      function model:updateOutput(input)
	 model:original_updateOutput(input)
	 local summed_loss = 0
	 for i = 1,#criteria_list do
	    summed_loss = summed_loss + criteria_list[i].output
	 end
	 model.output = torch.Tensor(1)
	 model.output[1] = summed_loss
	 return model.output
      end
   end

   print('criteria_list contains ' .. #criteria_list .. ' entries')

   return model, criteria_list, forward_dictionary, inverse_dictionary, base_explaining_away, base_shrink, explaining_away_copies, shrink_copies
end

-- Test full reconstructing-pooling model with Jacobian.  Maintain a list of all Criteria.  When running forward, sum over all Criteria to determine the gradient of a single unified energy (should be able to just run updateOutput on the Criteria).  When running backward, just call updateGradInput on the network.  No gradOutput needs to be provided, since all modules terminate in an output-free Criterion.
