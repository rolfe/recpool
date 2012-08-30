require 'torch'
require 'nn'
require 'kex'

DEBUG_shrink = true -- don't require that the shrink value be non-negative, to facilitate the comparison of backpropagated gradients to forward-propagated parameter perturbations
DEBUG_L2 = false
DEBUG_L1 = false
DEBUG_OUTPUT = false
FORCE_NONNEGATIVE_SHRINK_OUTPUT = true -- if the shrink output is non-negative, unrolled ISTA reconstructions tend to be poor unless there are more than twice as many hidden units as visible units, since about half of the hidden units will be prevented from growing smaller than zero, as would be required for optimal reconstruction

-- the input is a single tensor x (not a table)
-- the output is a table of three elements: the subject of the shrink operation z [1], the transformed input W*x [2], and the untransformed input x [3]
local function build_ISTA_first_iteration(encoding_feature_extraction_dictionary, base_shrink, layer_size)
   local first_ista_seq = nn.Sequential()
   first_ista_seq:add(nn.IdentityTable()) -- wrap the tensor in a table
   first_ista_seq:add(nn.CopyTable(1,2)) -- split into the transformed input W*x [1], and the untransformed input x [2]

   local first_WX = nn.ParallelTable()
   first_WX:add(encoding_feature_extraction_dictionary) -- transform the input by the transpose dictionary matrix, W*x [1]
   first_WX:add(nn.Identity()) -- preserve the input x [2]
   first_ista_seq:add(first_WX)

   first_ista_seq:add(nn.CopyTable(1,2)) -- split into the subject of the shrink operation z (initially just W*x) [1], the transformed input W*x [2], and the untransformed input x [3]

   local first_shrink_parallel = nn.ParallelTable() -- shrink z; stream is now z = h(W*x) [1], the transformed input W*x [2], and the untransformed input x [3]
   local first_shrink = nn.ParameterizedShrink(layer_size[2], FORCE_NONNEGATIVE_SHRINK_OUTPUT, DEBUG_shrink) 
   first_shrink:share(base_shrink, 'shrink_val')
   first_shrink_parallel:add(first_shrink)
   first_shrink_parallel:add(nn.Identity())
   first_shrink_parallel:add(nn.Identity())
   first_ista_seq:add(first_shrink_parallel)

   return first_ista_seq, first_shrink
end

-- the input is a table of three elments: the subject of the shrink operation z [1], the transformed input W*x [2], and the untransformed input x [3]
-- the output is a table of three elements: the subject of the shrink operation z [1], the transformed input W*x [2], and the untransformed input x [3]
local function build_ISTA_iteration(base_explaining_away, base_shrink, layer_size, DISABLE_NORMALIZATION)
   local explaining_away = nn.ConstrainedLinear(layer_size[2], layer_size[2], {no_bias = true}, DISABLE_NORMALIZATION)
   local shrink = nn.ParameterizedShrink(layer_size[2], FORCE_NONNEGATIVE_SHRINK_OUTPUT, DEBUG_shrink) -- EFFICIENCY NOTE: when using non-negative units this could be accomplished more efficiently using an unparameterized, one-sided rectification, just like Glorot, Bordes, and Bengio, along with a non-positive bias in the encoding_feature_extraction_dictionary.  However, both nn.SoftShrink and the shrinkage utility method implemented in kex are two-sided.
   
   explaining_away:share(base_explaining_away, 'weight', 'bias')
   shrink:share(base_shrink, 'shrink_val')
   
   local ista_seq = nn.Sequential()
   local explaining_away_parallel = nn.ParallelTable() 
   explaining_away_parallel:add(explaining_away) --subtract out the explained input: S*z [1]
   explaining_away_parallel:add(nn.Identity()) -- preserve W*x [2]
   explaining_away_parallel:add(nn.Identity()) -- preserve x [3]
   ista_seq:add(explaining_away_parallel)

   -- we need to do this in two stages, since S must be applied only to z, rather than to z + Wx; adding in Wx needs to be the first in the sequence of operations within a ParallelDistributingTable
   local add_in_WX_and_shrink_parallel = nn.ParallelDistributingTable() 
   local add_in_WX_and_shrink_seq = nn.Sequential()
   add_in_WX_and_shrink_seq:add(nn.SelectTable{1,2})
   add_in_WX_and_shrink_seq:add(nn.CAddTable())
   add_in_WX_and_shrink_seq:add(shrink)
   add_in_WX_and_shrink_parallel:add(add_in_WX_and_shrink_seq) -- z = h_theta(z + Wx + S*z) [1]
   add_in_WX_and_shrink_parallel:add(nn.SelectTable{2}) -- preserve W*x [2]
   add_in_WX_and_shrink_parallel:add(nn.SelectTable{3}) -- preserve x [3]
   ista_seq:add(add_in_WX_and_shrink_parallel)

   return ista_seq, explaining_away, shrink
end

-- the input is a table of three elments: the subject of the shrink operation z [1], and the untransformed input x [untransformed_input_index] ; other signals are ignored
-- the output is a table of three elements: the subject of the shrink operation z [1], the reconstructed input D*z [2], and the untransformed input x [3]
local function linearly_reconstruct_input(decoding_dictionary, untransformed_input_index)
   local reconstruct_input_parallel = nn.ParallelDistributingTable() -- reconstruct the input from the shrunk code z
   local reconstruct_input_seq = nn.Sequential()
   reconstruct_input_seq:add(nn.SelectTable{1})
   reconstruct_input_seq:add(decoding_dictionary)
   reconstruct_input_parallel:add(nn.SelectTable{1}) -- preserve the subject of the shrink operation z [1]
   reconstruct_input_parallel:add(reconstruct_input_seq) -- reconstruct the input D*z from the shrunk code z [2]
   reconstruct_input_parallel:add(nn.SelectTable{untransformed_input_index}) -- preserve the untransformed input x [3]

   return reconstruct_input_parallel
end

-- the input is a table of three elments: the layer n code z [1], the reconstructed layer n-1 code r [2], and untransformed layer n-1 input x [3]
-- the output is a table of one element: the layer n code z [1]
local function build_L2_reconstruction_loss(L2_lambda, criteria_list)
   local combined_loss = nn.ParallelDistributingTable() -- calculate the MSE between the reconstructed layer n-1 code r [2] and untransformed layer n-1 input x [3]; only pass on the layer n code z [1]
   
   local L2_loss_seq = nn.Sequential() -- calculate the MSE between the reconstructed input D*z [2] and the untransformed input x [4]
   L2_loss_seq:add(nn.SelectTable{2,3})
   if DEBUG_L2 or DEBUG_OUTPUT then
      local sequential_zero = nn.Sequential()
      local parallel_zero = nn.ParallelTable() 
      parallel_zero:add(nn.ZeroModule())
      parallel_zero:add(nn.ZeroModule())
      sequential_zero:add(parallel_zero)
      sequential_zero:add(nn.SelectTable{1}) -- a SelectTable is necessary to ensure that the module outputs a single nil, which is ignored by the ParallelDistributingTable, rather than an empty table (i.e., a table of nils), which ParallelDistributingTable incorrectly passes onto its output
      L2_loss_seq:add(sequential_zero) 
   else
      if L2_lambda == nil then
	 error('L2_lambda is incorrectly assigned to be ', L2_lambda, ' with type ', type(L2_lambda))
      end
      local effective_L2_loss_function = nn.L2Cost(L2_lambda, 2)
      L2_loss_seq:add(effective_L2_loss_function) -- using this instead of MSECriterion fed through CriterionTable ensures that the gradient is propagated to both inputs
      L2_loss_seq:add(nn.Ignore()) -- don't pass the L2 loss value onto the rest of the network
      table.insert(criteria_list, effective_L2_loss_function)   
      print('inserting L2 loss into criteria list, resulting in ' .. #criteria_list .. ' entries')
   end

   combined_loss:add(nn.SelectTable{1}) -- throw away all streams but the shrunk code z [1]; the result is a table with a single entry
   combined_loss:add(L2_loss_seq)

   -- rather than using nn.Ignore on the output of the criterion, we could use a SelectTable{1} without a ParallelDistributingTable, which would output a tensor in the forward direction, and send a nil as gradOutput to the criterion in the backwards direction

   return combined_loss
end


-- the input is a table of one element: the subject of the shrink operation z [1]
-- the output is a table of one element: the subject of the shrink operation z [1]
local function build_sparsifying_loss(sparsifying_loss_function, sparsifying_lambda, layer_size, criteria_list)
   local L1_seq = nn.Sequential()
   L1_seq:add(nn.CopyTable(1, 2)) -- split into the output to the rest of the chain [1] and the output to the L1 norm [2]
   
   local apply_L1_norm = nn.ParallelDistributingTable()
   local scaled_L1_norm = nn.Sequential() -- scale the code copy [2], calculate its L1 norm, and throw away the output
   local L1_loss_scaling -- define local outside of the if block, so it is accessible throughout this function
   scaled_L1_norm:add(nn.SelectTable{2})
   if (sparsifying_lambda ~= nil) and (sparsifying_lambda ~= 1) then -- nn.L1Cost, defined in kex, does not include a scaling factor lambda; CauchyCost which I define does include a scaling factor
      L1_loss_scaling = nn.Mul(layer_size)
      scaled_L1_norm:add(L1_loss_scaling)
   end
   if DEBUG_L1 or DEBUG_OUTPUT then
      scaled_L1_norm:add(nn.ZeroModule())
   else
      scaled_L1_norm:add(nn.L1CriterionModule(sparsifying_loss_function)) -- also compute the L1 norm on the code
      scaled_L1_norm:add(nn.Ignore()) -- don't pass the L1 loss value onto the rest of the network
      table.insert(criteria_list, sparsifying_loss_function) -- make sure that we consider the sparsifying_loss_function when evaluating the total loss
      print('inserting sparsifying loss into criteria list, resulting in ' .. #criteria_list .. ' entries')
   end
   apply_L1_norm:add(nn.SelectTable{1}) -- pass the code [1] through unchanged for further processing   
   apply_L1_norm:add(scaled_L1_norm) -- since we add scaled_L1_norm without a SelectTable, it receives 
   L1_seq:add(apply_L1_norm)

   -- FOR THE LOVE OF GOD!!! TRY REMOVING THIS!!!
   --L1_seq:add(nn.SelectTable({1}, true)) -- when running updateGradInput, this passes a nil back to the L1Cost, which ignores it away; make sure that the output is a table, rather than a tensor
   
   if (sparsifying_lambda ~= nil) and (sparsifying_lambda ~= 1) then
      L1_loss_scaling.weight[1] = sparsifying_lambda -- make sure that the scaling factor on the L1 loss is constant
      L1_loss_scaling.accGradParameters = function() end -- disable updating
      L1_loss_scaling.updateParameters = function() end -- disable updating
      L1_loss_scaling.accUpdateParameters = function() end -- disable updating
   end
   
   return L1_seq
end

-- the input is a table of one element: the output of the shrink operation z [1]
-- the output is a table of two elements: the subject of the pooling operation s [1], and the preserved input z [2]
local function build_pooling(encoding_pooling_dictionary)
   local pool_seq = nn.Sequential()
   pool_seq:add(nn.CopyTable(1, 2)) -- split into split into the output to the rest of the chain z [1] and the preserved input z [2], used to calculate the reconstruction error

   local pool_features_parallel = nn.ParallelTable() -- compute s = sqrt(Q*z^2)
   local pooling_transformation_seq = nn.Sequential()
   --pooling_transformation_seq:add(nn.Square()) -- EFFICIENCY NOTE: nn.Square is almost certainly more efficient than Power(2), but it computes gradInput incorrectly on 1-D inputs; specifically, it fails to multiply the gradInput by two.  This can and should be corrected, but doing so will require modifying c code.
   pooling_transformation_seq:add(nn.Power(2))
   pooling_transformation_seq:add(encoding_pooling_dictionary)
   pooling_transformation_seq:add(nn.AddConstant(1e-5)) -- ensures that we never compute the gradient of sqrt(0), so long as encoding_pooling_dictionary is non-negative with no bias

   -- Note that backpropagating gradients through nn.Sqrt will generate NANs if any of the inputs are exactly zero.  However, this is correct behavior.  We should ensure that no input to the square root is exactly zero, perhaps by bounding the encoding_pooling_dictionary below by a number greater than zero.
   pooling_transformation_seq:add(nn.Sqrt())

   pool_features_parallel:add(pooling_transformation_seq) -- s = sqrt(Q*z^2) [1]
   pool_features_parallel:add(nn.Identity()) -- preserve z [2]
   pool_seq:add(pool_features_parallel) 

   return pool_seq
end


-- the input is a table of two elements: the subject of the pooling operation s [1], and the preserved input z [2]
-- the output is a table of one element: the subject of the pooling operation s [1]
local function build_pooling_L2_loss(decoding_pooling_dictionary, L2_reconstruction_lambda, L2_position_unit_lambda, mask_cauchy_lambda, criteria_list)
   -- the L2 reconstruction error and the L2 position unit magnitude both depend upon the denominator 1 + (P*s)^2, so calculate them together in a single function

   local pool_L2_loss_seq = nn.Sequential()
   pool_L2_loss_seq:add(nn.CopyTable(1, 2)) -- split into the output from the module s = sqrt(Q*z^2) [1], the basis of the denominator of the losses s [2], and the preserved input z [3]

   -- split into s [1], P*s [2], and z [3]
   local reconstruction_parallel = nn.ParallelTable() -- compute the reconstruction of the input P*s
   reconstruction_parallel:add(nn.Identity()) -- preserve the output s = sqrt(Q*z^2) [1]
   reconstruction_parallel:add(decoding_pooling_dictionary) -- compute the reconstruction P*s [2]
   reconstruction_parallel:add(nn.Identity()) -- preserve the original input z [3]
   pool_L2_loss_seq:add(reconstruction_parallel)

   -- compute the sparsifying regularizer on the pooling mask P*s [2]; output is still s [1], P*s [2], and z [3]
   local sparsifying_loss_parallel = nn.ParallelDistributingTable()
   local sparsifying_loss_seq = nn.Sequential()
   sparsifying_loss_seq:add(nn.SelectTable{2})
   if DEBUG_OUTPUT then
      sparsifying_loss_seq:add(nn.ZeroModule())
   else
      local sparsifying_mask_loss_function = nn.CauchyCost(mask_cauchy_lambda)
      sparsifying_loss_seq:add(nn.L1CriterionModule(sparsifying_mask_loss_function)) -- also compute the caucy norm on the code
      table.insert(criteria_list, sparsifying_mask_loss_function) -- make sure that we consider the sparsifying_loss_function when evaluating the total loss
      print('inserting sparsifying cauchy mask loss into criteria list, resulting in ' .. #criteria_list .. ' entries')
   end
   sparsifying_loss_seq:add(nn.Ignore()) -- don't pass the cauchy loss value onto the rest of the network
   sparsifying_loss_parallel:add(nn.SelectTable{1})
   sparsifying_loss_parallel:add(nn.SelectTable{2})
   sparsifying_loss_parallel:add(nn.SelectTable{3})
   sparsifying_loss_parallel:add(sparsifying_loss_seq)
   pool_L2_loss_seq:add(sparsifying_loss_parallel)

   -- the output from the module s [1], the preserved input z [2], the numerator of the position loss z*(P*s) [3], and the denominator of the loss 1 + (P*s)^2 [4] 
   local construct_num_denom_parallel = nn.ParallelDistributingTable('construct_num_denom_parallel') -- compute the numerator of the position loss z*(P*s) and the denominator of the losses 1 + (P*s)^2
   local construct_denominator_seq = nn.Sequential()
   construct_denominator_seq:add(nn.SelectTable{2})
   --construct_denominator_seq:add(nn.Square()) -- EFFICIENCY NOTE: nn.Square is almost certainly more efficient than Power(2), but it computes gradInput incorrectly on 1-D inputs; specifically, if fails to multiply the gradInput by two
   construct_denominator_seq:add(nn.Power(2))
   construct_denominator_seq:add(nn.AddConstant(1))
   local construct_numerator_seq = nn.Sequential()
   construct_numerator_seq:add(nn.SelectTable{2,3})
   construct_numerator_seq:add(nn.CMulTable())
   construct_num_denom_parallel:add(nn.SelectTable{1}) -- preserve the output s = sqrt(Q*z^2) [1]
   construct_num_denom_parallel:add(nn.SelectTable{3}) -- preserve the original input z [2]
   construct_num_denom_parallel:add(construct_numerator_seq) -- compute the position numerator z*(P*s) [3]
   construct_num_denom_parallel:add(construct_denominator_seq) -- compute the denominator 1 + (P*s)^2 [4]
   --construct_num_denom_parallel:add(nn.Identity()) 
   pool_L2_loss_seq:add(construct_num_denom_parallel)
   
   local compute_loss_parallel = nn.ParallelDistributingTable('compute_loss_parallel') -- compute the loss functions z / (1 + (P*s)^2) and z*(P*s) / (1 + (P*s)^2)
   local compute_reconstruction_loss_seq = nn.Sequential()
   compute_reconstruction_loss_seq:add(nn.SelectTable{2,4})
   compute_reconstruction_loss_seq:add(nn.CDivTable())
   if DEBUG_OUTPUT then
      compute_reconstruction_loss_seq:add(nn.ZeroModule())
   else
      local L2_reconstruction_loss = nn.L2Cost(L2_reconstruction_lambda, 1)
      table.insert(criteria_list, L2_reconstruction_loss) -- make sure that we consider the L2_reconstruction_loss when evaluating the total loss
      print('inserting L2 pooling reconstruction loss into criteria list, resulting in ' .. #criteria_list .. ' entries')

      compute_reconstruction_loss_seq:add(L2_reconstruction_loss)
   end
   compute_reconstruction_loss_seq:add(nn.Ignore)
   local compute_position_loss_seq = nn.Sequential()
   compute_position_loss_seq:add(nn.SelectTable{3,4})
   compute_position_loss_seq:add(nn.CDivTable())
   if DEBUG_OUTPUT then
      compute_position_loss_seq:add(nn.ZeroModule())
   else
      local L2_position_loss = nn.L2Cost(L2_position_unit_lambda, 1)
      table.insert(criteria_list, L2_position_loss) -- make sure that we consider the L2_reconstruction_loss when evaluating the total loss
      print('inserting L2 pooling position loss into criteria list, resulting in ' .. #criteria_list .. ' entries')
      
      compute_position_loss_seq:add(L2_position_loss)
   end
   compute_position_loss_seq:add(nn.Ignore)

   compute_loss_parallel:add(nn.SelectTable{1}) -- preserve the output s = sqrt(Q*z^2) [1]
   compute_loss_parallel:add(compute_reconstruction_loss_seq) -- compute the reconstruction loss ||z / (1 + (P*s)^2)||^2 [-]
   compute_loss_parallel:add(compute_position_loss_seq) -- compute the position loss ||z*(Q*s) / (1 + (P*s)^2) [-]
   pool_L2_loss_seq:add(compute_loss_parallel)

   -- IS THIS NECESSARY!?!?!
   --pool_L2_loss_seq:add(nn.SelectTable({1}, true)) -- when running updateGradInput, this passes a nil back to the L1Cost, which ignores it away; make sure that the output is a table, rather than a tensor

   return pool_L2_loss_seq
end

-- a reconstructing-pooling network.  This is like reconstruction ICA, but with reconstruction applied to both the feature extraction and the pooling, and using shrink operators rather than linear transformations for the feature extraction.  The initial version of this network is built with simple linear transformations, but it can just as easily be used to convolutions
-- use DISABLE_NORMALIZATION when testing parameter updates
function build_recpool_net(layer_size, lambdas, num_ista_iterations, DISABLE_NORMALIZATION) 
   -- lambdas: {ista_L2_reconstruction_lambda, ista_L1_lambda, pooling_L2_reconstruction_lambda, pooling_L2_position_unit_lambda, pooling_output_cauchy_lambda, pooling_mask_cauchy_lambda}
   local criteria_list = {} -- list of all criteria comprising the loss function.  These are necessary to run the Jacobian unit test forwards
   DISABLE_NORMALIZATION = DISABLE_NORMALIZATION or false

   local model = nn.Sequential()
   -- the exact range for the initialization of weight matrices by nn.Linear doesn't matter, since they are rescaled by the normalized_columns constraint
   local encoding_feature_extraction_dictionary = nn.ConstrainedLinear(layer_size[1],layer_size[2], {no_bias = true}, DISABLE_NORMALIZATION) 
   local decoding_feature_extraction_dictionary = nn.ConstrainedLinear(layer_size[2],layer_size[1], {no_bias = true, normalized_columns = true}, DISABLE_NORMALIZATION) 
   local base_explaining_away = nn.ConstrainedLinear(layer_size[2], layer_size[2], {no_bias = true}, DISABLE_NORMALIZATION) 
   local base_shrink = nn.ParameterizedShrink(layer_size[2], FORCE_NONNEGATIVE_SHRINK_OUTPUT, DEBUG_shrink)
   local explaining_away_copies = {}
   local shrink_copies = {}
   
   local encoding_pooling_dictionary = nn.ConstrainedLinear(layer_size[2], layer_size[3], {no_bias = true, non_negative = true}, DISABLE_NORMALIZATION) -- this should have zero bias
   local decoding_pooling_dictionary = nn.ConstrainedLinear(layer_size[3], layer_size[2], {no_bias = true, normalized_columns = true, non_negative = true}, DISABLE_NORMALIZATION) -- this should have zero bias, and columns normalized to unit magnitude

   local classification_dictionary = nn.Linear(layer_size[3], layer_size[4])
   local classification_criterion = nn.L1CriterionModule(nn.ClassNLLCriterion(), true) -- on each iteration classfication_criterion:setTarget(target) must be called

   local L1_loss_function = nn.L1Cost()
   local cauchy_loss_function = nn.CauchyCost(lambdas.pooling_output_cauchy_lambda)


   encoding_feature_extraction_dictionary.weight:copy(decoding_feature_extraction_dictionary.weight:t())
   encoding_pooling_dictionary.weight:copy(decoding_pooling_dictionary.weight:t())

   base_explaining_away.weight:copy(torch.mm(encoding_feature_extraction_dictionary.weight, decoding_feature_extraction_dictionary.weight)) -- the step constant should only be applied to explaining_away once, rather than twice
   encoding_feature_extraction_dictionary.weight:mul(0.2)
   base_explaining_away.weight:mul(-0.2)
   for i = 1,base_explaining_away.weight:size(1) do -- add the identity matrix into base_explaining_away
      base_explaining_away.weight[{i,i}] = base_explaining_away.weight[{i,i}] + 1
   end
   --base_explaining_away.weight:mul(0.01)
   base_shrink.shrink_val:fill(0.01)
   base_shrink.negative_shrink_val:mul(base_shrink.shrink_val, -1)

   -- take the initial input x and calculate the sparse code z [1], the transformed input W*x [2], and the untransformed input x [3]
   local ista_seq
   ista_seq, shrink_copies[#shrink_copies + 1] = build_ISTA_first_iteration(encoding_feature_extraction_dictionary, base_shrink, {layer_size[1], layer_size[2]})
   model:add(ista_seq)

   for i = 1,num_ista_iterations do
      --calculate the sparse code z [1]; preserve the transformed input W*x [2], and the untransformed input x [3]
      ista_seq, explaining_away_copies[#explaining_away_copies + 1], shrink_copies[#shrink_copies + 1] = build_ISTA_iteration(base_explaining_away, base_shrink, {layer_size[1], layer_size[2]}, DISABLE_NORMALIZATION)
      model:add(ista_seq)
   end
   -- reconstruct the input D*z [2] from the code z [1], leaving [1] and [3] unchanged
   model:add(linearly_reconstruct_input(decoding_feature_extraction_dictionary, 3))
   -- calculate the L2 distance between the reconstruction based on the shrunk code D*z [2], and the original input x [3]; discard all signals but the current code z [1]
   model:add(build_L2_reconstruction_loss(lambdas.ista_L2_reconstruction_lambda, criteria_list)) 
   -- calculate the L1 magnitude of the shrunk code z [1], returning the shrunk code z [1] unchanged
   model:add(build_sparsifying_loss(L1_loss_function, lambdas.ista_L1_lambda, layer_size[2], criteria_list))

   -- pool the input z [1] to obtain the pooled code s = sqrt(Q*z^2) [1], and the preserved input z [2]
   model:add(build_pooling(encoding_pooling_dictionary, decoding_pooling_dictionary))
   -- calculate the L2 reconstruction and position loss for pooling; return the pooled code s [1]
   model:add(build_pooling_L2_loss(decoding_pooling_dictionary, lambdas.pooling_L2_reconstruction_lambda, lambdas.pooling_L2_position_unit_lambda, lambdas.pooling_mask_cauchy_lambda, criteria_list))
   -- calculate the L1 magnitude of the pooling code s [1], returning the pooling code s [1] unchanged
   model:add(build_sparsifying_loss(cauchy_loss_function, nil, layer_size[3], criteria_list)) 
   -- ALSO ADD IN AND L1 PENALTY ON THE MASK INDUCED BY THE POOLING UNITS!!!
   model:add(nn.SelectTable{1})
   model:add(classification_dictionary)
   model:add(nn.LogSoftMax())
   if not(DEBUG_OUTPUT) then
      --model:add(nn.ZeroModule())
      model:add(classification_criterion) 

      --model.set_target = function (self, new_target) 
      function model:set_target(new_target) 
	 classification_criterion:setTarget(new_target) 
      end 

      print('model.set_target is now ', model.set_target)

      table.insert(criteria_list, classification_criterion)
      print('inserting classification negative log likelihood loss into criteria list, resulting in ' .. #criteria_list .. ' entries')
   else
      function model:set_target(new_target) print('WARNING: set_target does nothing when using DEBUG_OUTPUT') end
   end

   --model:add(nn.ParallelDistributingTable('throw away final output')) -- if no modules are added to a ParallelDistributingTable, it throws away its input; updateGradInput produces a tensor of zeros   

   if not(DEBUG_OUTPUT) then
      model.original_updateOutput = model.updateOutput
      
      -- note that this is different than the original model.output; model is a nn.Sequential, which ends in the classification_criterion, and thus consists of a single number
      -- it is not desirable to have the model produce a tensor output, since this would imply a corresponding gradOutput, at least in the standard implementation, whereas we want all gradients to be internally generated.  
      function model:updateOutput(input)
	 model:original_updateOutput(input)
	 local summed_loss = 0
	 for i = 1,#criteria_list do
	    summed_loss = summed_loss + criteria_list[i].output
	 end
	 -- this is necessary to maintain compatibility with ModifiedJacobian, which directly accesses the output field of the tested module
	 model.output = torch.Tensor(1)
	 model.output[1] = summed_loss
	 return model.output -- while it is tempting to return classification_dictionary.output here, it risks incompatibility with ModifiedJacobian and any other package that expects a single return value
      end

      function model:get_classifier_output()
	 return classification_dictionary.output
      end
   end

   print('criteria_list contains ' .. #criteria_list .. ' entries')      

   return model, criteria_list, encoding_feature_extraction_dictionary, decoding_feature_extraction_dictionary, encoding_pooling_dictionary, decoding_pooling_dictionary, classification_dictionary, base_explaining_away, base_shrink, explaining_away_copies, shrink_copies
end

-- Test full reconstructing-pooling model with Jacobian.  Maintain a list of all Criteria.  When running forward, sum over all Criteria to determine the gradient of a single unified energy (should be able to just run updateOutput on the Criteria).  When running backward, just call updateGradInput on the network.  No gradOutput needs to be provided, since all modules terminate in an output-free Criterion.
