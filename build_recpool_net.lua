require 'torch'
require 'nn'
require 'kex'

IGNORE_CLASSIFICATION = false
DEBUG_shrink = false -- don't require that the shrink value be non-negative, to facilitate the comparison of backpropagated gradients to forward-propagated parameter perturbations
DEBUG_L2 = false
DEBUG_L1 = false
DEBUG_OUTPUT = false
DEBUG_RF_TRAINING_SCALE = nil
FORCE_NONNEGATIVE_SHRINK_OUTPUT = true -- if the shrink output is non-negative, unrolled ISTA reconstructions tend to be poor unless there are more than twice as many hidden units as visible units, since about half of the hidden units will be prevented from growing smaller than zero, as would be required for optimal reconstruction

-- the input is x [1] (already wrapped in a table)
-- the output is a table of three elements: the subject of the shrink operation z [1], the transformed input W*x [2], and the untransformed input x [3]
local function build_ISTA_first_iteration(encoding_feature_extraction_dictionary, base_shrink)
   local first_ista_seq = nn.Sequential()
   first_ista_seq:add(nn.CopyTable(1,2)) -- split into the transformed input W*x [1], and the untransformed input x [2]

   local first_WX = nn.ParallelTable()
   first_WX:add(encoding_feature_extraction_dictionary) -- transform the input by the transpose dictionary matrix, W*x [1]
   first_WX:add(nn.Identity()) -- preserve the input x [2]
   first_ista_seq:add(first_WX)

   first_ista_seq:add(nn.CopyTable(1,2)) -- split into the subject of the shrink operation z (initially just W*x) [1], the transformed input W*x [2], and the untransformed input x [3]

   local first_shrink_parallel = nn.ParallelTable() -- shrink z; stream is now z = h(W*x) [1], the transformed input W*x [2], and the untransformed input x [3]
   
   --[[
      local first_shrink = nn.ParameterizedShrink(layer_size[2], FORCE_NONNEGATIVE_SHRINK_OUTPUT, DEBUG_shrink) 
      first_shrink:share(base_shrink, 'shrink_val', 'grad_shrink_val', 'negative_shrink_val')
      if (first_shrink.shrink_val:storage() ~= base_shrink.shrink_val:storage()) or (first_shrink.grad_shrink_val:storage() ~= base_shrink.grad_shrink_val:storage()) then
      print('in build_ISTA_first_iteration, shrink parameters are not shared properly')
      io.read()
      end
      current_shrink = first_shrink
   --]]
   
   first_shrink_parallel:add(base_shrink)
   first_shrink_parallel:add(nn.Identity())
   first_shrink_parallel:add(nn.Identity())
   first_ista_seq:add(first_shrink_parallel)

   return first_ista_seq --, current_shrink
end

local function duplicate_linear_explaining_away(base_explaining_away, layer_size, use_base_explaining_away, RUN_JACOBIAN_TEST)
   local explaining_away
   if use_base_explaining_away then
      explaining_away = base_explaining_away
   else
      explaining_away = nn.ConstrainedLinear(layer_size[2], layer_size[2], {no_bias = true, non_negative_diag = false}, RUN_JACOBIAN_TEST, DEBUG_RF_TRAINING_SCALE)
      explaining_away:share(base_explaining_away, 'weight', 'bias', 'gradWeight', 'gradBias')
   end

   return explaining_away
end

local function duplicate_shrink(base_shrink, layer_size, RUN_JACOBIAN_TEST)
   local shrink = nn.ParameterizedShrink(layer_size[2], FORCE_NONNEGATIVE_SHRINK_OUTPUT, DEBUG_shrink) -- EFFICIENCY NOTE: when using non-negative units this could be accomplished more efficiently using an unparameterized, one-sided rectification, just like Glorot, Bordes, and Bengio, along with a non-positive bias in the encoding_feature_extraction_dictionary.  However, both nn.SoftShrink and the shrinkage utility method implemented in kex are two-sided.
   shrink:share(base_shrink, 'shrink_val', 'grad_shrink_val', 'negative_shrink_val') -- SHOULD negative_shrink_val BE SHARED AS WELL!?!?!  FOR THE LOVE OF GOD!!!  FIX THIS!!!
   if (shrink.shrink_val:storage() ~= base_shrink.shrink_val:storage()) or (shrink.grad_shrink_val:storage() ~= base_shrink.grad_shrink_val:storage()) then
      print('in build_ISTA_iteration, shrink parameters are not shared properly')
      io.read()
   end

   return shrink
end
   



-- the input is a table of three elments: the subject of the shrink operation z [1], the transformed input W*x [2], and the untransformed input x [3]
-- the output is a table of three elements: the subject of the shrink operation z [1], the transformed input W*x [2], and the untransformed input x [3]
local function build_ISTA_iteration(explaining_away, shrink, RUN_JACOBIAN_TEST)
   local ista_seq = nn.Sequential()
   local explaining_away_parallel = nn.ParallelTable() 
   explaining_away_parallel:add(explaining_away) --subtract out the explained input: z + S*z [1]
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

   return ista_seq
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
-- the output is a table of two elements: the layer n code z [1], and the untransformed layer n-1 input x[2]
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
      table.insert(criteria_list.criteria, effective_L2_loss_function)   
      table.insert(criteria_list.names, 'L2 reconstruction loss')
      print('inserting L2 loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
   end

   combined_loss:add(nn.SelectTable{1}) -- throw away all streams but the shrunk code z [1] and the original input x [3]; the result is a table with two entries
   combined_loss:add(nn.SelectTable{3})
   combined_loss:add(L2_loss_seq)

   -- rather than using nn.Ignore on the output of the criterion, we could use a SelectTable{1} without a ParallelDistributingTable, which would output a tensor in the forward direction, and send a nil as gradOutput to the criterion in the backwards direction

   return combined_loss
end


-- the input is a table of two elements: the subject of the shrink operation z [1], and the original input x [2]
-- the output is a table of two elements: the subject of the shrink operation z [1], and the original input x [2]
local function build_sparsifying_loss(sparsifying_criterion_module, criteria_list)
   --local L1_seq = nn.Sequential()
   --L1_seq:add(nn.CopyTable(1, 2)) -- split into the output to the rest of the chain [1] and the output to the L1 norm [2]; original input x is now [3]
   
   local apply_L1_norm = nn.ParallelDistributingTable()
   local scaled_L1_norm = nn.Sequential() -- scale the code [1], calculate its L1 norm, and throw away the output
   scaled_L1_norm:add(nn.SelectTable{1}) 
   if DEBUG_L1 or DEBUG_OUTPUT then
      scaled_L1_norm:add(nn.ZeroModule())
   else
      --local sparsifying_criterion_module = nn.L1CriterionModule(sparsifying_loss_function, sparsifying_lambda) -- the L1CriterionModule, rather than the wrapped criterion, produces the correct scaled error
      --local sparsifying_criterion_module = ParameterizedL1Cost(layer_size, sparsifying_lambda, desired_criterion_value, lagrange_multiplier_learning_rate_scaling_factor)
      scaled_L1_norm:add(sparsifying_criterion_module) -- also compute the L1 norm on the code
      scaled_L1_norm:add(nn.Ignore()) -- don't pass the L1 loss value onto the rest of the network
      table.insert(criteria_list.criteria, sparsifying_criterion_module) -- make sure that we consider the sparsifying_loss_function when evaluating the total loss
      table.insert(criteria_list.names, 'sparsifying loss')
      print('inserting sparsifying loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
   end
   apply_L1_norm:add(nn.SelectTable{1}) -- pass the code [1] through unchanged for further processing   
   apply_L1_norm:add(nn.SelectTable{2}) -- pass the original input x [2] through unchanged for further processing   
   apply_L1_norm:add(scaled_L1_norm) -- since we add scaled_L1_norm without a SelectTable, it receives 
   --L1_seq:add(apply_L1_norm)

   return apply_L1_norm --L1_seq
end

-- the input is a table of two elements: the output of the shrink operation z [1], and the original input x [2]
-- the output is a table of three elements: the subject of the pooling operation s [1], the preserved input z [2], and the original input x [3]
local function build_pooling(encoding_pooling_dictionary)
   local pool_seq = nn.Sequential()
   pool_seq:add(nn.CopyTable(1, 2)) -- split into the output to the rest of the chain z [1] and the preserved input z [2], used to calculate the reconstruction error, and the original input x [3]

   local pool_features_parallel = nn.ParallelTable() -- compute s = sqrt(Q*z^2)
   local pooling_transformation_seq = nn.Sequential()
   pooling_transformation_seq:add(nn.Square()) -- EFFICIENCY NOTE: nn.Square is almost certainly more efficient than Power(2), but it computes gradInput incorrectly on 1-D inputs; specifically, it fails to multiply the gradInput by two.  This can and should be corrected, but doing so will require modifying c code.
   --pooling_transformation_seq:add(nn.SafePower(2))
   pooling_transformation_seq:add(encoding_pooling_dictionary)
   local safe_sqrt_const = 1e-10
   pooling_transformation_seq:add(nn.AddConstant(encoding_pooling_dictionary.weight:size(1), safe_sqrt_const)) -- ensures that we never compute the gradient of sqrt(0), so long as encoding_pooling_dictionary is non-negative with no bias.  Keep in mind that we take the square root of this quantity, so even a seemingly small number like 1e-5 becomes a relatively large constant of 0.00316.  At the same time, the gradient of the square root divides by the square root, and so becomes huge as the argument of the square root approaches zero

   -- Note that backpropagating gradients through nn.Sqrt will generate NANs if any of the inputs are exactly zero.  However, this is correct behavior.  We should ensure that no input to the square root is exactly zero, perhaps by bounding the encoding_pooling_dictionary below by a number greater than zero.
   pooling_transformation_seq:add(nn.Sqrt()) -- NORMAL VERSION
   pooling_transformation_seq:add(nn.AddConstant(encoding_pooling_dictionary.weight:size(1), -math.sqrt(safe_sqrt_const))) -- we add 1e-4 before the square root and subtract (1e-4)^1/2 after the square root, so zero is still mapped to zero, but the gradient contribution is bounded above by 100

   --pooling_transformation_seq:add(nn.SafePower(1/2)) -- EXPERIMENTAL VERSION
   --pooling_transformation_seq:add(nn.AddConstant(encoding_pooling_dictionary.weight:size(1), -math.pow(safe_sqrt_const, 1/2))) -- we add 1e-4 before the square root and subtract (1e-4)^1/2 after the square root, so zero is still mapped to zero, but the gradient contribution is bounded above by 100

   pool_features_parallel:add(pooling_transformation_seq) -- s = sqrt(Q*z^2) [1]
   pool_features_parallel:add(nn.Identity()) -- preserve z [2]
   pool_features_parallel:add(nn.Identity()) -- preserve x [3]
   pool_seq:add(pool_features_parallel) 

   return pool_seq
end


local function build_loss_tester(num_inputs, tested_inputs, criteria_list)
   print('ADDING LOSS TESTER!')
   local test_pdt = nn.ParallelDistributingTable()
   
   for i = 1,num_inputs do
      test_pdt:add(nn.SelectTable{i})
   end

   if tested_inputs then 
      for k,v in ipairs(tested_inputs) do
	 local test_L2_loss = nn.L2Cost(math.random(), 1)
	 table.insert(criteria_list.criteria, test_L2_loss) -- make sure that we consider the L2_position_loss when evaluating the total loss
	 table.insert(criteria_list.names, 'test L2 loss ' .. v)
	 print('inserting test L2 loss ' .. v .. ' into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
	 
	 local test_seq = nn.Sequential()
	 test_seq:add(nn.SelectTable{v})
	 test_seq:add(test_L2_loss)
	 test_seq:add(nn.Ignore())
	 test_pdt:add(test_seq)
      end
   end

   return test_pdt
end


-- creates two copies of the base_decoding_feature_extraction_dictionary, the second of which has the weight matrix transposed
local function duplicate_decoding_feature_extraction_dictionary(base_decoding_feature_extraction_dictionary, layer_size, RUN_JACOBIAN_TEST)
   local decoding_feature_extraction_dictionary_copy = nn.ConstrainedLinear(layer_size[2],layer_size[1], {no_bias = true, normalized_columns = true}, RUN_JACOBIAN_TEST, DEBUG_RF_TRAINING_SCALE) 
   decoding_feature_extraction_dictionary_copy:share(base_decoding_feature_extraction_dictionary, 'weight', 'bias', 'gradWeight', 'gradBias')

   local decoding_feature_extraction_dictionary_transpose = nn.ConstrainedLinear(layer_size[1],layer_size[2], {no_bias = true, normalized_columns = true}, RUN_JACOBIAN_TEST, DEBUG_RF_TRAINING_SCALE)
   --decoding_feature_extraction_dictionary_transpose:share(base_decoding_feature_extraction_dictionary, 'weight', 'gradWeight') -- this doesn't work, since it needs to be the transpose
   decoding_feature_extraction_dictionary_transpose.weight:set(base_decoding_feature_extraction_dictionary.weight:t())
   decoding_feature_extraction_dictionary_transpose.gradWeight:set(base_decoding_feature_extraction_dictionary.gradWeight:t())
   decoding_feature_extraction_dictionary_transpose.accUpdateGradParameters = decoding_feature_extraction_dictionary_transpose.sharedAccUpdateGradParameters
   return decoding_feature_extraction_dictionary_copy, decoding_feature_extraction_dictionary_transpose
end

-- the input is a table of three elements: the subject of the pooling operation s [1], the preserved shrink output z [2], and the original input x [3]
-- the output is a table of two elements: the subject of the pooling operation s [1] and the original input x [2]
local function build_pooling_L2_loss(decoding_pooling_dictionary, decoding_feature_extraction_dictionary, decoding_feature_extraction_dictionary_transpose, mask_sparsifying_module, lambdas, criteria_list)
   -- the L2 reconstruction error and the L2 position unit magnitude both depend upon the denominator lambda + (P*s)^2, so calculate them together in a single function
   --lambdas include: lambdas.pooling_L2_shrink_reconstruction_lambda, lambdas.pooling_L2_orig_reconstruction_lambda, lambdas.pooling_L2_shrink_position_unit_lambda, lambdas.pooling_L2_orig_position_unit_lambda,

   local USE_SQRT_POOLING_RECONSTRUCTION = true

   local pooling_L2_loss_seq = nn.Sequential()
   -- split into the output from the module s = sqrt(Q*z^2) [1], the basis of the denominator of the losses s [2], the preserved shrink output z [3], the original input x [4], and the basis for the projected original input D^t*x [5]
   pooling_L2_loss_seq:add(nn.CopyTable(1, 2)) -- copy the subject of the pooling operation s [1] to [2]
   pooling_L2_loss_seq:add(nn.CopyTable(4, 2)) -- copy the original input x [4] to [5]

   -- split into s [1], P*s [2], z [3], the original input x [4], and the projected original input D^t*x [5]
   local reconstruction_parallel = nn.ParallelTable() -- compute the reconstruction of the input P*s
   reconstruction_parallel:add(nn.Identity()) -- preserve the output s = sqrt(Q*z^2) [1]
   reconstruction_parallel:add(decoding_pooling_dictionary) -- compute the reconstruction P*s [2]
   reconstruction_parallel:add(nn.Identity()) -- preserve the shrink output z [3]
   reconstruction_parallel:add(nn.Identity()) -- preserve the original input x [4]
   reconstruction_parallel:add(decoding_feature_extraction_dictionary_transpose) -- compute D^t *x [5]
   pooling_L2_loss_seq:add(reconstruction_parallel)

   -- compute the sparsifying regularizer on the pooling mask P*s [2]; output is still s [1], P*s [2], z [3], the original input x [4], and the projected original input D^t*x [5]
   local sparsifying_loss_seq = nn.Sequential()
   sparsifying_loss_seq:add(nn.SelectTable{2})
   if DEBUG_OUTPUT then 
      sparsifying_loss_seq:add(nn.ZeroModule())
   else
      sparsifying_loss_seq:add(mask_sparsifying_module) -- also compute the sparsifying norm on the code
      table.insert(criteria_list.criteria, mask_sparsifying_module) -- make sure that we consider the sparsifying_loss_function when evaluating the total loss
      table.insert(criteria_list.names, 'sparsifying mask loss')
      print('inserting sparsifying mask loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
   end
   sparsifying_loss_seq:add(nn.Ignore()) -- don't pass the sparsifying loss value onto the rest of the network
   local sparsifying_loss_parallel = nn.ParallelDistributingTable() -- we could probably use a CopyTable with zero repeats here
   sparsifying_loss_parallel:add(nn.SelectTable{1}) -- s [1]
   sparsifying_loss_parallel:add(nn.SelectTable{2}) -- P*s [2] -- NORMAL VERSION!!!  
   --local double_rec_seq = nn.Sequential() -- DEBUG ONLY!!!
   --double_rec_seq:add(nn.SelectTable{2}) -- DEBUG ONLY!!!
   --double_rec_seq:add(nn.MulConstant(decoding_pooling_dictionary.weight:size(1), 1.5)) -- DEBUG ONLY!!!
   --sparsifying_loss_parallel:add(double_rec_seq) -- DEBUG ONLY!!!
   sparsifying_loss_parallel:add(nn.SelectTable{3}) -- z [3]
   sparsifying_loss_parallel:add(nn.SelectTable{4}) -- x [4] 
   sparsifying_loss_parallel:add(nn.SelectTable{5}) -- D^t*x [5] 
   sparsifying_loss_parallel:add(sparsifying_loss_seq) -- produces no output
   pooling_L2_loss_seq:add(sparsifying_loss_parallel)


   -- the output from the module s [1], the original input x [2], the numerator of the shrink reconstruction loss lambda_ratio*z [3], the numerator of the shrink position loss z*(P*s) [4], 
   -- the numerator of the original reconstruction loss z*(P*s)^2 [5], the numerator of the original position loss D^t*x*(P*s) [6], and the denominator of the loss lambda_ratio + (P*s)^2 [7] 
   -- lambda_ratio arises from minimizing the sum of the shrink reconstruction and the L2 position unit error with respect to the position units, in order to find the feedforward value of the position units
   local lambda_ratio = math.min((lambdas.pooling_L2_shrink_position_unit_lambda + lambdas.pooling_L2_orig_position_unit_lambda) / (lambdas.pooling_L2_shrink_reconstruction_lambda + lambdas.pooling_L2_orig_reconstruction_lambda), 1e5) -- bound the lambda_ratio, so signals and gradients remain finite even if L2_reconstruction_lambda = 0
   if lambda_ratio ~= lambda_ratio then lambda_ratio = 1 end-- if lambda_ratio == nan, set it to 1
   local construct_shrink_rec_numerator_seq = nn.Sequential() -- lambda_ratio * z
   construct_shrink_rec_numerator_seq:add(nn.SelectTable{3})
   construct_shrink_rec_numerator_seq:add(nn.MulConstant(decoding_pooling_dictionary.weight:size(1), lambda_ratio))  -- first argument just sets the size of the output, rather than the constant
   local construct_shrink_pos_numerator_seq = nn.Sequential() -- z*(P*s)
   construct_shrink_pos_numerator_seq:add(nn.SelectTable{2,3}) 
   construct_shrink_pos_numerator_seq:add(nn.SafeCMulTable())
   local construct_orig_rec_numerator_seq = nn.Sequential() -- z*(P*s)^2 
   construct_orig_rec_numerator_seq:add(nn.SelectTable{2,3})
   local square_Ps_for_orig_rec_numerator = nn.ParallelTable()
   --square_Ps_for_orig_rec_numerator:add(nn.SafePower(2)) -- EFFICIENCY NOTE: nn.Square is more efficient, but computes gradInput incorrectly on 1-D inputs
   square_Ps_for_orig_rec_numerator:add(nn.Square()) -- EFFICIENCY NOTE: nn.Square is more efficient, but computes gradInput incorrectly on 1-D inputs
   square_Ps_for_orig_rec_numerator:add(nn.Identity())
   construct_orig_rec_numerator_seq:add(square_Ps_for_orig_rec_numerator)
   construct_orig_rec_numerator_seq:add(nn.SafeCMulTable())
   local construct_orig_pos_numerator_seq = nn.Sequential() -- D^t*x*(P*s)
   construct_orig_pos_numerator_seq:add(nn.SelectTable{2,5}) 
   construct_orig_pos_numerator_seq:add(nn.SafeCMulTable())
   local construct_denominator_seq = nn.Sequential() -- lambda_ratio + (P*s)^2
   construct_denominator_seq:add(nn.SelectTable{2}) 
   --construct_denominator_seq:add(nn.Square()) -- EFFICIENCY NOTE: nn.Square is almost certainly more efficient than Power(2), but it computes gradInput incorrectly on 1-D inputs; specifically, if fails to multiply the gradInput by two
   --construct_denominator_seq:add(nn.SafePower(2))
   construct_denominator_seq:add(nn.Square())
   construct_denominator_seq:add(nn.AddConstant(decoding_pooling_dictionary.weight:size(1), lambda_ratio)) 

   -- alternate terms used when the position loss is ||sqrt(P*s)*y||^2 rather than ||y||^2, where y is the L2-normalized position units
   local construct_alt_pos_numerator_seq = nn.Sequential() -- z*(P*s)^0.5 
   construct_alt_pos_numerator_seq:add(nn.SelectTable{2,3})
   local safe_sqrt_seq = nn.Sequential()
   local safe_sqrt_const = 1e-7
   safe_sqrt_seq:add(nn.AddConstant(decoding_pooling_dictionary.weight:size(1), safe_sqrt_const)) -- ensures that we never compute the gradient of sqrt(0)
   safe_sqrt_seq:add(nn.Sqrt())
   safe_sqrt_seq:add(nn.AddConstant(decoding_pooling_dictionary.weight:size(1), -math.sqrt(safe_sqrt_const))) -- we add 1e-4 before the square root and subtract (1e-4)^1/2 after the square root, so zero is still mapped to zero, but the gradient contribution is bounded above by 100
   local sqrt_Ps_for_alt_pos_numerator = nn.ParallelTable()
   sqrt_Ps_for_alt_pos_numerator:add(safe_sqrt_seq) 
   sqrt_Ps_for_alt_pos_numerator:add(nn.Identity())
   construct_alt_pos_numerator_seq:add(sqrt_Ps_for_alt_pos_numerator)
   construct_alt_pos_numerator_seq:add(nn.SafeCMulTable())
   local construct_alt_denominator_seq = nn.Sequential() -- lambda_ratio + (P*s)
   construct_alt_denominator_seq:add(nn.SelectTable{2}) 
   construct_alt_denominator_seq:add(nn.AddConstant(decoding_pooling_dictionary.weight:size(1), lambda_ratio)) 

   local construct_num_denom_parallel = nn.ParallelDistributingTable('construct_num_denom_parallel') -- put it all together
   construct_num_denom_parallel:add(nn.SelectTable{1}) -- preserve the output s = sqrt(Q*z^2) [1]
   construct_num_denom_parallel:add(nn.SelectTable{4}) -- preserve the original input x [2]
   construct_num_denom_parallel:add(construct_shrink_rec_numerator_seq) -- compute the shrink reconstruction numerator lambda_ratio * z [3]
   construct_num_denom_parallel:add(construct_shrink_pos_numerator_seq) -- compute the shrink position numerator z*(P*s) [4]
   construct_num_denom_parallel:add(construct_orig_rec_numerator_seq) -- compute the original input reconstruction numerator z*(P*s)^2 [5]
   construct_num_denom_parallel:add(construct_orig_pos_numerator_seq) -- compute the original input position numerator D^t*x*(P*s) [6]
   construct_num_denom_parallel:add(construct_denominator_seq) -- compute the denominator lambda_ratio + (P*s)^2 [7]

   construct_num_denom_parallel:add(construct_alt_pos_numerator_seq) -- compute the shrink position numerator with modified L2 position loss z*(P*s)^0.5 [8]
   construct_num_denom_parallel:add(construct_alt_denominator_seq) -- compute the denominator with modified L2 position loss lambda_ratio + (P*s) [9]
   pooling_L2_loss_seq:add(construct_num_denom_parallel)
   

   -- Build the four loss functions.  They will all be plugged into a parallel distributing table, and so can begin with a SelectTable

   -- compute the shrink reconstruction loss ||z - (z*(P*s)^2)/(lambda_ratio + (P*s)^2)||^2 = ||lambda_ratio * z / (lambda_ratio + (P*s)^2)||^2
   local compute_shrink_reconstruction_loss_seq = nn.Sequential()
   if USE_SQRT_POOLING_RECONSTRUCTION then 
      compute_shrink_reconstruction_loss_seq:add(nn.SelectTable{3,9}) -- DEBUG ONLY!!! SHOULD BE {3,7} to use normal losses
   else
      compute_shrink_reconstruction_loss_seq:add(nn.SelectTable{3,7}) -- NORMAL VERSION!!!
   end
   compute_shrink_reconstruction_loss_seq:add(nn.CDivTable())
   if DEBUG_OUTPUT then 
      compute_shrink_reconstruction_loss_seq:add(nn.ZeroModule())
   else
      local L2_shrink_reconstruction_loss = nn.L2Cost(lambdas.pooling_L2_shrink_reconstruction_lambda, 1)
      table.insert(criteria_list.criteria, L2_shrink_reconstruction_loss) -- make sure that we consider the L2_shrink_reconstruction_loss when evaluating the total loss
      table.insert(criteria_list.names, 'pooling L2 shrink reconstruction loss')
      print('inserting L2 pooling shrink reconstruction loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')

      compute_shrink_reconstruction_loss_seq:add(L2_shrink_reconstruction_loss)
   end
   compute_shrink_reconstruction_loss_seq:add(nn.Ignore)
   

   -- compute the original reconstruction loss ||x - D* (z*(P*s)^2 / (lambda_ratio + (P*s)^2))||^2 
   local compute_orig_reconstruction_loss_seq = nn.Sequential()
   compute_orig_reconstruction_loss_seq:add(nn.SelectTable{2,5,7})
   local divide_input_rec_seq = nn.Sequential()
   divide_input_rec_seq:add(nn.SelectTable{2,3})
   divide_input_rec_seq:add(nn.CDivTable())
   divide_input_rec_seq:add(decoding_feature_extraction_dictionary)
   local divide_input_rec_parallel = nn.ParallelDistributingTable('divide_input_rec')
   divide_input_rec_parallel:add(nn.SelectTable{1})
   divide_input_rec_parallel:add(divide_input_rec_seq)
   compute_orig_reconstruction_loss_seq:add(divide_input_rec_parallel)
   if DEBUG_OUTPUT then 
      local parallel_zero = nn.ParallelTable() 
      parallel_zero:add(nn.ZeroModule())
      parallel_zero:add(nn.ZeroModule())
      compute_orig_reconstruction_loss_seq:add(parallel_zero)
      compute_orig_reconstruction_loss_seq:add(nn.SelectTable{1}) -- a SelectTable is necessary to ensure that the module outputs a single nil, which is ignored by the ParallelDistributingTable, rather than an empty 
   else
      local L2_orig_reconstruction_loss = nn.L2Cost(lambdas.pooling_L2_orig_reconstruction_lambda, 2)
      table.insert(criteria_list.criteria, L2_orig_reconstruction_loss) -- make sure that we consider the L2_orig_reconstruction_loss when evaluating the total loss
      table.insert(criteria_list.names, 'pooling L2 orig reconstruction loss')
      print('inserting L2 pooling orig reconstruction loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
      
      compute_orig_reconstruction_loss_seq:add(L2_orig_reconstruction_loss)
   end
   compute_orig_reconstruction_loss_seq:add(nn.Ignore)


   -- compute the shrink position loss ||z*(P*s) / (lambda_ratio + (P*s)^2)||^2
   local compute_shrink_position_loss_seq = nn.Sequential()
   if USE_SQRT_POOLING_RECONSTRUCTION then
      compute_shrink_position_loss_seq:add(nn.SelectTable{8,9}) -- DEBUG ONLY!!!
   else
      compute_shrink_position_loss_seq:add(nn.SelectTable{4,7}) -- NORMAL VERSION
   end
   local compute_shrink_position_units = nn.CDivTable()
   compute_shrink_position_loss_seq:add(compute_shrink_position_units)
   if DEBUG_OUTPUT then 
      compute_shrink_position_loss_seq:add(nn.ZeroModule())
   else
      local L2_shrink_position_loss = nn.L2Cost(lambdas.pooling_L2_shrink_position_unit_lambda, 1)
      table.insert(criteria_list.criteria, L2_shrink_position_loss) -- make sure that we consider the L2_shrink_position_loss when evaluating the total loss
      table.insert(criteria_list.names, 'pooling L2 shrink position loss')
      print('inserting L2 pooling shrink position loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
      
      compute_shrink_position_loss_seq:add(L2_shrink_position_loss)
   end
   compute_shrink_position_loss_seq:add(nn.Ignore)


   -- compute the original input position loss ||D^t*x*(P*s) / (lambda_ratio + (P*s)^2)||^2
   local compute_orig_position_loss_seq = nn.Sequential()
   compute_orig_position_loss_seq:add(nn.SelectTable{6,7})
   local compute_orig_position_units = nn.CDivTable()
   compute_orig_position_loss_seq:add(compute_orig_position_units)
   if DEBUG_OUTPUT then 
      compute_orig_position_loss_seq:add(nn.ZeroModule())
   else
      local L2_orig_position_loss = nn.L2Cost(lambdas.pooling_L2_orig_position_unit_lambda, 1)
      table.insert(criteria_list.criteria, L2_orig_position_loss) -- make sure that we consider the L2_orig_position_loss when evaluating the total loss
      table.insert(criteria_list.names, 'pooling L2 orig position loss')
      print('inserting L2 pooling orig position loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
      
      compute_orig_position_loss_seq:add(L2_orig_position_loss)
   end
   compute_orig_position_loss_seq:add(nn.Ignore)


   -- put everything together
   local compute_loss_parallel = nn.ParallelDistributingTable('compute_loss_parallel') 
   compute_loss_parallel:add(nn.SelectTable{1}) -- preserve the output s = sqrt(Q*z^2) [1]
   compute_loss_parallel:add(nn.SelectTable{2}) -- preserve the original input x [2]
   compute_loss_parallel:add(compute_shrink_reconstruction_loss_seq) -- compute the shrink reconstruction loss ||lambda_ratio * z / (lambda_ratio + (P*s)^2)||^2 [-]
   compute_loss_parallel:add(compute_orig_reconstruction_loss_seq) -- compute the original reconstruction loss ||x - z*(P*s)^2 / (lambda_ratio + (P*s)^2)||^2 [-]
   compute_loss_parallel:add(compute_shrink_position_loss_seq) -- compute the position loss ||z*(P*s) / (lambda_ratio + (P*s)^2)||^2 [-]
   compute_loss_parallel:add(compute_orig_position_loss_seq) -- compute the original input position loss ||D^t*x*(P*s) / (lambda_ratio + (P*s)^2)||^2 [-]
   pooling_L2_loss_seq:add(compute_loss_parallel)

   local debug_modules = {pooling_L2_loss_seq = pooling_L2_loss_seq, 
			  compute_shrink_position_units = compute_shrink_position_units, 
			  compute_orig_position_units = compute_orig_position_units, 
			  compute_shrink_reconstruction_loss_seq = compute_shrink_reconstruction_loss_seq, 
			  compute_orig_reconstruction_loss_seq = compute_orig_reconstruction_loss_seq,
			  compute_shrink_position_loss_seq = compute_shrink_position_loss_seq, 
			  compute_orig_position_loss_seq = compute_orig_position_loss_seq, 
			  construct_shrink_rec_numerator_seq = construct_shrink_rec_numerator_seq,
			  construct_shrink_pos_numerator_seq = construct_shrink_pos_numerator_seq, 
			  construct_orig_rec_numerator_seq = construct_orig_rec_numerator_seq, 
			  construct_orig_pos_numerator_seq = construct_orig_pos_numerator_seq, 
			  construct_denominator_seq = construct_denominator_seq}
   return pooling_L2_loss_seq, debug_modules
end



-- recpool_config_prefs are num_ista_iterations, shrink_style, disable_pooling, use_squared_weight_matrix, normalize_each_layer
-- lambdas consists of an array of arrays, so it's difficult to make an off-by-one-error when initializing
-- build a stack of reconstruction-pooling layers, followed by a classfication dictionary and associated criterion
function build_recpool_net(layer_size, lambdas, classification_criterion_lambda, lagrange_multiplier_targets, lagrange_multiplier_learning_rate_scaling_factors, recpool_config_prefs, data_set, RUN_JACOBIAN_TEST)
   -- lambdas: {ista_L2_reconstruction_lambda, ista_L1_lambda, pooling_L2_shrink_reconstruction_lambda, pooling_L2_orig_reconstruction_lambda, pooling_L2_shrink_position_unit_lambda, pooling_L2_orig_position_unit_lambda, pooling_output_cauchy_lambda, pooling_mask_cauchy_lambda}
   -- disable_pooling specifies that the pooling layers should be created (otherwise all of the test code that depends on their existence will fail), but that they should not be added to model = nn.Sequential().  As a result, they should not slow down the running time at all.  

   local criteria_list = {criteria = {}, names = {}} -- list of all criteria and names comprising the loss function.  These are necessary to run the Jacobian unit test forwards
   RUN_JACOBIAN_TEST = RUN_JACOBIAN_TEST or false
   
   recpool_config_prefs.shrink_style = recpool_config_prefs.shrink_style or 'ParameterizedShrink'

   local model = nn.Sequential()
   local layer_list = {} -- an array of the component layers for easy access
   -- size of the classification dictionary depends upon whether we're using pooling or not
   local classification_dictionary
   if not(recpool_config_prefs.disable_pooling) then
      classification_dictionary = nn.Linear(layer_size[#layer_size-1], layer_size[#layer_size])
      --classification_dictionary = nn.ConstrainedLinear(layer_size[#layer_size-1], layer_size[#layer_size], {normalized_rows = true})
   else
      -- if we're not pooling, then the classification dictionary needs to map from the feature extraction layer to the classes, rather than from the pooling layer to the classes
      classification_dictionary = nn.Linear(layer_size[#layer_size-2], layer_size[#layer_size])
   end
   local classification_criterion = nn.L1CriterionModule(nn.ClassNLLCriterion(), classification_criterion_lambda) -- on each iteration classfication_criterion:setTarget(target) must be called
   --local classification_criterion = nn.L1CriterionModule(nn.MSECriterion(), classification_criterion_lambda) -- DEBUG ONLY!!! FOR THE LOVE OF GOD!!!

   classification_dictionary.bias:zero()

   -- make the criteria_list and filters accessibel from the nn.Sequential module for ease of debugging and reporting
   model.criteria_list = criteria_list
   -- each recpool layer generates a separate filter list, which we combine into a common list so we can easily generate collective diagnostics
   model.filter_list = {}
   model.filter_enc_dec_list = {}
   model.filter_name_list = {}
   
   model.layers = layer_list -- used in train_recpool_net to access debug information
   model.disable_pooling = recpool_config_prefs.disable_pooling

   -- take the initial input x and wrap it in a table x [1]
   model:add(nn.IdentityTable()) -- wrap the tensor in a table

   for layer_i = 1,#lambdas do
      -- each layer has an input stage, a feature extraction stage, and a pooling stage.  Since the pooling output of one layer is the input of the next, increment the layer_size index by two per layer
      local current_layer_size = {}
      for i = 1,3 do
	 current_layer_size[i] = layer_size[i + (layer_i-1)*2] 
      end

      -- the global criteria_list is passed into each layer to be built up progressively
      local current_layer = build_recpool_net_layer(layer_i, current_layer_size, lambdas[layer_i], lagrange_multiplier_targets[layer_i], lagrange_multiplier_learning_rate_scaling_factors[layer_i], recpool_config_prefs, criteria_list, data_set, RUN_JACOBIAN_TEST) 
      model:add(current_layer)
      layer_list[layer_i] = current_layer -- this is used in a closure for model:repair()

      -- add this layer's filters to the collective filter lists
      local current_filter_list = {current_layer.module_list.encoding_feature_extraction_dictionary.weight, current_layer.module_list.decoding_feature_extraction_dictionary.weight, 
				   current_layer.module_list.explaining_away.weight, current_layer.module_list.encoding_pooling_dictionary.weight, 
				   current_layer.module_list.decoding_pooling_dictionary.weight, current_layer.module_list.encoding_pooling_dictionary.gradWeight
      }
      local current_filter_enc_dec_list = {'encoder', 'decoder', 'encoder', 'encoder', 'decoder', 'encoder'}
      local current_filter_name_list = {'encoding feature extraction dictionary', 'decoding feature extraction dictionary', 'explaining away', 'encoding pooling dictionary', 'decoding pooling dictionary', 'encoding pooling dictionary gradient'}
      
      for i,v in ipairs(current_filter_list) do table.insert(model.filter_list, v) end
      for i,v in ipairs(current_filter_enc_dec_list) do table.insert(model.filter_enc_dec_list, v) end
      for i,v in ipairs(current_filter_name_list) do table.insert(model.filter_name_list, v .. '_' .. layer_i) end
   end

   -- add the classification dictionary to the collective filter lists
   table.insert(model.filter_list, classification_dictionary.weight)
   table.insert(model.filter_enc_dec_list, 'encoder')
   table.insert(model.filter_name_list, 'classification dictionary')

   model:add(nn.SelectTable{1}) -- unwrap the pooled output from the table
   model:add(classification_dictionary)
   model:add(nn.LogSoftMax())
   --model:add(nn.SoftMax()) -- DEBUG ONLY!!! FOR THE LOVE OF GOD!!!
   
   model.module_list = {classification_dictionary = classification_dictionary}

   if DEBUG_OUTPUT then
      function model:set_target(new_target) print('WARNING: set_target does nothing when using DEBUG_OUTPUT') end
      if IGNORE_CLASSIFICATION then error('cannot use both DEBUG_OUTPUT and IGNORE_CLASSIFICATION simultaneously') end
   else
      if IGNORE_CLASSIFICATION then 
	 model:add(nn.ZeroModule()) -- use if we want to remove the classification criterion for debugging purposes
	 model:add(nn.Ignore())
      else
	 model:add(classification_criterion) 

	 table.insert(criteria_list.criteria, classification_criterion)
	 table.insert(criteria_list.names, 'classification criterion')
	 print('inserting classification negative log likelihood loss into criteria list, resulting in ' .. #(criteria_list.criteria) .. ' entries')
      end

      function model:set_target(new_target) 
	 --print('setting classification criterion target to ', new_target)
	 classification_criterion:setTarget(new_target) 
      end 

      function model:reset_classification_lambda(new_lambda)
	 classification_criterion:reset_lambda(new_lambda)
      end

      local original_updateOutput = model.updateOutput -- by making this local, it is impossible to access outside of the closures created below
      
      -- note that this is different than the original model.output; model is a nn.Sequential, which ends in the classification_criterion, and thus consists of a single number
      -- it is not desirable to have the model produce a tensor output, since this would imply a corresponding gradOutput, at least in the standard implementation, whereas we want all gradients to be internally generated.  
      function model:updateOutput(input)
	 original_updateOutput(self, input)
	 local summed_loss = 0
	 for i = 1,#(criteria_list.criteria) do
	    if criteria_list.criteria[i].output then -- some criteria may not be defined if we've disabled pooling or other layers
	       summed_loss = summed_loss + criteria_list.criteria[i].output
	    end
	 end
	 -- the output of a tensor is necessary to maintain compatibility with ModifiedJacobian, which directly accesses the output field of the tested module
	 model.output = torch.Tensor(1)
	 model.output[1] = summed_loss
	 return model.output -- while it is tempting to return classification_dictionary.output here, it risks incompatibility with ModifiedJacobian and any other package that expects a single return value
      end

      function model:get_classifier_output()
	 return classification_dictionary.output
      end

      function model:repair()  -- repair each layer; classification dictionary is a normal linear, and so does not need to be repaired
	 for i = 1,#layer_list do
	    layer_list[i]:repair()
	 end
	 --classification_dictionary:repair()
      end

      function model:reset_learning_scale_factor(new_scale_factor)  -- reset the learning scale factor for each layer; classification dictionary is left unchanged
	 for i = 1,#layer_list do
	    layer_list[i]:reset_learning_scale_factor(new_scale_factor)
	 end
      end

      function model:randomize_pooling()
	 for i = 1,#layer_list do
	    layer_list[i]:randomize_pooling()
	 end
      end
   end -- not(DEBUG_OUTPUT)

   --model:add(nn.ParallelDistributingTable('throw away final output')) -- if no modules are added to a ParallelDistributingTable, it throws away its input; updateGradInput produces a tensor of zeros   

   print('criteria_list contains ' .. #(criteria_list.criteria) .. ' entries')      

   return model
end


-- input is a table of one element x [1], output is a table of one element s [1]
-- a reconstructing-pooling network.  This is like reconstruction ICA, but with reconstruction applied to both the feature extraction and the pooling, and using shrink operators rather than linear transformations for the feature extraction.  The initial version of this network is built with simple linear transformations, but it can just as easily be used to convolutions
-- use RUN_JACOBIAN_TEST when testing parameter updates
function build_recpool_net_layer(layer_id, layer_size, lambdas, lagrange_multiplier_targets, lagrange_multiplier_learning_rate_scaling_factors, recpool_config_prefs, criteria_list, data_set, RUN_JACOBIAN_TEST) 
   -- lambdas: {ista_L2_reconstruction_lambda, ista_L1_lambda, pooling_L2_shrink_reconstruction_lambda, pooling_L2_orig_reconstruction_lambda, pooling_L2_shrink_position_unit_lambda, pooling_L2_orig_position_unit_lambda, pooling_output_cauchy_lambda, pooling_mask_cauchy_lambda}
   if not(criteria_list) then -- if we're building a multi-layer network, a common criteria list is passed in
      criteria_list = {criteria = {}, names = {}} -- list of all criteria and names comprising the loss function.  These are necessary to run the Jacobian unit test forwards
   end
   RUN_JACOBIAN_TEST = RUN_JACOBIAN_TEST or false
   local disable_trainable_pooling = false
   if DEBUG_RF_TRAINING_SCALE == 0 then
      disable_trainable_pooling = true
      print('Disabling trainable pooling!')
      io.read()
   end
   
   if recpool_config_prefs.use_squared_weight_matrix == nil then
      error('recpool_config_prefs.use_squared_weight_matrix was not defined')
   end
   local use_swm = recpool_config_prefs.use_squared_weight_matrix

   -- Build the reconstruction pooling network
   local this_layer = nn.Sequential()

   -- the exact range for the initialization of weight matrices by nn.Linear doesn't matter, since they are rescaled by the normalized_columns constraint
   -- threshold-normalized rows are a bad idea for the encoding feature extraction dictionary, since if a feature is not useful, it will be turned off via the shrinkage, and will be extremely difficult to reactivate later.  It's better to allow the encoding dictionary to be reduced in magnitude.
   local encoding_feature_extraction_dictionary = nn.ConstrainedLinear(layer_size[1],layer_size[2], {no_bias = (recpool_config_prefs.shrink_style == 'ParameterizedShrink'), normalized_rows = false}, RUN_JACOBIAN_TEST, DEBUG_RF_TRAINING_SCALE) 
   local base_decoding_feature_extraction_dictionary = nn.ConstrainedLinear(layer_size[2],layer_size[1], {no_bias = true, normalized_columns = true}, RUN_JACOBIAN_TEST, DEBUG_RF_TRAINING_SCALE) 
   local base_explaining_away = nn.ConstrainedLinear(layer_size[2], layer_size[2], {no_bias = true, non_negative_diag = false}, RUN_JACOBIAN_TEST, DEBUG_RF_TRAINING_SCALE) 
   local base_shrink 
   if recpool_config_prefs.shrink_style == 'ParameterizedShrink' then
      base_shrink = nn.ParameterizedShrink(layer_size[2], FORCE_NONNEGATIVE_SHRINK_OUTPUT, DEBUG_shrink)
   elseif recpool_config_prefs.shrink_style == 'FixedShrink' then
      base_shrink = nn.FixedShrink(layer_size[2])
   elseif recpool_config_prefs.shrink_style == 'SoftPlus' then
      base_shrink = nn.SoftPlus()
   else
      error('shrink style ' .. recpool_config_prefs.shrink_style .. ' was not recognized')
   end

   local explaining_away_copies = {}
   local shrink_copies = {}

   local pooling_dictionary_scaling_factor = 1
   local encoding_pooling_dictionary = nn.ConstrainedLinear(layer_size[2], layer_size[3], {no_bias = true, non_negative = true, normalized_rows_pooling = false, squared_weight_matrix = use_swm}, RUN_JACOBIAN_TEST, (disable_trainable_pooling and 0) or pooling_dictionary_scaling_factor) -- this should have zero bias

   local dpd_training_scale_factor = 1 -- factor by which training of decoding_pooling_dictionary is accelerated
   if not(RUN_JACOBIAN_TEST) then 
      dpd_training_scale_factor = (disable_trainable_pooling and 0) or pooling_dictionary_scaling_factor --5 -- decoding_pooling_dictionary is trained faster than any other module
      --dpd_training_scale_factor = 0 -- DEBUG ONLY!!!
   else -- make sure that all lagrange_multiplier_scaling_factors are -1 when testing, so the update matches the gradient
      for k,v in pairs(lagrange_multiplier_learning_rate_scaling_factors) do
	 if v ~= -1 then
	    error('When doing jacobian test, lagrange_multiplier_scaling_factor ' .. k .. ' was ' .. v .. ' rather than -1')
	 end
      end
   end
   
   if (dpd_training_scale_factor ~= 1) and (DEBUG_shrink or DEBUG_L2 or DEBUG_L1 or DEBUG_OUTPUT) then
      print('SET decoding_pooling_dictionary training scale factor to 1 before testing!!!  Alternatively, turn off debug mode')
      io.read()
   end
   local decoding_pooling_dictionary = nn.ConstrainedLinear(layer_size[3], layer_size[2], {no_bias = true, normalized_columns_pooling = true, non_negative = true, squared_weight_matrix = false}, RUN_JACOBIAN_TEST, dpd_training_scale_factor) -- this should have zero bias, and columns normalized to unit magnitude

   -- there's really no reason to define these here rather than where they're used
   local use_lagrange_multiplier_for_L1_regularizer = false
   local feature_extraction_sparsifying_module, pooling_sparsifying_module, mask_sparsifying_module
   if use_lagrange_multiplier_for_L1_regularizer then
      --feature_extraction_sparsifying_module = nn.ParameterizedL1Cost(layer_size[2], lambdas.ista_L1_lambda, lagrange_multiplier_targets.feature_extraction_target, lagrange_multiplier_learning_rate_scaling_factors.feature_extraction_scaling_factor, RUN_JACOBIAN_TEST)
      feature_extraction_sparsifying_module = nn.L1CriterionModule(nn.L1Cost(), lambdas.ista_L1_lambda) 

      --pooling_sparsifying_module = nn.ParameterizedL1Cost(layer_size[3], lambdas.pooling_output_cauchy_lambda, lagrange_multiplier_targets.pooling_target, lagrange_multiplier_learning_rate_scaling_factors.pooling_scaling_factor, RUN_JACOBIAN_TEST) 
      pooling_sparsifying_module = nn.L1CriterionModule(nn.L1Cost(), lambdas.pooling_output_cauchy_lambda) 

      mask_sparsifying_module = nn.ParameterizedL1Cost(layer_size[2], lambdas.pooling_mask_cauchy_lambda, lagrange_multiplier_targets.mask_target, lagrange_multiplier_learning_rate_scaling_factors.mask_scaling_factor, RUN_JACOBIAN_TEST) 
      --mask_sparsifying_module = nn.L1CriterionModule(nn.L1Cost(), lambdas.pooling_mask_cauchy_lambda) 
   else
      -- the L1CriterionModule, rather than the wrapped criterion, produces the correct scaled error
      feature_extraction_sparsifying_module = nn.L1CriterionModule(nn.L1Cost(), lambdas.ista_L1_lambda) 
      pooling_sparsifying_module = nn.L1CriterionModule(nn.L1Cost(), lambdas.pooling_output_cauchy_lambda) 
      --local pooling_sparsifying_loss_function = nn.L1Cost() --nn.CauchyCost(lambdas.pooling_output_cauchy_lambda) -- remove lambda from build function if we ever switch back to a cauchy cost!
      mask_sparsifying_module = nn.L1CriterionModule(nn.L1Cost(), lambdas.pooling_mask_cauchy_lambda) 
   end

   local copied_decoding_feature_extraction_dictionary, decoding_feature_extraction_dictionary_transpose = 
      duplicate_decoding_feature_extraction_dictionary(base_decoding_feature_extraction_dictionary, {layer_size[1], layer_size[2]}, RUN_JACOBIAN_TEST)
   
   
   -- it's easier to create a single module list, which can be indexed with pairs, than to add each separately to the nn.Sequential module corresponding to this layer
   -- there's a separate debug_module_list defined below that contains the internal nn.Sequential modules, so we can see the processing flow explicitly
   local module_list = {encoding_feature_extraction_dictionary = encoding_feature_extraction_dictionary, 
			decoding_feature_extraction_dictionary = base_decoding_feature_extraction_dictionary, 
			decoding_feature_extraction_dictionary_transpose = decoding_feature_extraction_dictionary_transpose,
			explaining_away = base_explaining_away, 
			shrink = base_shrink, 
			explaining_away_copies = explaining_away_copies,
			shrink_copies = shrink_copies,
			encoding_pooling_dictionary = encoding_pooling_dictionary, 
			decoding_pooling_dictionary = decoding_pooling_dictionary, 
			feature_extraction_sparsifying_module = feature_extraction_sparsifying_module, 
			pooling_sparsifying_module = pooling_sparsifying_module, 
			mask_sparsifying_module = mask_sparsifying_module}
   
   -- Initialize the parameters to consistent values -- this should probably go in a separate function
   --base_decoding_feature_extraction_dictionary.weight:apply(function(x) return ((x < 0) and 0) or x end) -- make the feature extraction dictionary non-negative, so activities don't need to be balanced in order to get a zero background
   if data_set and (layer_id == 1) then
      for i = 1,0 do --base_decoding_feature_extraction_dictionary.weight:size(2) do
	 base_decoding_feature_extraction_dictionary.weight:select(2,i):copy(data_set.data[math.random(data_set:size())])
      end
   end
   base_decoding_feature_extraction_dictionary:repair(true) -- make sure that the norm of each column is 1
   --print(base_decoding_feature_extraction_dictionary.weight)
   --io.read()
   encoding_feature_extraction_dictionary.weight:copy(base_decoding_feature_extraction_dictionary.weight:t())

   base_explaining_away.weight:copy(torch.mm(encoding_feature_extraction_dictionary.weight, base_decoding_feature_extraction_dictionary.weight)) -- the step constant should only be applied to explaining_away once, rather than twice
   if recpool_config_prefs.shrink_style == 'SoftPlus' then
      encoding_feature_extraction_dictionary.weight:mul(math.max(0.1, 10/(recpool_config_prefs.num_ista_iterations + 1)))
   else
      encoding_feature_extraction_dictionary.weight:mul(math.max(0.1, 1.25/(recpool_config_prefs.num_ista_iterations + 1))) -- the first ista iteration doesn't count towards num_ista_iterations
   end
   base_explaining_away.weight:mul(-math.max(0.1, 1.25/(recpool_config_prefs.num_ista_iterations + 1)))
   --[[
      -- this is only necessary when we preload the feature extraction dictionary with elements of the data set, in which case explaining_away has many strongly negative elements.  
      encoding_feature_extraction_dictionary.weight:mul(1e-1)
      base_explaining_away.weight:mul(0) 
   --]]
   for i = 1,base_explaining_away.weight:size(1) do -- add the identity matrix into base_explaining_away
      base_explaining_away.weight[{i,i}] = base_explaining_away.weight[{i,i}] + 1
   end
   
   if recpool_config_prefs.shrink_style == 'ParameterizedShrink' then
      base_shrink.shrink_val:fill(1e-5) --1e-4) --1e-5 -- this should probably be very small, and learn to be the appropriate size!!!
      --base_shrink.shrink_val:fill(1e-2) -- this is only necessary when we preload the feature extraction dictionary with elements from the data set
      base_shrink.negative_shrink_val:mul(base_shrink.shrink_val, -1)
   else
      encoding_feature_extraction_dictionary.bias:fill(-1e-5)
   end
      
   if not(recpool_config_prefs.randomize_pooling_dictionary) then
      decoding_pooling_dictionary.weight:zero() -- initialize to be roughly diagonal
      for i = 1,layer_size[2] do -- for each unpooled row entry, find the corresponding pooled column entry, plus those beyond it (according to the loop over j)
	 --print('setting entry ' .. i .. ', ' .. math.ceil(i * layer_size[3] / layer_size[2]))
	 for j = 0,0 do
	    decoding_pooling_dictionary.weight[{i, math.min(layer_size[3], j + math.ceil(i * layer_size[3] / layer_size[2]))}] = 1
	 end
      end
      
      --decoding_pooling_dictionary.weight:add(torch.rand(decoding_pooling_dictionary.weight:size()):mul(0.02))
      --for i = 1,layer_size[3] do
      --   decoding_pooling_dictionary.weight[{math.random(decoding_pooling_dictionary.weight:size(1)), i}] = 1
      --end
   end

   decoding_pooling_dictionary:repair(true) -- make sure that the norm of each column of decoding_pooling_dictionary is 1, even after it is thinned out
   encoding_pooling_dictionary.weight:copy(decoding_pooling_dictionary.weight:t())
   --encoding_pooling_dictionary.weight:mul(0.2) -- 1.25 --DEBUG ONLY!!!
   if use_swm then
      encoding_pooling_dictionary.weight:pow(0.5)
   end
   --decoding_pooling_dictionary.weight:add(torch.mul(torch.rand(decoding_pooling_dictionary.weight:size()), 0.3))
   --encoding_pooling_dictionary.weight:add(torch.mul(torch.rand(encoding_pooling_dictionary.weight:size()), 0.3))
   encoding_pooling_dictionary:repair(true) -- remember that encoding_pooling_dictionary does not have normalized columns
   --]]

   function this_layer:randomize_pooling(avoid_reset)
      if not(avoid_reset) then -- when using RUN_JACOBIAN_TEST, constraints are disabled after the first initialization.  A reset here will thus incorrectly include e.g. negative values.
	 decoding_pooling_dictionary:reset()
      end
      -- if we don't thin out the pooling dictionary a little, there is no symmetry breaking; all pooling units output about the same output for each input, so the only reliable way to decrease the L1 norm is by turning off all elements.
      --decoding_pooling_dictionary:percentage_zeros_per_column(1 - (1 - 0.8) * 200/layer_size[2]) --0.8) -- 0.9 works well! -- keep in mind that half of the values are negative, and will be set to zero when repaired
      decoding_pooling_dictionary:percentage_zeros_per_column(0.8)
      decoding_pooling_dictionary:repair(true) -- make sure that the norm of each column of decoding_pooling_dictionary is 1, even after it is thinned out
      --print('num ones is ' .. decoding_pooling_dictionary.weight:nElement() - torch.eq(decoding_pooling_dictionary.weight, 0):sum())
      --io.read()

      encoding_pooling_dictionary.weight:copy(decoding_pooling_dictionary.weight:t())
      if use_swm then
	 encoding_pooling_dictionary.weight:pow(0.5)
      end
      --encoding_pooling_dictionary.weight:mul(0.5) -- this helps the initial magnitude of the pooling reconstruction match the actual shrink output -- a value larger than 1 will probably bring the initial values closer to their final values; small columns in the decoding pooling dictionary yield even small reconstructions, since they cause the position units to be small as well
      encoding_pooling_dictionary:repair(true) -- remember that encoding_pooling_dictionary does not have normalized columns

      -- DEBUG ONLY
      --decoding_pooling_dictionary.weight:fill(1)
      --decoding_pooling_dictionary:repair(true)
   end

   if recpool_config_prefs.randomize_pooling_dictionary then
      this_layer:randomize_pooling(RUN_JACOBIAN_TEST)
   end
   --decoding_pooling_dictionary.weight:zero():add(torch.rand(decoding_pooling_dictionary.weight:size()))
   --decoding_pooling_dictionary:repair(true)



   -- take the input x [1] and calculate the sparse code z [1], the transformed input W*x [2], and the untransformed input x [3]
   local ista_seq
   --ista_seq, shrink_copies[#shrink_copies + 1] = build_ISTA_first_iteration(encoding_feature_extraction_dictionary, base_shrink)
   ista_seq = build_ISTA_first_iteration(encoding_feature_extraction_dictionary, base_shrink) -- this function does not actually copy shrink, so a shrink_copy does not need to be returned
   this_layer:add(ista_seq)

   for i = 1,recpool_config_prefs.num_ista_iterations do
      --calculate the sparse code z [1]; preserve the transformed input W*x [2], and the untransformed input x [3]
      local use_base_explaining_away = (i == 1) -- the base_explaining_away must be used once directly for parameter flattening in nn.Module to work properly; base_shrink is already used by build_ISTA_first_iteration
      local this_explaining_away = duplicate_linear_explaining_away(base_explaining_away, {layer_size[1], layer_size[2]}, use_base_explaining_away, RUN_JACOBIAN_TEST)
      local this_shrink 
      if recpool_config_prefs.shrink_style == 'ParameterizedShrink' then
	 this_shrink = duplicate_shrink(base_shrink, {layer_size[1], layer_size[2]}, RUN_JACOBIAN_TEST)
      elseif recpool_config_prefs.shrink_style == 'FixedShrink' then
	 this_shrink = nn.FixedShrink(layer_size[2])
      elseif recpool_config_prefs.shrink_style == 'SoftPlus' then
	 this_shrink = nn.SoftPlus()
      else
	 error('shrink style ' .. recpool_config_prefs.shrink_style .. ' was not recognized')
      end

      explaining_away_copies[#explaining_away_copies + 1], shrink_copies[#shrink_copies + 1] = this_explaining_away, this_shrink
      ista_seq = build_ISTA_iteration(this_explaining_away, this_shrink, RUN_JACOBIAN_TEST)
      this_layer:add(ista_seq)
   end
   -- reconstruct the input D*z [2] from the code z [1], leaving z [1] and x [3->2] unchanged
   this_layer:add(linearly_reconstruct_input(base_decoding_feature_extraction_dictionary, 3))
   -- calculate the L2 distance between the reconstruction based on the shrunk code D*z [2], and the original input x [3]; discard all signals but the current code z [1] and the original input x [2]
   this_layer:add(build_L2_reconstruction_loss(lambdas.ista_L2_reconstruction_lambda, criteria_list)) 
   -- calculate the L1 magnitude of the shrunk code z [1] (input also contains the original input x [2]), returning the shrunk code z [1] and original input x[2] unchanged


   -- compute the sparsifying loss on the shrunk code; input and output are the subject of the shrink operation z [1], and the original input x [2]
   local ista_sparsifying_loss_seq = build_sparsifying_loss(feature_extraction_sparsifying_module, criteria_list)
   this_layer:add(ista_sparsifying_loss_seq)

   -- pool the input z [1] to obtain the pooled code s = sqrt(Q*z^2) [1], the preserved input z [2], and the original input x[3]
   local pooling_seq = build_pooling(encoding_pooling_dictionary)
   if not(recpool_config_prefs.disable_pooling) then
      this_layer:add(pooling_seq)
   end


   --[[
   -- the gradient of the L2 position loss is only negative if P*s >= sqrt(alpha)
   -- add a constant additive offset to the pooling units
   local offset_pt = nn.ParallelTable()
   local lambda_ratio = math.min((lambdas.pooling_L2_shrink_position_unit_lambda + lambdas.pooling_L2_orig_position_unit_lambda) / (lambdas.pooling_L2_shrink_reconstruction_lambda + lambdas.pooling_L2_orig_reconstruction_lambda), 1e5)
   offset_pt:add(nn.AddConstant(layer_size[3], math.sqrt(lambda_ratio))) -- / (layer_size[2] * layer_size[3]))))
   offset_pt:add(nn.Identity())
   offset_pt:add(nn.Identity())
   this_layer:add(offset_pt)
   --]]
   
   --[[
   -- divisively normalize the pooling units before reconstruction
   local normalize_pt = nn.ParallelTable()
   local normalize_seq = nn.Sequential()
   --local normalize_pooled_output = nn.NormalizeTensor()
   --normalize_seq:add(normalize_pooled_output)
   normalize_seq:add(nn.MulConstant(layer_size[3], 1)) --50*math.sqrt(2)))
   normalize_pt:add(normalize_seq)
   normalize_pt:add(nn.Identity())
   normalize_pt:add(nn.Identity())
   this_layer:add(normalize_pt)
   --]]

   -- calculate the L2 reconstruction and position loss for pooling; return the pooled code s [1] and the original input x [2]
   local pooling_L2_loss_seq, pooling_L2_debug_module_list =
      build_pooling_L2_loss(decoding_pooling_dictionary, copied_decoding_feature_extraction_dictionary, decoding_feature_extraction_dictionary_transpose, mask_sparsifying_module, lambdas, criteria_list)
   if not(recpool_config_prefs.disable_pooling) then
      this_layer:add(pooling_L2_loss_seq)
   end

   -- calculate the L1 magnitude of the pooling code s [1] (also contains the original input x[2]), returning the pooling code s [1] and the original input x [2] unchanged
   local pooling_sparsifying_loss_seq = build_sparsifying_loss(pooling_sparsifying_module, criteria_list)
   if not(recpool_config_prefs.disable_pooling) then
      this_layer:add(pooling_sparsifying_loss_seq)
   end

   local extract_pooled_output = nn.ParallelDistributingTable('remove unneeded original input x') -- ensure that the original input x [2] receives a gradOutput of zero
   extract_pooled_output:add(nn.SelectTable{1})
   this_layer:add(extract_pooled_output)

   -- THIS SHOULD PROBABLY ONLY BE DONE BEFORE THE CLASSIFICATION LOSS!!!
   -- normalize the output of each layer; output is normalized s [1]
   local normalize_output = nn.NormalizeTable()
   this_layer:add(normalize_output) -- this is undefined if all outputs are zero -- DEBUG ONLY!!! FOR THE LOVE OF GOD!!!


   this_layer.module_list = module_list
   -- used when checking for nans in train_recpool_net
   this_layer.debug_module_list = pooling_L2_debug_module_list
   this_layer.debug_module_list.ista_sparsifying_loss_seq = ista_sparsifying_loss_seq
   this_layer.debug_module_list.pooling_seq = pooling_seq
   this_layer.debug_module_list.pooling_sparsifying_loss_seq = pooling_sparsifying_loss_seq
   this_layer.debug_module_list.normalize_output = normalize_output
   this_layer.debug_module_list.normalize_pooled_output = normalize_pooled_output

   -- the constraints on the parameters of these modules cannot be enforced by updateParameters when used in conjunction with the optim package, since it adjusts the parameters directly
   -- THIS IS A HACK!!!
   local repair_counter = 0
   --function this_layer:repair()
   --end
   
   function this_layer:repair()
      -- normalizing these two large dictionaries is the slowest part of the algorithm, consuming perhaps 75% of the running time.  Normalizing less often obviously increases running speed considerably.  We'll need to evaluate whether it's safe...
      if repair_counter % 5 == 0 then
	 encoding_feature_extraction_dictionary:repair(nil, math.max(0.1, 1.25/(recpool_config_prefs.num_ista_iterations + 1)))
	 base_decoding_feature_extraction_dictionary:repair(true) -- force full normalization of columns
      end
      repair_counter = (repair_counter + 1) % 5
      
      base_explaining_away:repair()
      if recpool_config_prefs.shrink_style == 'ParameterizedShrink' then
	 base_shrink:repair() -- repairing the base_shrink doesn't help if the parameters aren't linked!!!
      end
      if not(disable_trainable_pooling) then
	 encoding_pooling_dictionary:repair() --true) -- force full normalization of rows -- DEBUG ONLY!!!  FOR THE LOVE OF GOD!!  RECONSIDER THIS!!!
	 decoding_pooling_dictionary:repair(true) -- force full normalization of columns
      end
   end

   function this_layer:reset_learning_scale_factor(new_scale_factor)
      -- ALSO NEED TO RESET ALL OF THE COPIES!!!
      encoding_feature_extraction_dictionary:reset_learning_scale_factor(new_scale_factor)
      base_decoding_feature_extraction_dictionary:reset_learning_scale_factor(new_scale_factor)
      copied_decoding_feature_extraction_dictionary:reset_learning_scale_factor(new_scale_factor)
      decoding_feature_extraction_dictionary_transpose:reset_learning_scale_factor(new_scale_factor)
      base_explaining_away:reset_learning_scale_factor(new_scale_factor)
      for i = 1,#explaining_away_copies do
	 explaining_away_copies[i]:reset_learning_scale_factor(new_scale_factor)
      end
      encoding_pooling_dictionary:reset_learning_scale_factor(new_scale_factor)
      decoding_pooling_dictionary:reset_learning_scale_factor(new_scale_factor)
   end

   
   --return this_layer, criteria_list, encoding_feature_extraction_dictionary, base_decoding_feature_extraction_dictionary, encoding_pooling_dictionary, decoding_pooling_dictionary, feature_extraction_sparsifying_module, pooling_sparsifying_module, mask_sparsifying_module, base_explaining_away, base_shrink, explaining_away_copies, shrink_copies
   return this_layer
end

-- Test full reconstructing-pooling model with Jacobian.  Maintain a list of all Criteria.  When running forward, sum over all Criteria to determine the gradient of a single unified energy (should be able to just run updateOutput on the Criteria).  When running backward, just call updateGradInput on the network.  No gradOutput needs to be provided, since all modules terminate in an output-free Criterion.
