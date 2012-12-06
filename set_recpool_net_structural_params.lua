function set_recpool_net_structural_params(recpool_config_prefs, data, command_line_params, sparsity_scaling)
   sparsity_scaling = sparsity_scaling or 1

   local sl_mag = nil -- L1 loss on the sparse coding / feature extraction units
   local rec_mag = nil -- L2 reconstruction loss between the inputs and the sparse coding / feature extraction units
   local pooling_rec_mag = nil -- L2 reconstruction loss between the sparse coding / feature extraction units and the pooling units
   local pooling_orig_rec_mag = nil -- L2 reconstruction loss between the inputs and the pooling units
   local pooling_shrink_position_L2_mag = nil -- L2 loss on the implicit set of position units: ||z*(P*s) / (lambda_ratio + (P*s)^2)||^2
   local pooling_orig_position_L2_mag = nil -- L2 loss on the implicit set of position units, when reconstruction of the original inputs rather than the shrink units is optimized
   local pooling_sl_mag = nil -- L1 loss applied to the output of the pooling operation; induces group sparsity
   local mask_mag = nil -- L1 loss applied to the pooling reconstruction mask, consisting of P*s, where s are the pooling units
   
   pooling_sl_mag = 0.5e-2 --0.9e-2 --0.5e-2 --0.15e-2 --0.25e-2 --2e-2 --5e-2 -- keep in mind that there are four times as many mask outputs as pooling outputs in the first layer -- also remember that the columns of decoding_pooling_dictionary are normalized to be the square root of the pooling factor.  However, before training, this just ensures that all decoding projections have a magnitude of one
   mask_mag = 0.3e-2 --0.2e-2 --0.3e-2 --0.4e-2 --0.5e-2 --0 --0.75e-2 --0.5e-2 --0.75e-2 --8e-2 --4e-2 --2.5e-2 --1e-1 --5e-2
   
   --pooling_sl_mag = 0.1e-2
   --mask_mag = 0.35e-2 --0.375e-2 --0.35e-2 --0.325e-2
   
   --pooling_sl_mag = 1.7e-2
   --mask_mag = 0
   
   if not(recpool_config_prefs.disable_pooling) and not(recpool_config_prefs.disable_pooling_losses) then
      --sl_mag = 0 -- this *SHOULD* be scaled by L1_scaling
      sl_mag = 0.33e-2 --now scaled by L1_scaling = 3    was: 1e-2 -- attempt to duplicate good run on 10/11
      --sl_mag = 0.025e-2 -- used in addition to group sparsity
   else
      sl_mag = 3e-2 * sparsity_scaling -- now scaled by L1_scaling = 3 was: 9e-2 --3e-2
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
   if recpool_config_prefs.disable_pooling_losses then
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
   if command_line_params.selected_dataset == 'cifar' then
      rec_mag = 5
   end
   if command_line_params.num_layers == 1 then
      if command_line_params.fe_layer_size == 200 then
	 L1_scaling = 3 --4 --2.5 -- with square root L2 position loss
	 --L1_scaling = 6 -- with classification loss added in
	 --L1_scaling = 3 --2.5 --1.5 -- straight L2 position, sqrt-sum-of-squares pooling
	 --L1_scaling = 2 --1.25 --6 --1 -- straight L2 position, cube-root-sum-of-squares pooling
	 if command_line_params.p_layer_size == 200 then
	    L1_scaling = 3 * 3/4
	 end
      else
	 -- SHOULD SCALE INITIAL SPARSITY RATHER THAN L1 SCALING WHEN CHANGING THE NUMBER OF UNITS
	 L1_scaling = 3 --3.1 --2 --3/math.sqrt(2) -- for use with 400 FE units
	 mask_mag = mask_mag * math.sqrt(200/50) / math.sqrt(command_line_params.fe_layer_size/command_line_params.p_layer_size)
      end
   elseif command_line_params.num_layers == 2 then
      -- TRY ONLY ADJUSTING THE LAYER 2 SCALING WHEN ADDING A SECOND LAYER!!!
      L1_scaling = 2.25 --1 --0.25 
   else
      error('L1_scaling not specified for command_line_params.num_layers')
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
   if false and (command_line_params.num_layers == 1) then
      print(lambdas)
      layer_size = {data:dataSize(), 200, 50, 10}
      layered_lambdas = {lambdas}
      local this_layer_lagrange_multiplier_targets = {}
      this_layer_lagrange_multiplier_targets.feature_extraction_target = lagrange_multiplier_targets_1.feature_extraction_target * layer_size[2]
      this_layer_lagrange_multiplier_targets.pooling_target = lagrange_multiplier_targets_1.pooling_target * layer_size[3]
      this_layer_lagrange_multiplier_targets.mask_target = lagrange_multiplier_targets_1.mask_target * layer_size[2]
      layered_lagrange_multiplier_targets = {this_lagrange_multiplier_targets}
      layered_lagrange_multiplier_learning_rate_scaling_factors = {lagrange_multiplier_learning_rate_scaling_factors_1}
   else
      print(lambdas_1, lambdas_2)
      layer_size = {data:dataSize()}
      layered_lambdas = {}
      layered_lagrange_multiplier_targets = {}
      layered_lagrange_multiplier_learning_rate_scaling_factors = {}
      for i = 1,command_line_params.num_layers do
	 if i == 1 then
	    table.insert(layer_size, command_line_params.fe_layer_size) --200)
	    table.insert(layer_size, command_line_params.p_layer_size)
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
   
   return layer_size, layered_lambdas, layered_lagrange_multiplier_targets, layered_lagrange_multiplier_learning_rate_scaling_factors
end