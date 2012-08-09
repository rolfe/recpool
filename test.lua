require 'torch'
require 'nn'

function test_factored_sparse_coder_main_chain(fsc_net)

   local cmul_code_size = fsc_net.cmul_code_size
   local L1_code_size = fsc_net.L1_code_size

   local mytester = torch.Tester()
   local jac
   
   local precision = 1e-5
   local expprecision = 1e-4
   
   local fsc_basic_test = {}
   

   function fsc_basic_test.InputCMul()
      local in_dimension = math.random(10,20)
      local full_in_dimension = 2 * in_dimension
      local input = torch.Tensor(full_in_dimension):zero()
      local module = nn.InputCMul()
      
      local err = jac.testJacobian(module,input)
      mytester:assertlt(err,precision, 'error on state ')
   end
   
   local function generic_module_test(module, input)
      local err = jac.testJacobian(module,input)
      mytester:assertlt(err,precision, 'error on state ')
      
      local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
      mytester:assertlt(err,precision, 'error on weight ')
      
      local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
      mytester:assertlt(err,precision, 'error on bias ')
      
      local err = jac.testJacobianUpdateParameters(module, input, module.weight)
      mytester:assertlt(err,precision, 'error on weight [direct update] ')
      
      local err = jac.testJacobianUpdateParameters(module, input, module.bias)
      mytester:assertlt(err,precision, 'error on bias [direct update] ')
      
      for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
	 mytester:assertlt(err, precision, string.format(
			      'error on weight [%s]', t))
      end
      
      for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
	 mytester:assertlt(err, precision, string.format(
			      'error on bias [%s]', t))
      end
   end

   function fsc_basic_test.module_test_cmul_dictionary()
      local input = torch.Tensor(cmul_code_size):zero()
      generic_module_test(fsc_net.cmul_dictionary, input)
   end

   function fsc_basic_test.module_test_L1_dictionary()
      if fsc_net.use_L1_dictionary then
	 local input = torch.Tensor(L1_code_size):zero()
	 generic_module_test(fsc_net.L1_dictionary, input)
      end
   end
   
   function fsc_basic_test.processing_chain_test()
      local input = torch.Tensor(L1_code_size + cmul_code_size):zero()
      
      local err = jac.testJacobian(fsc_net.processing_chain,input)
      mytester:assertlt(err,precision, 'error on processing chain state ')
      

      local err = jac.testJacobianParameters(fsc_net.processing_chain, input, fsc_net.L1_dictionary.weight, fsc_net.L1_dictionary.gradWeight)
      mytester:assertlt(err,precision, 'error on L1_dictionary weight ')

      local err = jac.testJacobianUpdateParameters(fsc_net.processing_chain, input, fsc_net.L1_dictionary.weight, {fsc_net.cmul_dictionary.weight, fsc_net.cmul_dictionary.bias, fsc_net.L1_dictionary.bias})
      mytester:assertlt(err,precision, 'error on L1_dictionary weight [full processing chain, direct update] ')


      local err = jac.testJacobianParameters(fsc_net.processing_chain, input, fsc_net.L1_dictionary.bias, fsc_net.L1_dictionary.gradBias)
      mytester:assertlt(err,precision, 'error on L1_dictionary bias ')

      local err = jac.testJacobianUpdateParameters(fsc_net.processing_chain, input, fsc_net.L1_dictionary.bias, {fsc_net.cmul_dictionary.weight, fsc_net.cmul_dictionary.bias, fsc_net.L1_dictionary.weight})
      mytester:assertlt(err,precision, 'error on L1_dictionary bias [full processing chain, direct update] ')


      local err = jac.testJacobianParameters(fsc_net.processing_chain, input, fsc_net.cmul_dictionary.weight, fsc_net.cmul_dictionary.gradWeight)
      mytester:assertlt(err,precision, 'error on cmul_dictionary weight ')

      local err = jac.testJacobianUpdateParameters(fsc_net.processing_chain, input, fsc_net.cmul_dictionary.weight, {fsc_net.cmul_dictionary.bias, fsc_net.L1_dictionary.weight, fsc_net.L1_dictionary.bias})
      mytester:assertlt(err,precision, 'error on cmul_dictionary weight [full processing chain, direct update] ')


      local err = jac.testJacobianParameters(fsc_net.processing_chain, input, fsc_net.cmul_dictionary.bias, fsc_net.cmul_dictionary.gradBias)
      mytester:assertlt(err,precision, 'error on cmul_dictionary bias ')

      local err = jac.testJacobianUpdateParameters(fsc_net.processing_chain, input, fsc_net.cmul_dictionary.bias, {fsc_net.cmul_dictionary.weight, fsc_net.L1_dictionary.weight, fsc_net.L1_dictionary.bias})
      mytester:assertlt(err,precision, 'error on cmul_dictionary bias [full processing chain, direct update] ')
   end

   
   mytester:add(fsc_basic_test) -- build up a table of functions fsc_basic_test with all of the function definitions above

   jac = nn.Jacobian
   mytester:run()
   
end




function test_factored_sparse_coder_ista(fsc_psd, input, target) -- input is the bottom-up pixel represention; target is the top-down class

   local fsc_net = fsc_psd.decoder
   local cmul_code_size = fsc_net.cmul_code_size
   local L1_code_size = fsc_net.L1_code_size

   local mytester = torch.Tester()
   local jac
   
   local precision = 1e-5
   local expprecision = 1e-4
   
   local fsc_ista_test = {}

   function fsc_ista_test.ista()
      -- pick random target
      -- run updateOutput
      -- check that each unit is either equal to zero or has a very small gradient; ideally, we would check the subgradient, but this might require adding subgradients from the pooling (L1) unit and cmul mask regularizers, and is not performed automatically by our functions.  The current test is less conservative, and assumes that the subgradient of the L1 regularizer is sufficient to balance out the gradient of the smooth cost function if the unit is equal to zero

      local fista_params_copy = {}
      
      -- make a copy of all fista params before running the test, since we're going to need to use much finer tolerances on convergence
      for k,v in pairs(fsc_net.params) do
	 fista_params_copy[k] = v
      end
      
      fsc_net.params.errthres = 1e-12
      fsc_net.params.maxiter = 10000


      fsc_net.wake_sleep_stage = 'wake'
      local err,h = fsc_psd:updateOutput(input, target) -- all gradients should be calculated in the process of running fista
      local cost, grad_concat_code_smooth = fsc_net.smooth_cost(fsc_net.concat_code, 'dx')
      local L1_cost, grad_concat_code_nonsmooth = fsc_net.nonsmooth_cost(fsc_net.concat_code, 'dx')
      local grad_concat_code_total = grad_concat_code_smooth:clone()
      
      fsc_net.extract_L1_from_concat(grad_concat_code_total):add(grad_concat_code_nonsmooth)
      fsc_net.extract_L1_from_concat(grad_concat_code_total)[torch.eq(fsc_net.extract_L1_from_concat(fsc_net.concat_code), 0)] = 0
      
      -- reset the fista params to their normal values
      for k,v in pairs(fsc_net.params) do
	 fsc_net.params[k] = fista_params_copy[k]
      end
      

      print(torch.cat(fsc_net.concat_code, grad_concat_code_total, 2):t())
      mytester:assertlt(grad_concat_code_total:abs():max(), 1e-2, 'error on ista ')
   end

   local function sign(x)
      if x > 0 then
	 return 1
      elseif x < 0 then
	 return -1
      else
	 return 0
      end
   end
      

   -- calculate the approximate gradient with respect to element chosen_index of param by perturbing up and down by perturbation_size
   local function forward_param(max_error, param, param_grad, chosen_index, perturbation_size, fsc_net)
      local flattened_param = param.new(param):resize(param:nElement())
      local flattened_param_grad = param_grad.new(param_grad):resize(param_grad:nElement())
      
      local current_code = fsc_net.concat_code
      local this_smooth_cost, this_nonsmooth_cost = 0,0
      local this_down_cost, this_up_cost, this_cost_derivative = 0,0,0
      local this_error = 0

      flattened_param[chosen_index] = flattened_param[chosen_index] - perturbation_size
      fsc_net:updateOutput(input, current_code, target)
      this_smooth_cost = fsc_net.smooth_cost(fsc_net.concat_code)
      this_nonsmooth_cost = fsc_net.nonsmooth_cost(fsc_net.concat_code)
      this_down_cost = this_smooth_cost + this_nonsmooth_cost
      
      flattened_param[chosen_index] = flattened_param[chosen_index] + 2*perturbation_size
      this_smooth_cost = fsc_net.smooth_cost(fsc_net.concat_code)
      this_nonsmooth_cost = fsc_net.nonsmooth_cost(fsc_net.concat_code)
      this_up_cost = this_smooth_cost + this_nonsmooth_cost
      
      flattened_param[chosen_index] = flattened_param[chosen_index] - perturbation_size
      this_cost_derivative = (this_up_cost - this_down_cost) / (2*perturbation_size)

      print('Compare: ', flattened_param_grad[chosen_index], this_cost_derivative, flattened_param_grad[chosen_index] / this_cost_derivative)
      if (math.abs(flattened_param_grad[chosen_index]) > 1e-10) and (math.abs(this_cost_derivative) > 1e-10) then -- ignore the difference between derivatives if they are both very small, since this makes the exact calculation unreliable, and even large percentage differences are basically irrelevant
	 if sign(flattened_param_grad[chosen_index]) ~= sign(this_cost_derivative) then
	    print('Error: mismatched sign')
	    max_error = math.max(max_error, 10)
	 else
	    this_error = math.abs(math.log(math.abs(flattened_param_grad[chosen_index])) - math.log(math.abs(this_cost_derivative)))
	    print('Error: ', this_error)
	    max_error = math.max(max_error, this_error)
	 end
      end
      
      return max_error
   end
   
   function fsc_ista_test.implicit_grad()
      -- pick random target
      -- run updateOutput
      -- calculate and save the gradient of the energy with respect to the parameters
      -- for a randomly selected subset of the parameters, perturb the parameters, reminimize with updateOutput, and compare the change in the energy (self.smooth_cost() + self.nonsmooth_cost()) with the parameter gradient calcualted previously
      -- for efficiency, make sure that most of the chosen parameters correspond to active units, but choose some that connect to inactive units
      -- ideally, increase the number of ista iterations iterations when perturbing to calculate the gradient.  Better yet, do one long convergence initially, and don't reset the units between the perturbations, to ensure that little additional work needs to be done.

      local fista_params_copy = {}

      -- make a copy of all fista params before running the test, since we're going to need to use much finer tolerances on convergence
      for k,v in pairs(fsc_net.params) do
	 fista_params_copy[k] = v
      end

      fsc_net.params.errthres = 1e-15
      fsc_net.params.maxiter = 50000

      fsc_net.wake_sleep_stage = 'wake'
      local err,h = fsc_psd:updateOutput(input, target) -- all gradients should be calculated in the process of running fista
      print('ran for ', #h, ' iters')
      fsc_net:zeroGradParameters()
      fsc_net:accGradParameters()
      local cmul_dictionary_grad = fsc_net.cmul_dictionary.gradWeight:clone()
      local L1_dictionary_grad = fsc_net.L1_dictionary.gradWeight:clone() -- this does *not* include the non-smooth gradient due to shrinking the cmul-mask, which only falls on the L1_dictionary
      local L1_dictionary_grad_after_cmul

      -- the nonsmooth gradient directly on the parameters is not calculated by accGradParameters, since it must be implemented by a shrinkage function
      if fsc_net.shrink_cmul_mask then 
	 local L1_code_from_concat = fsc_net.extract_L1_from_concat(fsc_net.concat_code)
	 
	 L1_dictionary_shrink_val_L1_code_part = torch.abs(L1_code_from_concat) -- |a*b| = |a|*|b|

	 if fsc_net.use_lagrange_multiplier_cmul_mask then 
	    current_L1_dictionary_shrink_val = torch.ger(fsc_net.lagrange_multiplier_cmul_mask, L1_dictionary_shrink_val_L1_code_part)
	    local L1_dictionary_weight_sign = fsc_net.L1_dictionary.weight:clone()
	    L1_dictionary_weight_sign:sign()
	    L1_dictionary_weight_sign[torch.eq(L1_dictionary_weight_sign,0)] = 1 -- the subgradients required to match the empirical gradient is probably 1 (with non-negative units) if the unit changes during the calculation of the empirical gradient.  However, it's unclear why this doesn't disrupt 

	    L1_dictionary_grad:add(current_L1_dictionary_shrink_val:cmul(L1_dictionary_weight_sign))
	 else
	    error('THIS ISNT FINISHED!')
	    -- we only shrink in the wake stage, so scale the shrinkage down by the difference between the wake and sleep stage learning rates
	    L1_dictionary_shrink_val_L1_code_part:mul(self.L1_lambda_cmul)
	    local L1_units_abs_duplicate_rows = torch.Tensor(L1_dictionary_shrink_val_L1_code_part:storage(), L1_dictionary_shrink_val_L1_code_part:storageOffset(), self.L1_dictionary.weight:size(), torch.LongStorage{0,1}) -- this doesn't allocate new main storage, so it should be relatively efficient, even though a new Tensor view is created on each iteration
	    self.L1_dictionary_shrinkage(self.L1_dictionary.weight, L1_units_abs_duplicate_rows) -- NOTE that L1_dictionary_shrink_val_L1_code_part cannot be reused after this, since it is multiplied by -1 when shrinking
	 end 
      end -- shrink_cmul_mask






      local current_code = fsc_net.concat_code
      fsc_net.params.errthres = 1e-12
      fsc_net.params.maxiter = 5000


      
      local chosen_index = 0
      local perturbation_size = 1e-5
      -- 1D view of input
      local max_error_cmul = 0;
      local max_error_L1 = 0;


      local param = fsc_net.cmul_dictionary.weight
      local param_grad = cmul_dictionary_grad
      
      for i = 1,100 do
	 -- select a random parameter; increment it; calculate the cost; decrement it; calculate the cost; reset it
	 chosen_index = math.random(param:nElement())

	 max_error_cmul = forward_param(max_error_cmul, param, param_grad, chosen_index, perturbation_size, fsc_net)
      end


      print('L1_dictionary gradient')
      param = fsc_net.L1_dictionary.weight
      param_grad = L1_dictionary_grad
      
      for i = 1,100 do
	 -- select a random parameter; increment it; calculate the cost; decrement it; calculate the cost; reset it
	 chosen_index = math.random(param:nElement())

	 max_error_L1 = forward_param(max_error_L1, param, param_grad, chosen_index, perturbation_size, fsc_net)
      end



      -- reset the fista params to their normal values
      for k,v in pairs(fsc_net.params) do
	 fsc_net.params[k] = fista_params_copy[k]
      end

      if fsc_net.params.maxiter > 100 then
	 error('did not successfully reset maxiter')
      end

      mytester:assertlt(max_error_cmul, 3e-2, 'error on cmul_dictionary gradient ')
      mytester:assertlt(max_error_L1, 3e-2, 'error on L1_dictionary gradient ')
   end

   
   mytester:add(fsc_ista_test) -- build up a table of functions fsc_ista_test with all of the function definitions above
   
   jac = nn.Jacobian
   mytester:run()
   
end

