-- initialize the factored_sparse_coder cost functions, used for FISTA optimization

function init_fsc_cost_functions(self) -- We're basically using this as a single-use factory.  It would make more sense to define the cost functions with : rather than . and always pass self in explicitly, but this is not compatible with the optim package

   -- the smooth cost (f) passed to FISTA, which captures the reconstruction error, the top-level classification error, and potentially the distance between the encoder output and the code found by FISTA
   -- smooth_cost takes the internal code as input, performs the reconstruction of the "input" and top-level classification, and calculates costs (and possibly derivatives)
   self.smooth_cost = function(concat_code, mode)
      local grad_concat_code_smooth = nil
      local L2_code_from_concat = self.extract_L2_from_concat(concat_code)
      local L1_code_from_concat = self.extract_L1_from_concat(concat_code)
      
      if not(self.use_L2_code) then
	 L2_code_from_concat:fill(1) -- this effectively disables the L2_code, since each output of the L1 branch is then just multiplied by one
      end
      
      -- forward function evaluation
      local reconstruction = self.processing_chain:updateOutput(concat_code)
      local reconstruction_cost = self.input_reconstruction_cost:updateOutput(reconstruction, self.input) 
      local L2_cost = self.L2_lambda * 0.5 * self.concat_code_L2_cost:updateOutput(L2_code_from_concat, self.concat_constant_zeros_L2)
      if self.L1_L2_lambda and (self.L1_L2_lambda ~= 0) then 
	 L2_cost = L2_cost + self.L1_L2_lambda * 0.5 * self.concat_code_L1_L2_cost:updateOutput(L1_code_from_concat, self.concat_constant_zeros_L1_L2) 
      end
      
      local classification = nil
      local classification_cost = 0
      if self.use_top_level_classifier and (not(self.wake_sleep_stage == 'sleep') or self.sleep_stage_use_classifier) then -- WARNING: this is UNNECESSARILY INEFFICIENT!!!  Only need to update classification to check if output is correct
	 classification = self.top_level_classifier:updateOutput(concat_code)
	 -- don't update the classification cost in the test phase, since it's gradient is not used, and the two must be matched for the line search in the fista algorithm
	 if not(self.wake_sleep_stage == 'test') then 
	    -- minimize rather than maximize the likelihood of the correct target during sleep
	    classification_cost = self.top_level_classifier_scaling_factor * self.top_level_classification_cost:updateOutput(classification, self.target)
	    if self.wake_sleep_stage == 'sleep' then classification_cost = -1*classification_cost end
	 end
      end
      local cost = reconstruction_cost + L2_cost + classification_cost
      
      local reconstruction_grad_mag, L2_grad_mag, classification_grad_mag = 0,0,0
      if mode and mode:match('verbose') then
	 self.cost_output_counter = self.cost_output_counter + 1
      end
      
      
      -- derivative wrt code
      if mode and (mode:match('dx') or (mode:match('verbose') and self.cost_output_counter >= 250)) then
	 local grad_reconstruction = self.input_reconstruction_cost:updateGradInput(reconstruction, self.input)
	 grad_concat_code_smooth = self.processing_chain:updateGradInput(concat_code, grad_reconstruction)
	 --self.grad_concat_code_smooth:copy(grad_concat_code_smooth)
	 
	 if mode and mode:match('verbose') and self.cost_output_counter >= 250 then
	    reconstruction_grad_mag = grad_concat_code_smooth:norm()
	 end
	 
	 if not(self.use_L2_code) then
	    self.extract_L2_from_concat(grad_concat_code_smooth):fill(0)
	 else
	    -- THIS WAS INCORRECTLY SET TO A CONSTANT BEFORE 6/21, DUE TO THE ACCIDENTAL SUBSTITUTION OF A COMMA FOR A MULTIPLICATION!!!
	    local L2_grad = self.concat_code_L2_cost:updateGradInput(L2_code_from_concat, self.concat_constant_zeros_L2):mul(self.L2_lambda * 0.5)
	    self.extract_L2_from_concat(grad_concat_code_smooth):add(L2_grad)
	    
	    if mode and mode:match('verbose') and self.cost_output_counter >= 250 then
	       L2_grad_mag = L2_grad:norm()
	    end
	 end
	 
	 if self.L1_L2_lambda and (self.L1_L2_lambda ~= 0) then 
	    -- NOTE: it might be more parsimonious to do this with a nn.Narrow layer
	    local L1_L2_grad = self.concat_code_L1_L2_cost:updateGradInput(L1_code_from_concat, self.concat_constant_zeros_L1_L2):mul(self.L1_L2_lambda * 0.5)
	    self.extract_L1_from_concat(grad_concat_code_smooth):add(L1_L2_grad)
	 end
	 
	 if self.use_top_level_classifier and ((self.wake_sleep_stage == 'wake') or ((self.wake_sleep_stage == 'sleep') and self.sleep_stage_use_classifier)) then  
	    local grad_classification_cost = self.top_level_classification_cost:updateGradInput(classification, self.target):mul(self.top_level_classifier_scaling_factor)
	    if self.wake_sleep_stage == 'sleep' then grad_classification_cost:mul(-1) end -- minimize rather than maximize the likelihood of the correct target during sleep
	    local classification_grad = self.top_level_classifier:updateGradInput(concat_code, grad_classification_cost)
	    grad_concat_code_smooth:add(classification_grad)
	    if mode and mode:match('verbose') and self.cost_output_counter >= 250 then
	       classification_grad_mag = classification_grad:norm()
	    end
	 end
	 
      end -- dx mode
      
      if mode and mode:match('verbose') and (self.cost_output_counter >= 250) then
	 print('rec: ' .. reconstruction_cost .. ' ; clas: ' .. classification_cost .. ' ; L2 ' .. L2_cost .. ' ; rec grad: ' .. reconstruction_grad_mag .. ' ; clas grad ' .. classification_grad_mag .. ' ; L2 grad ' .. L2_grad_mag)
      end
      
      return cost, grad_concat_code_smooth
   end
   
   
   
   
   -- when shrinking the cmul mask, we need to calculate the gradient on the pooling (L1) units due to the L1 regularizer on the outputs of the L1_dictionary by projecting the lagrange multipliers (or a vector of ones, if we're not using lagrange multipliers) through the L1_dictionary matrix
   -- the nonsmooth gradient (i.e. the shrink magnitude) is needed in both nonsmooth_cost and minimize_nonsmooth, so just define it once
   -- output is returned in current_shrink_val
   function self.nonsmooth_shrinkage_cmul_factory()
      local L1_dictionary_abs_copy
      if self.use_L1_dictionary then
	 L1_dictionary_abs_copy = torch.Tensor()
      else
	 L1_dictionary_abs_copy = self.L1_dictionary_identity_matrix -- this results in unnecessarily inefficient operations, but it should only be used for debugging purposes
      end
      
      -- current_shrink_val is stored in the nonsmooth_cost_factory, so doesn't need to be duplicated here
      local function calculate_nonsmooth_shrinkage_cmul(current_shrink_val) 
	 -- Make sure not to alter the gradients in the processing chain, since this will affect the parameter updates; the weight update due to an L1 regularizer needs to be done via a shrink, rather than a standard gradient step
	 -- we need the sum of the absolute values of each column of the weight matrix, with each row scaled by the appropriate lagrange multiplier
	 if self.use_L1_dictionary then
	    -- WARNING: this is UNNECESSARILY INEFFICIENT if L1_dictionary is non-negative; in this case, neither the abs operation is redundant
	    L1_dictionary_abs_copy:resizeAs(self.L1_dictionary.weight)
	    L1_dictionary_abs_copy:abs(self.L1_dictionary.weight) 
	 end
	 current_shrink_val:resize(L1_dictionary_abs_copy:size(2)) -- this should be the same size as L1_code_from_concat, but there's no need to recreate it here
	 
	 if self.use_lagrange_multiplier_cmul_mask then
	    current_shrink_val:mv(L1_dictionary_abs_copy:t(), self.lagrange_multiplier_cmul_mask)
	 else	    
	    current_shrink_val:sum(L1_dictionary_abs_copy, 1) -- WARNING: this is slightly INEFFICIENT; it would probably be best to write a new c function that performs the abs-sum directly, rather than making a copy of the L1_dictionary in order to perform the absolute value computation
	    current_shrink_val:mul(self.L1_lambda_cmul) 
	 end
      end -- calculate_nonsmooth_shrinkage_cmul
      
      return calculate_nonsmooth_shrinkage_cmul
   end
   
   self.calculate_nonsmooth_shrinkage_cmul = self.nonsmooth_shrinkage_cmul_factory() -- instantiate the factory

      
   
   
   -- construct the non-smooth cost function (g) required by fista; this is the sparse regularizer.  We include the ability to calculate the gradient for completeness, but this is only used for unit testing and diagnostics
   function self.nonsmooth_cost_factory() -- create a continuation so it's easy to bind a persistent copy of current_shrink_val to the function
      local current_shrink_val = torch.Tensor()
      local L1_code_from_concat_sign = torch.Tensor()
      local code_abs_L1_units = torch.Tensor() -- used to accumulate the L1 norm
      local code_abs_cmul_mask = torch.Tensor() -- used to accumulate the L1 norm

      local function nonsmooth_cost(concat_code, mode)
	 --local L2_code_from_concat = self.extract_L2_from_concat(concat_code)
	 local L1_code_from_concat = self.extract_L1_from_concat(concat_code)
	 
	 local grad_concat_code_nonsmooth = nil
	 local L1_cost = 0
	 local L1_grad_mag = 0
	 if self.shrink_L1_units then 
	    if self.use_lagrange_multiplier_L1_units then
	       code_abs_L1_units:resizeAs(L1_code_from_concat)
	       code_abs_L1_units:abs(L1_code_from_concat)
	       L1_cost = L1_cost + self.lagrange_multiplier_L1_units:dot(code_abs_L1_units)
	    else
	       L1_cost = L1_cost + self.L1_lambda * self.concat_code_L1_cost:updateOutput(L1_code_from_concat) 
	    end
	 end
	 
	 if self.shrink_cmul_mask then
	    if self.use_lagrange_multiplier_cmul_mask then
	       -- This is *NOT* correct if any of the L1 units or L1 dictionary entries are negative
	       code_abs_cmul_mask:resizeAs(self.L1_processing_chain.output) 
	       code_abs_cmul_mask:abs(self.L1_processing_chain.output)
	       L1_cost = L1_cost + self.lagrange_multiplier_cmul_mask:dot(code_abs_cmul_mask)
	    else
	       L1_cost = L1_cost + self.L1_lambda_cmul * self.L1_processing_chain.output:norm(1) -- THIS IS NOT CORRECT; we should actually take the absolute value of the dictionary matrix and the L1_code_from_concat before multiplying.  However, this calculation does not affect either fista or the weight update, and is correct if everything is nonnegative, so we'll leave this for now
	    end
	 end
	 
	 
	 if mode and (mode:match('dx') or (mode:match('verbose') and self.cost_output_counter >= 250)) then
	    --print('Gradient of nonsmooth cost should never be evaluated')
	    if self.shrink_L1_units then
	       if self.use_lagrange_multiplier_L1_units then
		  grad_concat_code_nonsmooth = self.concat_code_L1_cost:updateGradInput(L1_code_from_concat):cmul(self.lagrange_multiplier_L1_units) 
	       else
		  grad_concat_code_nonsmooth = self.concat_code_L1_cost:updateGradInput(L1_code_from_concat):mul(self.L1_lambda) 
	       end
	    end
	    
	    if self.shrink_cmul_mask then
	       self.calculate_nonsmooth_shrinkage_cmul(current_shrink_val) 
	       
	       L1_code_from_concat_sign:resizeAs(L1_code_from_concat)
	       L1_code_from_concat_sign:sign(L1_code_from_concat)
	       current_shrink_val:cmul(L1_code_from_concat_sign) 
	       
	       if not(grad_concat_code_nonsmooth) then 
		  grad_concat_code_nonsmooth = current_shrink_val -- this is *only* safe so long as current_shrink_val is not used elsewhere for other computations!!!
	       else
		  grad_concat_code_nonsmooth:add(current_shrink_val)
	       end
	    end
	    
	    if mode and mode:match('verbose') and self.cost_output_counter >= 250 then
	       L1_grad_mag = grad_concat_code_nonsmooth:norm()
	       print('L1: ' .. L1_cost .. ' ; L1 grad: ' .. L1_grad_mag)
	       self.cost_output_counter = 0
	    end
	 end
	 
	 return L1_cost, grad_concat_code_nonsmooth
      end
      
      return nonsmooth_cost -- the local function, bound to the persistent local variable, is the output of the factory
   end
   
   self.nonsmooth_cost = self.nonsmooth_cost_factory() -- instantiate the factory

   


   -- Koray's shrinkage implementation in kex requires that all elements of a tensor be shrunk by the same amount.  shrinkage_factory produces a shrinkage function which can apply a different shrink magnitude to each element of a tensor, as is required if we have lagrange multipliers on the L1 units or are shrinking the cmul mask
   function self.shrinkage_factory(nonnegative_L1_units)
      local shrunk_indices = torch.ByteTensor() -- allocate this down here for clarity, so it's close to where it's used
      local shrink_sign = torch.Tensor() -- sign() returns a DoubleTensor
      local shrink_val_bounded = torch.Tensor()
      local unrepeated_shrink_val = torch.Tensor()
      if type(nonnegative_L1_units) == 'nil' then nonnegative_L1_units = true end -- use nonnegative units by default


      local function this_shrinkage(vec, shrink_val)
	 --self.shrink_le:resize(vec_size)
	 --self.shrink_ge:resize(vec_size)

	 if nonnegative_L1_units then
	    -- if any elements of shrink_val are < 0, it is essential that the corresponding elements of vec be set equal to zero -- WHY?!?
	    shrink_val_bounded:resizeAs(shrink_val)
	    shrink_val_bounded:copy(shrink_val)
	    shrink_val_bounded[torch.le(shrink_val, 0)] = 0 -- it should be safe to remove this to allow for negative lagrange multipliers
	    shrunk_indices = torch.le(vec, shrink_val_bounded)
	    vec:add(-1, shrink_val) -- don't worry about adding to negative values, since they will be set equal to zero by shrunk_indices
	 else
	    local vec_size = vec:size()
	    shrunk_indices:resize(vec_size)
	    shrink_sign:resize(vec_size)
	    unrepeated_shrink_val:set(shrink_val:storage())
	    
	    local shrink_le = torch.le(vec, shrink_val) -- WARNING: this is INEFFICIENT, since torch.le and torch.ge unnecessarily allocate new memory on every iteration
	    
	    unrepeated_shrink_val:mul(-1) -- if shrink_val has a stride of length 0 to repeat a constant row or column, make sure that the multiplied constant is only applied once per underlying entry

	    local shrink_ge = torch.ge(vec, shrink_val) -- WARNING: this is INEFFICIENT, since torch.le and torch.ge unnecessarily allocate new memory on every iteration
	    shrunk_indices:cmul(shrink_le, shrink_ge)
	    shrink_sign:sign(vec)

	    vec:addcmul(shrink_val, shrink_sign) -- shrink_val has already been multiplied by -1
	 end
	 
	 --shrunk_indices:cmul(torch.le(vec, shrink_val), torch.ge(vec, torch.mul(shrink_val, -1))) -- WARNING: this is INEFFICIENT, since torch.le and torch.ge unnecessarily allocate new memory on every iteration
	 --vec:addcmul(-1, shrink_val, torch.sign(vec)) -- WARNING: this is INEFFICIENT, since torch.sign unnecessarily allocates memory on every iteration
	 
	 vec[shrunk_indices] = 0
      end

      return this_shrinkage
   end
   


   
   -- construct a function that minimizes the code with respect to the non-smooth cost: argmin_x Q(x,y)
   function self.minimizer_factory() -- create a continuation so it's easy to bind a persistent copy of current_shrink_val to the function
      local current_shrink_val_L1 = torch.Tensor()
      local current_shrink_val_cmul_mask = torch.Tensor()

      local function minimize_nonsmooth(concat_code, L)
	 local L1_code_from_concat = self.extract_L1_from_concat(concat_code)
	 
	 if self.shrink_L1_units then
	    if self.use_lagrange_multiplier_L1_units then 
	       current_shrink_val_L1:resize(self.lagrange_multiplier_L1_units:size())
	       current_shrink_val_L1:div(self.lagrange_multiplier_L1_units, L) -- this will be multiplied by negative one in self.shrinkage, so it must be recomputed each time
	       self.L1_shrinkage(L1_code_from_concat, current_shrink_val_L1) 
	    else
	       L1_code_from_concat:shrinkage(self.L1_lambda/L) -- this uses Koray's optimized shrink which only accepts a single shrink value
	    end
	 end
	 
	 if self.shrink_cmul_mask then
	    self.calculate_nonsmooth_shrinkage_cmul(current_shrink_val_cmul_mask) 
	    current_shrink_val_cmul_mask:div(L) 
	    self.L1_shrinkage(L1_code_from_concat, current_shrink_val_cmul_mask) 
	 end
	 
	 local nonnegative_L2_units = true
	 if nonnegative_L2_units then
	    local L2_code_from_concat = self.extract_L2_from_concat(concat_code)
	    --L2_code_from_concat:shrinkage(self.L2_lambda/(2*L))
	    L2_code_from_concat[torch.lt(L2_code_from_concat,0)] = 0 -- WARNING: this is extremely INEFFICIENT, since torch.lt() allocates new memory on each call
	 end
      end
      
      return minimize_nonsmooth -- the local function, bound to the persistent local variable, is the output of the factory
   end

end