-- Wrap a criterion, for which only a single input comes from the network, into a module.  The criterion can also take a fixed target, specified by setTarget.  DEBUG_MODE, presently disable, potentially outputs diagnostic messages during each operation.

local L1CriterionModule, parent = torch.class('nn.L1CriterionModule', 'nn.Module')

function L1CriterionModule:__init(criterion, initial_lambda, desired_criterion_value, learning_rate_scaling_factor, exempt_max)
   self.criterion = criterion
   self.criterion_output = 0
   --self.gradInput = criterion.gradInput
   self.gradInput = torch.Tensor()
   
   -- only expose the scaling factor lambda as a weight if we want it to be trained; otherwise, it is subject to manipulations like weight decay by optim modules (like sgd) that violate abstraction barriers  
   if desired_criterion_value then
      self.weight = torch.Tensor(1)
      self.gradWeight = torch.Tensor(1)

      self.weight[1] = initial_lambda or 1
   end
   
   self.lambda = initial_lambda or 1
   self.lambda_scaling = 1 -- TEST CODE FOR AUTO-CATEGORICAL UNITS
   self.desired_criterion_value = desired_criterion_value
   self.learning_rate_scaling_factor = learning_rate_scaling_factor or 1
   self.exempt_max = exempt_max
end

-- this is required for ClassNLLCriterion
function L1CriterionModule:setTarget(target) -- the target need not be set if the criterion's updateOutput and updateGradOutput only take the single argument input; target is then nil by default, and ignored
   self.target = target
end

function L1CriterionModule:reset_lambda(new_lambda)
   self.lambda = new_lambda
   if self.weight then
      self.weight[1] = self.lambda
   end
end

function L1CriterionModule:updateOutput(input) 
   if self.weight then
      self.lambda = self.weight[1]
   end
   
   -- find max; use L1Cost.c; add back in max value and save index for gradient calculation
   self.criterion_output = self.criterion:updateOutput(input, self.target) -- self.target is ignored by L1Cost, but used by ClassNLLCriterion

   -- use L1 reweighted in proportion to the magnitude of each unit, so that units near 0 are subject to a full L1 norm, whereas the most active units are subject to a greatly reduced L1 norm: \sum_i (1 - |z_i|/(\sum_i |z_i|)) * |z_i| = \sum_i |z_i| - (\sum_i |z_i|^2)/(\sum_i |z_i|)
   if self.exempt_max and (self.exempt_max < 1) then 
      self.L1_norm_vec = self.L1_norm_vec or torch.Tensor()
      self.L2_norm_vec = self.L2_norm_vec or torch.Tensor()
      self.abs_input = self.abs_input or torch.Tensor()
      self.abs_input_squared = self.abs_input_squared or torch.Tensor()
      --[[ RESIZING IS DONE AUTOMATICALLY!!!
      local norm_vec_size = input:size().new(input:size():size() - 1) -- create a new Storage with dimensionality one less than the Storage holding the size of input
      for i = 1,input:size():size() - 1 do -- create a size Storage that matches the size Storage of input up to the last dimension
	 norm_vec_size[i] = input:size()[i]
      end
      self.L1_norm_vec:resize(norm_vec_size)
      self.L2_norm_vec:resize(norm_vec_size)
      self.output_vector_size = input:size(input:dim()) -- number of elements in the last dimension of the input, which indexes over hidden units rather than batches
      --]]

      self.abs_input:resizeAs(input):copy(input):abs()
      self.abs_input_squared:resizeAs(input):copy(self.abs_input):pow(2)
      torch.sum(self.L1_norm_vec, self.abs_input, input:dim()) -- resize automatically; keep in mind that the dimension over which the sum is performed is still present, but has extent of 1
      torch.sum(self.L2_norm_vec, self.abs_input_squared, input:dim())
      if input:dim() == 2 then
	 self.L1_norm_vec = self.L1_norm_vec:select(input:dim(),1) -- eliminate the vestigial dimension -- not necessary if we sum
	 self.L2_norm_vec = self.L2_norm_vec:select(input:dim(),1)
      end
      self.L2_norm_vec:cdiv(self.L1_norm_vec) --:mul(input:size(input:dim())) -- \sum_i |z_i|^2 / (\sum_j |z_j|)
      --print(self.L1_norm_vec, self.L2_norm_vec)

      ---[[
      if math.abs(self.criterion_output - torch.sum(self.L1_norm_vec)) > 1e-5 then
	 error('WARNING!!! ' .. self.criterion_output .. ' ~= ' .. torch.sum(self.L1_norm_vec))
      end
      --]]
      self.criterion_output = torch.sum(self.L1_norm_vec) - torch.sum(self.L2_norm_vec) -- a scalar is returned when torch.sum operates over all dimensions

      -- prepare for updateGradInput
      self.L2_norm_vec:cdiv(self.L1_norm_vec) -- compute n * (\sum_i |z_i|^2) / (\sum_i |z_i|)^2, since we've already put n * (\sum_i |z_i|^2) / (\sum_i |z_i|) in L2_norm_vec

      --[[
      local max_vals, max_indices = input:max(input:size(input:dim()))
      local min_vals, min_indices = input:min(input:size(input:dim()))
      if max_vals:dim() == 2 then
	 max_vals = max_vals:select(2,1)
	 min_vals = min_vals:select(2,1)
      end
      self.extreme_elements = self.extreme_elements or torch.Tensor()
      self.extreme_elements:resize(max_vals:size(1), 2)
      self.extreme_elements:select(2,1):copy(max_vals)
      self.extreme_elements:select(2,2):copy(min_vals)
      local abs_max_vals, abs_max_signs = self.extreme_elements:abs():max(2)
      self.criterion_output = criterion_output - self.abs_max_vals:sum()
      --]]

      -- use a^n * \sum_i (1/a - |z_i|/(\sum_j |z_j|))^n * |z_i| = \sum_i (1 - a * |z_i| / (\sum_j |z_j|))^n * |z_i|; n = exempt_max; a = internal_factor
   elseif self.exempt_max and (self.exempt_max >= 1)  then 
      self.L1_norm_vec = self.L1_norm_vec or torch.Tensor() -- \sum_i |z_i| for each batch
      self.L2_norm_vec = self.L2_norm_vec or torch.Tensor() -- (\sum_i |z_i|^2) for each batch
      self.abs_input = self.abs_input or torch.Tensor()
      self.abs_input_squared = self.abs_input_squared or torch.Tensor()

      self.abs_input:resizeAs(input):copy(input):abs() -- make |z_i|
      self.abs_input_squared:resizeAs(input):copy(self.abs_input):pow(2) -- make |z_i|^2
      -- L1_norm_vec: vector_batch \sum_i |z_i| 
      torch.sum(self.L1_norm_vec, self.abs_input, input:dim()) -- resize automatically; keep in mind that the dimension over which the sum is performed is still present, but has extent of 1
      if input:dim() == 2 then
	 self.L1_norm_vec = self.L1_norm_vec:select(input:dim(),1) -- eliminate the vestigial dimension -- not necessary if we sum
      end
      
      -- scaling term: vector_i of (1 - |z_i| / (\sum_j |z_j|))
      local internal_factor = 4
      self.lambda_scaling = internal_factor^self.exempt_max
      local scaling_term = torch.cmul(self.abs_input, torch.ger(torch.pow(self.L1_norm_vec, -1), torch.ones(input:size(input:dim())))):mul(-1):add(1/internal_factor) -- (1/a - |z_i| / (\sum_j |z_j|))
      local thresh = -math.pow(0.4, 1/self.exempt_max) / internal_factor -- -0.1
      scaling_term:maxN(thresh)
      self.scaling_term_pow_n = torch.pow(scaling_term, self.exempt_max) -- (1 - |z_i| / (\sum_j |z_j|))^n
      self.scaling_term_pow_n_minus_1 = torch.pow(scaling_term, self.exempt_max-1) -- (1 - |z_i| / (\sum_j |z_j|))^(n-1)
      self.scaling_term_pow_n_minus_1:zeroLtN2(scaling_term, thresh)

      self.criterion_output = torch.sum(torch.cmul(self.scaling_term_pow_n, self.abs_input)) -- sum over both i for each batch, and over batches

      -- L2_norm_vec: \sum_i [1 - |z_i| / (\sum_j |z_j|)]^(n-1) * |z_i|^2
      torch.sum(self.L2_norm_vec, torch.cmul(self.abs_input_squared, self.scaling_term_pow_n_minus_1), input:dim()) 
      if input:dim() == 2 then
	 self.L2_norm_vec = self.L2_norm_vec:select(input:dim(),1)
      end
      -- L2_norm_vec: \sum_i n*[1 - |z_i| / (\sum_j |z_j|)]^(n-1) * [|z_i|^2 / (\sum_j |z_j|)^2]
      self.L2_norm_vec:cdiv(torch.pow(self.L1_norm_vec,2)):mul(self.exempt_max)
      
      --self.criterion_output = torch.sum(self.L1_norm_vec) - torch.sum(self.L2_norm_vec) -- a scalar is returned when torch.sum operates over all dimensions
   end

   self.output = self.lambda_scaling * self.lambda * self.criterion_output
   --print(self.output)
   --self.output = self.criterion_output

   return self.output
end
    
function L1CriterionModule:updateGradInput(input) -- we leave out the standard gradOutput argument here to make it clear that L1CriterionModule's gradInput is sui generis
   if self.weight then
      self.lambda = self.weight[1]
   end
   
   local criterion_grad_input = self.criterion:updateGradInput(input, self.target)
   self.gradInput:resizeAs(criterion_grad_input)
   self.gradInput:copy(criterion_grad_input)
   --self.gradInput:mul(criterion_grad_input, self.lambda)

   -- need to construct the vector (1 - |z_i| / \sum_j |z_j|)

   if self.exempt_max and (false or (self.exempt_max < 1)) then -- don't apply the L1 loss to the largest element
      self.second_term = self.second_term or torch.Tensor()
      self.third_term = self.third_term or torch.Tensor()
      self.second_term = torch.ger(torch.pow(self.L1_norm_vec, -1), torch.ones(input:size(input:dim())))
      self.third_term = torch.ger(self.L2_norm_vec, torch.ones(input:size(input:dim()))) -- L2_norm_vec is already (\sum_i |z_i|^2) / (\sum_i |z_i|)^2
      self.gradInput:cmul(torch.add(torch.ones(input:size()), self.third_term)) -- sign(z_j) + sign(z_j) * (\sum_i |z_i|^2) / (\sum_i |z_i|)^2
      self.gradInput:add(-2, torch.cmul(input, self.second_term)) -- input:size(input:dim())
   elseif self.exempt_max and (self.exempt_max >= 1) then
      -- L2_norm_vec is already (\sum_i |z_i|^2) / (\sum_i |z_i|)^2
      local first_and_second_term = torch.add(torch.ger(self.L2_norm_vec, torch.ones(input:size(input:dim()))):cmul(self.gradInput), -1,
					      torch.ger(torch.pow(self.L1_norm_vec, -1), torch.ones(input:size(input:dim()))):cmul(self.scaling_term_pow_n_minus_1):cmul(input):mul(self.exempt_max))
      self.gradInput:cmul(self.scaling_term_pow_n) -- third term: (1 - |z_j|/(\sum_i |z_i|))^n * sign(z_j)
      self.gradInput:add(first_and_second_term)
   end

   self.gradInput:mul(self.lambda_scaling * self.lambda)

   --self.gradInput = criterion_grad_input

  return self.gradInput
end 


function L1CriterionModule:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   
   -- if the criterion is applied to minibatches, then the constraint enforced by the lagrange multiplier is similarly on entire minibatches, rather than on each element separately.  At the very least, this might require a modification of the desired_criterion_value
   if self.weight then
      self.gradWeight[1] = self.gradWeight[1] - scale*learning_rate_scaling_factor*(self.criterion_output - desired_criterion_value) -- minus, since we want to maximize the error with respect to the lagrange multiplier
   end
end
