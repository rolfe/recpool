local ParameterizedL1Cost, parent = torch.class('nn.ParameterizedL1Cost', 'nn.Module')

function ParameterizedL1Cost:__init(layer_size, initial_lambda, desired_criterion_value, learning_rate_scaling_factor, running_jacobian_test)
   self.criterion = nn.L1Cost()
   self.scaled_input = torch.Tensor(layer_size)
   self.gradInput = torch.Tensor(layer_size)
   self.running_jacobian_test = running_jacobian_test or false

   if self.running_jacobian_test == 'full test' then -- output a tensor of one element if running Jacobian test; otherwise, just output a number
      self.output = torch.Tensor(1)
   else
      self.output = 0
   end
   -- only expose the scaling factor lambda as a weight if we want it to be trained; otherwise, it is subject to manipulations like weight decay by optim modules (like sgd) that violate abstraction barriers  
   self.weight = torch.Tensor(layer_size):fill(initial_lambda or 1)
   self.gradWeight = torch.Tensor(layer_size)

   self.desired_criterion_value = desired_criterion_value
   self.learning_rate_scaling_factor = learning_rate_scaling_factor or 1
end

--[[ --If we're using an optimization function other than optim.sgd, this will keep the lagrange multipliers from being updated along with the other parameters, since we want the lagrange multipliers to change very slowly
   function ParameterizedL1Cost:parameters()
   end
--]]


-- \sum_i w_i * (|x_i| - c) = \sum_i (x_i - c) if x_i is non-negative
function ParameterizedL1Cost:updateOutput(input) 
   self.scaled_input:abs(input)
   if self.running_jacobian_test then -- only output the scaled difference between the input and the desired L1 norm if we're running a Jacobian test; otherwise, output the scaled L1 norm without reference to the desired L1 norm
      self.scaled_input:add(-1*self.desired_criterion_value) 
   end

   self.scaled_input:cmul(self.scaled_input, self.weight)
   if self.running_jacobian_test == 'full test' then
      self.output[1] = torch.sum(self.scaled_input)
   else
      self.output = torch.sum(self.scaled_input)
   end

   --self.output = self.criterion:updateOutput(self.scaled_input) -- scaling before passing to the criterion ONLY works for the L1Cost; we use it simply because it is more efficient

   return self.output
end
    
function ParameterizedL1Cost:updateGradInput(input, gradOutput) -- we leave out the standard gradOutput argument here to make it clear that ParameterizedL1Cost's gradInput is sui generis
   local current_grad_output = 1
   if gradOutput then
      print('using gradOutput in ParameterizedL1Cost')
      current_grad_output = gradOutput[1]
   end


   self.gradInput:cmul(self.criterion:updateGradInput(input), self.weight)
   if gradOutput then -- avoid the cost of the mul if we don't need it
      self.gradInput:mul(current_grad_output)
   end

  return self.gradInput
end 


function ParameterizedL1Cost:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   local current_grad_output = 1
   if gradOutput then
      current_grad_output = gradOutput[1]
   end

   -- add -1*learning_rate_scaling_factor*(self.criterion_output - desired_criterion_value)
   self.scaled_input:abs(input)
   self.gradWeight:add(-1 * scale * current_grad_output * self.learning_rate_scaling_factor, self.scaled_input) -- minus, since we want to maximize the error with respect to the lagrange multiplier - use learning_rate_scaling_factor = -1 for testing
   self.gradWeight:add(scale * current_grad_output * self.learning_rate_scaling_factor * self.desired_criterion_value) 
end
