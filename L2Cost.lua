-- A module implementing the L2 distance between the (first two) elements of a table of tensors.  
-- The output is a single number (the desired L2 cost).  A single gradOutput, if provided, is used to scale gradInput.

local L2Cost, parent = torch.class('nn.L2Cost', 'nn.Module')

-- use_true_L2_cost specifies that we should use the square root of the sum of squares, rather than just the sum of squares
function L2Cost:__init(L2_lambda, num_inputs, use_true_L2_cost)
   parent.__init(self)
   if type(L2_lambda) ~= 'number' then
      error('L2_lambda input to L2Cost is of type ' .. type(L2_lambda))
   end
   self.L2_lambda = L2_lambda
   if (num_inputs ~= 1) and (num_inputs ~= 2) then
      error('L2Cost expects either 1 or 2 inputs, but was initialized with ' .. num_inputs)
   end
   self.num_inputs = num_inputs
   self.use_true_L2_cost = use_true_L2_cost

   -- state
   if self.num_inputs == 2 then -- one input is handled correctly by the default nn.Module __init()
      self.gradInput = {torch.Tensor(), torch.Tensor()}
      if use_true_L2_cost then
	 error('L2Cost is not configured to compute the true L2 cost with two inputs')
      end
   end
   --self.output = torch.Tensor(1)
   self.output = 1
end 
  
function L2Cost:updateOutput(input)
   --self.output[1]=input[1]:dist(input[2])
   --self.output:pow(2):mul(1/2); -- we want the sum of squares of the difference, not the square root of the sum of squares
   if self.num_inputs == 1 then
      if self.use_true_L2_cost then
	 self.output = self.L2_lambda * input:norm()
      else
	 self.output = self.L2_lambda * 0.5 * math.pow(input:norm(), 2)
      end
   elseif self.num_inputs == 2 then
      self.output = self.L2_lambda * 0.5 * math.pow(input[1]:dist(input[2]), 2)
   else
      error('Illegal self.num_inputs')
   end
   return self.output
end

function L2Cost:updateGradInput(input, gradOutput)
   if self.num_inputs == 1 then
      self.gradInput:resizeAs(input) 
      self.gradInput:copy(input)
      self.gradInput:mul(self.L2_lambda);
      if self.use_true_L2_cost then
	 self.gradInput:div(input:norm())
      end
      if gradOutput then
	 self.gradInput:mul(gradOutput[1]);
      end
   elseif self.num_inputs == 2 then
      self.gradInput[1]:resizeAs(input[1]) 
      self.gradInput[2]:resizeAs(input[2]) 
      self.gradInput[1]:copy(input[1])
      self.gradInput[1]:add(-1, input[2])
      self.gradInput[1]:mul(self.L2_lambda);
      if gradOutput then
	 self.gradInput[1]:mul(gradOutput[1]);
      end
      
      self.gradInput[2]:zero():add(-1, self.gradInput[1])
   else
      error('Illegal self.num_inputs')
   end
   
  return self.gradInput
end
