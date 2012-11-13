-- Wrap a criterion, for which only a single input comes from the network, into a module.  The criterion can also take a fixed target, specified by setTarget.  DEBUG_MODE, presently disable, potentially outputs diagnostic messages during each operation.

local L1CriterionModule, parent = torch.class('nn.L1CriterionModule', 'nn.Module')

function L1CriterionModule:__init(criterion, initial_lambda, desired_criterion_value, learning_rate_scaling_factor)
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
   self.desired_criterion_value = desired_criterion_value
   self.learning_rate_scaling_factor = learning_rate_scaling_factor or 1
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
   
   self.criterion_output = self.criterion:updateOutput(input, self.target) -- self.target is ignored by L1Cost, but used by ClassNLLCriterion
   self.output = self.lambda * self.criterion_output
   --self.output = self.criterion_output

   return self.output
end
    
function L1CriterionModule:updateGradInput(input) -- we leave out the standard gradOutput argument here to make it clear that L1CriterionModule's gradInput is sui generis
   if self.weight then
      self.lambda = self.weight[1]
   end
   
   local criterion_grad_input = self.criterion:updateGradInput(input, self.target)
   self.gradInput:resizeAs(criterion_grad_input)
   self.gradInput:mul(criterion_grad_input, self.lambda)
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
