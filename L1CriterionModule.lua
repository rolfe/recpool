local L1CriterionModule, parent = torch.class('nn.L1CriterionModule', 'nn.Module')

function L1CriterionModule:__init(criterion, DEBUG_MODE)
   self.criterion = criterion
   self.gradInput = criterion.gradInput
   self.DEBUG_MODE = DEBUG_MODE or false
end

function L1CriterionModule:setTarget(target) -- the target need not be set if the criterion's updateOutput and updateGradOutput only take the single argument input; target is then nil by default, and ignored
   self.target = target
end

function L1CriterionModule:updateOutput(input) 
   self.output = self.criterion:updateOutput(input, self.target)
   --[[
   if self.DEBUG_MODE then
      print('target is: ' .. self.target .. '; input is:, ', input)
   end
   --]]
   return self.output
end
    
function L1CriterionModule:updateGradInput(input, gradOutput)
  self.gradInput = self.criterion:updateGradInput(input, self.target)
   --[[
   if self.DEBUG_MODE then
      print('target is: ' .. self.target .. '; input is:, ', input)
      print('gradInput is: ', self.gradInput)
   end
   --]]
  return self.gradInput
end 
