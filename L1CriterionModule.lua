local L1CriterionModule, parent = torch.class('nn.L1CriterionModule', 'nn.Module')

function L1CriterionModule:__init(criterion)
   self.criterion = criterion
   self.gradInput = criterion.gradInput
end

function L1CriterionModule:updateOutput(input) 
   self.output = self.criterion:updateOutput(input)
   return self.output
end
    
function L1CriterionModule:updateGradInput(input, gradOutput)
  self.gradInput = self.criterion:updateGradInput(input)
  return self.gradInput
end 
