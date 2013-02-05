-- Wrap a criterion, for which only a single input comes from the network, into a module.  

local SumCriterionModule, parent = torch.class('nn.SumCriterionModule', 'nn.Module')

function SumCriterionModule:__init()
   self.output = 0
   self.gradInput = torch.Tensor()
end

function SumCriterionModule:updateOutput(input) 
   self.output = input:sum()
   return self.output
end
    
function SumCriterionModule:updateGradInput(input, gradOutput) -- we leave out the standard gradOutput argument here to make it clear that SumCriterionModule's gradInput is sui generis
   self.gradInput:resizeAs(input):fill(1)
   if gradOutput then
      self.gradInput:mul(gradOutput[1]);
   end
   return self.gradInput
end 
