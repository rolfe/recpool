local HingeClassNLLCriterion, parent = torch.class('nn.HingeClassNLLCriterion', 'nn.Criterion')

function HingeClassNLLCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
   self.hinge_point = math.log(0.8)
end

function HingeClassNLLCriterion:updateOutput(input, target)
   if input:dim() == 1 then
      self.output = -math.min(input[target], self.hinge_point)
   elseif input:dim() == 2 then
      local output = 0
      for i=1,target:size(1) do
         output = output - math.min(input[i][target[i]], self.hinge_point)
      end
      if self.sizeAverage then
         output = output / target:size(1)
      end
      self.output = output
   else
      error('matrix or vector expected')
   end
   return self.output
end

function HingeClassNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()

   if input:dim() == 1 then
      if input[target] < self.hinge_point then
	 self.gradInput[target] = -1
      end
   else
      local z = -1
      if self.sizeAverage then
         z = z / target:size(1)
      end
      local gradInput = self.gradInput
      for i=1,target:size(1) do
	 if input[i][target[i]] < self.hinge_point then
	    gradInput[i][target[i]] = z
	 end
      end
   end
   
   return self.gradInput
end
