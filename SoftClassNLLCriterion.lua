local SoftClassNLLCriterion, parent = torch.class('nn.SoftClassNLLCriterion', 'nn.Criterion')

function SoftClassNLLCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
   self.desired_class_prob = 0.9 -- we recover the original ClassNLLCriterion if we use self.desired_class_prob = 1
   --local desired_prob_tensor = torch.Tensor()
end

function SoftClassNLLCriterion:updateOutput(input, target)
   if input:dim() == 1 then
      self.gradInput:resizeAs(input):fill(-(1-self.desired_class_prob)/(input:size(1) - 1))
      self.gradInput[target] = -1*self.desired_class_prob
      self.output = self.gradInput:dot(input)
      --self.output = -self.desired_class_prob * input[target]
   elseif input:dim() == 2 then
      self.gradInput:resizeAs(input):fill(-(1-self.desired_class_prob)/(input:size(2) - 1))
      if input:size(1) ~= target:size(1) then
	 error('input size does not match target size in SoftClassNLLCriterion!!!')
      end
      for i=1,target:size(1) do
	 self.gradInput[i][target[i]] = -1*self.desired_class_prob
      end
      if self.sizeAverage then
         self.gradInput:div(target:size(1))
      end
      self.output = self.gradInput:dot(input)
   else
      error('matrix or vector expected')
   end
   return self.output
end

function SoftClassNLLCriterion:updateGradInput(input, target)
   return self.gradInput
end
