local SafeCMulTable, parent = torch.class('nn.SafeCMulTable', 'nn.Module')

function SafeCMulTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function SafeCMulTable:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])
   for i=2,#input do
      self.output:cmul(input[i])
   end
   return self.output
end

function SafeCMulTable:updateGradInput(input, gradOutput)
   -- nn.CMulTable divides the output by the input to efficiently calculate the product of all but one input.  However, if one of the input == 0, this results in a nan.  We could probably use that more efficient strategy if we could efficiently set all nans in the output to zero, or better yet, go through and set all elements of the gradInput to zero for which the corresponding input is zero (so if the input is actually nan, this is preserved in the gradInput).  The present solution is less efficient, but doesn't require any C code.
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or torch.Tensor()
      self.gradInput[i]:resizeAs(input[i]):copy(gradOutput)
      for j=1,#input do
	 if j ~= i then
	    self.gradInput[i]:cmul(input[j])
	 end
      end
   end
   return self.gradInput
end
