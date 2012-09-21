local NormalizeTable, parent = torch.class('nn.NormalizeTable', 'nn.Module')

function NormalizeTable:__init()
   --parent.__init(self)
   self.output = {}
   self.gradInput = {}
   self.grad_accum = {}
   self.norm = {}
end

-- for each element of the output table, divisively normalize by the L2 norm: x_i -> x_i / sqrt( 1e-5 + \sum_j x_j^2 ).  The constant added to the normalization avoids nans when all x_i = 0
function NormalizeTable:updateOutput(input)
   for i=1,#input do
      self.output[i] = self.output[i] or torch.Tensor()
      self.output[i]:resizeAs(input[i]):copy(input[i])
      self.norm[i] = math.sqrt(math.pow(input[i]:norm(), 2) + 1e-8) -- avoids nans if input[i] is all zeros; relies upon input[i] being non-negative
      self.output[i]:div(self.norm[i])
   end
   return self.output
end

function NormalizeTable:updateGradInput(input, gradOutput)
   -- nn.CMulTable divides the output by the input to efficiently calculate the product of all but one input.  However, if one of the input == 0, this results in a nan.  We could probably use that more efficient strategy if we could efficiently set all nans in the output to zero, or better yet, go through and set all elements of the gradInput to zero for which the corresponding input is zero (so if the input is actually nan, this is preserved in the gradInput).  The present solution is less efficient, but doesn't require any C code.
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or torch.Tensor()
      local dot = torch.dot(input[i], gradOutput[i])
      self.gradInput[i]:resizeAs(self.output[i]):copy(input[i])
      self.gradInput[i]:mul(-1 * dot/math.pow(self.norm[i], 3))
      self.gradInput[i]:add(1/self.norm[i], gradOutput[i])
   end
   return self.gradInput
end
