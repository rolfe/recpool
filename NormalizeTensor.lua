local NormalizeTensor, parent = torch.class('nn.NormalizeTensor', 'nn.Module')

function NormalizeTensor:__init()
   parent.__init(self)
   --self.output = torch.Tensor()
   --self.gradInput = torch.Tensor()
   self.norm = 0
end

-- for each element of the output table, divisively normalize by the L2 norm: x_i -> x_i / sqrt( 1e-5 + \sum_j x_j^2 ).  The constant added to the normalization avoids nans when all x_i = 0
function NormalizeTensor:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.norm = math.sqrt(input:dot(input) + 1e-8) -- avoids nans if input[i] is all zeros; relies upon input[i] being non-negative
   self.output:div(self.norm)
   return self.output
end


function NormalizeTensor:updateGradInput(input, gradOutput)
   -- nn.CMulTable divides the output by the input to efficiently calculate the product of all but one input.  However, if one of the input == 0, this results in a nan.  We could probably use that more efficient strategy if we could efficiently set all nans in the output to zero, or better yet, go through and set all elements of the gradInput to zero for which the corresponding input is zero (so if the input is actually nan, this is preserved in the gradInput).  The present solution is less efficient, but doesn't require any C code.
   local dot = torch.dot(input, gradOutput)
   self.gradInput:resizeAs(self.output):copy(input)
   self.gradInput:mul(-1 * dot/math.pow(self.norm, 3))
   self.gradInput:add(1/self.norm, gradOutput)
   
   return self.gradInput
end
