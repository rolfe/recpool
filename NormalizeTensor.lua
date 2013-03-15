local NormalizeTensor, parent = torch.class('nn.NormalizeTensor', 'nn.Module')

function NormalizeTensor:__init()
   parent.__init(self)
   --self.output = torch.Tensor()
   --self.gradInput = torch.Tensor()
   self.norm = nil
end

-- for each element of the output table, divisively normalize by the L2 norm: x_i -> x_i / sqrt( 1e-5 + \sum_j x_j^2 ).  The constant added to the normalization avoids nans when all x_i = 0
function NormalizeTensor:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if input:dim() == 1 then
      self.norm = math.sqrt(input:dot(input) + 1e-8) -- avoids nans if input[i] is all zeros; relies upon input[i] being non-negative
      self.output:div(self.norm)
   elseif input:dim() == 2 then
      --[[
      self.norm = self.norm or {}
      for j=1,input:size(1) do -- EFFICIENCY NOTE: this can be done more efficiently with torch.expand
	 local current_row = input:select(1,j)
	 self.norm[j] = math.sqrt(current_row:dot(current_row) + 1e-8) -- avoids nans if input[i] is all zeros; relies upon input[i] being non-negative
	 self.output:select(1,j):div(self.norm[j])
	 end
      --]]
      -- there's actually no need to copy the input
      self.squared_input = self.squared_input or torch.Tensor()
      self.norm = self.norm or torch.Tensor()
      self.squared_input:resizeAs(input):cmul(input, input)
      self.norm:resize(input:size(1), 1):sum(self.squared_input, 2):sqrt():add(1e-8)
      self.output:cdiv(torch.expandAs(self.norm, input))
   else
      error('expected vector or matrix')
   end

   return self.output
end


function NormalizeTensor:updateGradInput(input, gradOutput)
   -- nn.CMulTable divides the output by the input to efficiently calculate the product of all but one input.  However, if one of the input == 0, this results in a nan.  We could probably use that more efficient strategy if we could efficiently set all nans in the output to zero, or better yet, go through and set all elements of the gradInput to zero for which the corresponding input is zero (so if the input is actually nan, this is preserved in the gradInput).  The present solution is less efficient, but doesn't require any C code.
   self.gradInput:resizeAs(self.output):copy(input)

   if input:dim() == 1 then
      local dot = torch.dot(input, gradOutput)
      self.gradInput:mul(-1 * dot/math.pow(self.norm, 3))
      self.gradInput:add(1/self.norm, gradOutput)
   elseif input:dim() == 2 then
      --[[
      for j=1,input:size(1) do
	 local current_input = input:select(1,j)
	 local current_gradOutput = gradOutput:select(1,j)
	 local current_gradInput = self.gradInput:select(1,j)
	 
	 local dot = torch.dot(current_input, current_gradOutput)
	 current_gradInput:mul(-1 * dot/math.pow(self.norm[j], 3))
	 current_gradInput:add(1/self.norm[j], current_gradOutput)
      end
      --]]
      self.input_gradOutput_prod = self.input_gradOutput_prod or torch.Tensor()
      self.input_gradOutput_prod_sum = self.input_gradOutput_prod_sum or torch.Tensor()
      self.norm_cubed = self.norm_cubed or torch.Tensor()
      self.norm_cubed:resizeAs(self.norm):pow(self.norm, 3)
      self.input_gradOutput_prod:resizeAs(input):cmul(input, gradOutput)
      self.input_gradOutput_prod_sum:resize(input:size(1), 1):sum(self.input_gradOutput_prod, 2):cdiv(self.norm_cubed):mul(-1)
      self.gradInput:cmul(torch.expandAs(self.input_gradOutput_prod_sum, input))
      self.gradInput:addcdiv(gradOutput, torch.expandAs(self.norm, input))
      
   else
      error('expected vector or matrix')
   end

   return self.gradInput
end
