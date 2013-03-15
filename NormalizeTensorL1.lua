local NormalizeTensorL1, parent = torch.class('nn.NormalizeTensorL1', 'nn.Module')

function NormalizeTensorL1:__init()
   parent.__init(self)
   --self.output = torch.Tensor()
   --self.gradInput = torch.Tensor()
   self.norm = nil
end

-- for each element of the output table, divisively normalize by the L1 norm: x_i -> x_i / ( 1e-8 + \sum_j x_j ).  The constant added to the normalization avoids nans when all x_i = 0
function NormalizeTensorL1:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if input:dim() == 1 then
      self.norm = torch.sum(input) + 1e-8 -- avoids nans if input[i] is all zeros; relies upon input[i] being non-negative
      self.output:div(self.norm)
   elseif input:dim() == 2 then
      self.norm = self.norm or torch.Tensor()
      self.norm:resize(input:size(1), 1):sum(input, 2):add(1e-8)
      self.output:cdiv(torch.expandAs(self.norm, input))
   else
      error('expected vector or matrix')
   end

   return self.output
end


function NormalizeTensorL1:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(self.output)

   if input:dim() == 1 then
      local dot = torch.dot(input, gradOutput)
      self.gradInput:fill(-1 * dot/math.pow(self.norm, 2))
      self.gradInput:add(1/self.norm, gradOutput)
   elseif input:dim() == 2 then
      self.input_gradOutput_prod = self.input_gradOutput_prod or torch.Tensor()
      self.input_gradOutput_prod_sum = self.input_gradOutput_prod_sum or torch.Tensor()
      self.norm_squared = self.norm_squared or torch.Tensor()

      self.norm_squared:resizeAs(self.norm):pow(self.norm, 2)
      self.input_gradOutput_prod:resizeAs(input):cmul(input, gradOutput)
      self.input_gradOutput_prod_sum:resize(input:size(1), 1):sum(self.input_gradOutput_prod, 2):cdiv(self.norm_squared):mul(-1)
      self.gradInput:copy(torch.expandAs(self.input_gradOutput_prod_sum, input))
      self.gradInput:addcdiv(gradOutput, torch.expandAs(self.norm, input))
      
   else
      error('expected vector or matrix')
   end

   return self.gradInput
end
