local SumWithinBatch, parent = torch.class('nn.SumWithinBatch', 'nn.Module')

function SumWithinBatch:__init()
   parent.__init(self)
   --self.output = torch.Tensor()
   --self.gradInput = torch.Tensor()
end

-- for each element of the output table, divisively normalize by the L1 norm: x_i -> x_i / ( 1e-8 + \sum_j x_j ).  The constant added to the normalization avoids nans when all x_i = 0
function SumWithinBatch:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(1)
      self.output[1] = torch.sum(input)
   elseif input:dim() == 2 then
      self.sum_result = self.sum_result or torch.Tensor()
      self.sum_result:resize(input:size(1)):sum(input, 2)
      self.output = self.sum_result:select(2,1) -- make sure that the output is one-dimensional
   else
      error('expected vector or matrix')
   end

   return self.output
end


function SumWithinBatch:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)

   if input:dim() == 1 then
      if (gradOutput:dim() ~= 1) or (gradOutput:size(1) ~= 1) then
	 error('input dimensions did not match gradOutput dimensions')
      end
      self.gradInput:fill(gradOutput[1])
   elseif input:dim() == 2 then
      if (gradOutput:dim() ~= 1) or (gradOutput:size(1) ~= input:size(1)) then
	 print('sizes are', input:size(), gradOutput:size(), self.output:size())
	 error('input dimensions did not match gradOutput dimensions')
      end
      self.gradOutput_resize = self.gradOutput_resize or torch.Tensor()
      self.gradOutput_resize:resize(gradOutput:size(1), 1)
      self.gradOutput_resize:copy(gradOutput)
      self.gradInput:copy(self.gradOutput_resize:expandAs(input))
   else
      error('expected vector or matrix')
   end

   return self.gradInput
end
