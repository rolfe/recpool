local Transpose, parent = torch.class('nn.Transpose', 'nn.Module')

-- Transpose the input tensor if it has more than one dimension

function Transpose:__init()
   parent.__init(self)
end

function Transpose:updateOutput(input)
   if input:dim() > 1 then
      self.output = input:transpose(1,input:dim())
   else
      self.output = input
   end
   return self.output
end


function Transpose:updateGradInput(input, gradOutput)
   if input:dim() > 1 then
      self.gradInput = gradOutput:transpose(1,input:dim())
   else
      self.gradInput = gradOutput
   end
   return self.gradInput
end
