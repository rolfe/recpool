local MulConstant, parent = torch.class('nn.MulConstant', 'nn.Module')

function MulConstant:__init(input_size, constant_value)
   parent.__init(self)

   --self.gradInput:resize(input_size)
   --self.output:resize(input_size) 
   
   self.constant_value = constant_value
end

function MulConstant:updateOutput(input)
   self.output:resizeAs(input)
   self.output:copy(input):mul(self.constant_value)
   return self.output
end


function MulConstant:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput)
   self.gradInput:copy(gradOutput):mul(self.constant_value)
   return self.gradInput
end
