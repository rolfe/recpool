local AddConstant, parent = torch.class('nn.AddConstant', 'nn.Module')

function AddConstant:__init(input_size, constant_value)
   parent.__init(self)
   
   --self.output:resize(input_size) 

   self.constant_value = constant_value
end

function AddConstant:updateOutput(input)
   self.output:resizeAs(input)
   self.output:copy(input):add(self.constant_value)
   return self.output
end


function AddConstant:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end
