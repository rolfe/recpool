local PrintModule, parent = torch.class('nn.PrintModule', 'nn.Module')

function PrintModule:__init(name)
   parent.__init(self)
   self.name = name or ''
end

function PrintModule:updateOutput(input)
   self.output = input
   if type(self.output) == 'table' then
      print('updateOutput ' .. self.name)
      for i = 1,#input do
	 print(self.output[i]:unfold(1,10,10))
      end
   else
      print('updateOutput ' .. self.name, self.output)
   end
   return self.output
end


function PrintModule:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   --print('updateGradInput ' .. self.name, self.gradInput)
   return self.gradInput
end
