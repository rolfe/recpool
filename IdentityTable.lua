local IdentityTable, parent = torch.class('nn.IdentityTable', 'nn.Module')

-- Convert a tensor input into a table of one tensor output; reverse the process when updating the gradient

function IdentityTable:__init()
   parent.__init(self)
   self.output = {}
end

function IdentityTable:updateOutput(input)
   self.output = {input}
   return self.output
end


function IdentityTable:updateGradInput(input, gradOutput)
   --print('received ', gradOutput)
   --print('returning ', gradOutput[1])

   if type(gradOutput) ~= 'table' then
      error('gradOutput received by IdentityTable was not a table')
   end



   self.gradInput = gradOutput[1]
   return self.gradInput
end
