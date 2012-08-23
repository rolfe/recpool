local CopyTable, parent = torch.class('nn.CopyTable', 'nn.Module')

function CopyTable:__init(copy_index, number_of_copies, DEBUG_MODE)
   parent.__init(self)
   self.copy_index = copy_index or 1
   self.number_of_copies = number_of_copies or 2
   self.DEBUG_MODE = DEBUG_MODE or false
   self.output = {}
   self.gradInput = {}
end


-- it's not possible to split the output of a module within a ParallelTable, since the gradOutputs don't align with the modules, and cannot be routed one-to-one
-- CopyTable takes in all inputs and duplicates a single selected index.  This is a specialization of Linear with a non-square matrix of zeros and ones


function CopyTable:updateOutput(input)
   if type(input) ~= 'table' then
      error('input was of type ', type(input), ' but expected a table of tensors')
   elseif self.copy_index > #input then
      error('copy_index ' .. self.copy_index ' > the number of inputs ' .. #inputs)
   end
   
   for i=1,self.copy_index-1 do
      self.output[i] = input[i]
   end
   
   for i=1,self.number_of_copies do
      self.output[self.copy_index + i - 1] = input[self.copy_index]
   end

   for i=self.copy_index+1,#input do 
      self.output[self.number_of_copies - 1 + i] = input[i]
   end

   if self.DEBUG_MODE then
      print('input to CopyTable ' .. self.copy_index .. ', ', self.number_of_copies, ' is ', input)
      print('output from CopyTable is ', self.output)
   end

   return self.output
end



-- All modules receive all inputs, and so should be expected to produce a table of gradInputs matching the table of inputs.  In general, PDT should only be used in conjunction with SelectTables, which route both the inputs and gradInputs appropriately.  We'd like to avoid allocating new memory each time updateGradInput is called, so gradInputs will maintain its own gradInput tensors, and add from the component modules

function CopyTable:updateGradInput(input, gradOutput)
   -- Resize the gradInputs based on the inputs, and zero.  
   for i = 1,#gradOutput do
      if i < self.copy_index then
	 self.gradInput[i] = gradOutput[i] --.new():resizeAs(gradOutput[i]):zero()
      elseif (i == self.copy_index) and (self.number_of_copies >= 1) then
	 self.gradInput[i] = self.gradInput[i] or gradOutput[i].new()
	 self.gradInput[i]:resizeAs(gradOutput[i]):copy(gradOutput[i])
      elseif i <= self.copy_index + self.number_of_copies - 1 then
	 self.gradInput[self.copy_index]:add(gradOutput[i])
      else
	 self.gradInput[i - (self.number_of_copies - 1)] = gradOutput[i] --.new():resizeAs(gradOutput[i]):zero()
      end
   end
   
   return self.gradInput
end


function CopyTable:updateGradInputDEBUG(input, gradOutput)
   -- Resize the gradInputs based on the inputs, and zero.  
   for i = 1,#gradOutput do
      self.gradInput[i] = gradOutput[i]
   end

   return self.gradInput
end

function CopyTable:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = 'nn.CopyTable' .. tab .. self.copy_index .. ', ' .. self.number_of_copies .. ' times'
   --str = str .. ' {' .. line .. tab .. 'input'
   return str
end
