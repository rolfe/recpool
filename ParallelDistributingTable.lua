local ParallelDistributingTable, parent = torch.class('nn.ParallelDistributingTable', 'nn.ParallelTable')

-- It's not possible to split the output of a module within a ParallelTable, since the gradOutputs don't align with the modules, and cannot be routed one-to-one
-- Each module added to a PDT takes in the table of all inputs and returns a single tensor as output.  Each module is thus expected to produce a table of gradInputs matching the table of inputs.  However, the entries of this table are allowed to be nil, in which case they are treated like a tensor of all zeros.  In general, PDT should only be used in conjunction with SelectTables, which route both the inputs and gradInputs appropriately.  We'd like to avoid allocating new memory each time updateGradInput is called, so gradInputs will maintain its own gradInput tensors when necessary, and add from the component modules.  If a given input only feeds into a single module, the modules gradOutput is passed through directly, rather than copied to local storage, for efficiency.  

function ParallelDistributingTable:__init(name)
   parent.__init(self)
   self.num_grad_input_sources = torch.Tensor() -- keeps track of how many modules are connected to a given input, so gradOutput from a module can be routed directly to gradInput if only one module connects to the corresponding input
   self.name = name
end

function ParallelDistributingTable:updateOutput(input)
   for i=1,#self.modules do
      -- it would be easy and no less efficient to append a table of outputs from the current module to the end of the list.  However, we would then need to keep track of which gradOutputs get routed to which module
      self.output[i] = self.modules[i]:updateOutput(input) -- rather than input[i], in ParallelTable
   end

   return self.output
end

function ParallelDistributingTable:updateGradInput(input, gradOutput)
   -- determine the number of modules connected to each entry of the input table.  If an input tensor only feeds into one module, the corresponding gradOutput can be copied directly into the PDT's gradInput.  Otherwise, we need to maintain a distinct gradInput tensor and add the contribution from each connected module.
   self.num_grad_input_sources:resize(#input)
   self.num_grad_input_sources:zero()
   for i,module in ipairs(self.modules) do
      for j in pairs(module:updateGradInput(input, gradOutput[i])) do
	 self.num_grad_input_sources[j] = self.num_grad_input_sources[j] + 1
      end
   end

   -- If we're not routing directly from gradOutput, resize the gradInputs based on the inputs, and zero.  
   for i,input_tensor_i in ipairs(input) do
      if self.num_grad_input_sources[i] == 1 then
	 self.gradInput[i] = nil -- if only one module connects to input i, pass the gradOutput through directly
      else
	 if not(self.gradInput[i]) then
	    self.gradInput[i] = input_tensor_i.new() -- create a new tensor of the same type as the input
	 end
	 self.gradInput[i]:resizeAs(input_tensor_i):zero()
      end
   end
   
   -- Iterate over the modules.  Within each module, iterate over the entries of gradInputs using pairs (since some entries will be nil).  For all existent entries, add the gradInput into the PDT's gradInput
   for i,module in ipairs(self.modules) do
      for j,gradInput_tensor_j in pairs(module.gradInput) do -- updateGradInput was already called before
	 if self.gradInput[j] == nil then
	    self.gradInput[j] = gradInput_tensor_j
	 else
	    self.gradInput[j]:add(gradInput_tensor_j)
	 end
      end
   end

   return self.gradInput
end

function ParallelDistributingTable:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   for i,module in ipairs(self.modules) do
      module:accGradParameters(input, gradOutput[i], scale) -- rather than input[i]
   end
end

function ParallelDistributingTable:accUpdateGradParameters(input, gradOutput, lr)
   lr = lr or 1
   for i,module in ipairs(self.modules) do
      module:accUpdateGradParameters(input, gradOutput[i], lr) -- rather than input[i]
   end
end


function ParallelDistributingTable:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = 'nn.ParallelDistributingTable' 
   if self.name then
      str = str .. ' (' .. self.name .. ')'
   end
   str = str .. ' {' .. line .. tab .. 'input'
   for i=1,#self.modules do
      if i == #self.modules then
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. 'output'
   str = str .. line .. '}'
   return str
end
