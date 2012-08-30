-- Module that passes on a sub-array of an array of tensors.
-- SelectTable should only be used in conjunction with ParallelDistributingTable.  SelectTable returns gradInputs of nil rather than all zeros to save time when an input has been selected out
-- ParallelDistributingTable expects each component module to produce a single tensor, despite taking a table of tensors as input.  SelectTable, in contrast, allows its output to take as input and return as gradOutput a table of tensors.  In general, the modules added to a ParallelDistributingTable will be Sequences, beginning with a SelectTable.  Some module in the sequence will take in the table of tensors produced by SelectTable and return a single tensor as output.  

local SelectTable, parent = torch.class('nn.SelectTable', 'nn.Module')

-- selected_indices is an array of indices to be output (in the desired order)
function SelectTable:__init(selected_indices, force_table_output)
   self.selected_indices = selected_indices or {}
   self.force_table_output = force_table_output or false -- normally, if selected_indices only contains a single index, the output is a tensor rather than a table

   parent.__init(self)
   self.output = {}
   self.gradInput = {}
end

function SelectTable:numSelectedIndices()
   return #self.selected_indices 
end

function SelectTable:selectedIndices()
   return self.selected_indices 
end

function SelectTable:updateOutput(input)
   if (#self.selected_indices > 1) or (self.force_table_output == true) then
      self.output = {}
      for i = 1,#self.selected_indices do
	 self.output[i] = input[self.selected_indices[i]]
      end
   elseif #self.selected_indices == 1 then
      self.output = input[self.selected_indices[1]]
   else
      error('self.selected_indices is smaller than one')
   end

   return self.output
end


function SelectTable:updateGradInput(input, gradOutput)
   self.gradInput = {}
   if (#self.selected_indices > 1) or (self.force_table_output == true) then
      for i = 1,#self.selected_indices do
	 self.gradInput[self.selected_indices[i]] = gradOutput[i]
      end
   elseif #self.selected_indices == 1 then
      self.gradInput[self.selected_indices[1]] = gradOutput
   else
      error('self.selected_indices is smaller than one')
   end
   return self.gradInput
end


-- FINISH THIS!!!
function SelectTable:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = 'nn.SelectTable'
   str = str .. line .. tab .. 'selecting input' .. '(' 
   for i=1,#self.selected_indices do
      str = str .. tab .. self.selected_indices[i] 
   end
   str = str .. tab .. ')' .. extlast
   return str
end
