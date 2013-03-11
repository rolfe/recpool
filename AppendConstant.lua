local AppendConstant, parent = torch.class('nn.AppendConstant', 'nn.Module')

-- takes a number or table of constant values, and appends them to the end of the selected dimension for each input
function AppendConstant:__init(constant_value, dimension)
   parent.__init(self)
   
   --self.output:resize(input_size) 
   
   self.dimension = dimension or 2
   self.constant_value = constant_value
   if type(self.constant_value) == 'number' then
      self.constant_value = {self.constant_value}
   end
end

function AppendConstant:updateOutput(input)
   local input_size = input:size()
   local current_dimension = math.min(self.dimension, input:dim()) -- make sure that non-batched inputs are handled correctly; two dimensions rather than one
   local orig_extended_dimension = input_size[current_dimension]
   input_size[current_dimension] = input_size[current_dimension] + #self.constant_value 
   self.output:resize(input_size)
   self.output:narrow(current_dimension, 1, orig_extended_dimension):copy(input)
   for i = 1,#self.constant_value do
      self.output:narrow(current_dimension, orig_extended_dimension + i, 1):fill(self.constant_value[i])
   end
   return self.output
end


function AppendConstant:updateGradInput(input, gradOutput)
   local current_dimension = math.min(self.dimension, input:dim()) -- make sure that non-batched inputs are handled correctly; two dimensions rather than one
   self.gradInput = gradOutput:narrow(current_dimension, 1, gradOutput:size(current_dimension) - #self.constant_value)
   return self.gradInput
end
