local InputCMul, parent = torch.class('nn.InputCMul', 'nn.Module')

function InputCMul:__init()
   parent.__init(self)
end


function InputCMul:updateOutput(input)
   -- input effectively contains two one-dimensional tensors of equal length, stored consecutively in input
   if(input:dim() ~= 1) then
      error('Expected 1 dimension in the input to InputCMul:updateOutput, rather than ' .. input:dim())
   end
   local input_size = input:size(1)
   if input_size % 2 ~= 0 then
      error('Expected an input with even length to InputCMul:updateOutput, rather than ' .. input:size(1))
   end

   local input_1 = input:narrow(1, 1, input_size/2)
   local input_2 = input:narrow(1,input_size/2 + 1, input_size/2)
   self.output:resizeAs(input_1)
   self.output:cmul(input_1, input_2)

   --print('updateOutput: cmul input is of size ' .. input_size .. ' output is of size ' .. self.output:size()[1])

   return self.output
end

function InputCMul:updateGradInput(input, gradOutput)
   if self.gradInput then
      --self:updateOutput(input) -- This seems redundant, since by assumption updateOutput was just run on the input, but this is done in the Euclidean module, so I've included it here

      local input_size = input:size(1)
      local output_size = input_size/2

      if input_size % 2 ~= 0 then
	 error('Expected an input with even length to InputCMul:updateOutput, rather than ' .. input:size(1))
      end
      if gradOutput:size(1) ~= output_size then
	 error('Expected the output to be half the length of the input')
      end

      local input_1 = input:narrow(1, 1, output_size)
      local input_2 = input:narrow(1,output_size + 1, output_size)
   
      self.gradInput:resizeAs(input)

      --[[
      local const_max = function(x)
	 if x >= 0 then 
	    return math.max(x, 1)
	 else
	    return math.min(x, -1)
	 end
      end
      --]]

      self.gradInput:narrow(1, 1, output_size):copy(input_2):cmul(gradOutput)
      self.gradInput:narrow(1, output_size + 1, output_size):copy(input_1):cmul(gradOutput)

      --[[ -- Lower bound the magnitude of the gradient
      local grad_input_1 = self.gradInput:narrow(1, 1, output_size):copy(input_2)
      local grad_input_2 = self.gradInput:narrow(1, output_size + 1, output_size):copy(input_1)

      grad_input_1[torch.lt(grad_input_1, 1e-3) and torch.ge(grad_input_1,0)] = 1e-3
      grad_input_1:cmul(gradOutput)

      grad_input_2[torch.gt(grad_input_2, -1e-3) and torch.le(grad_input_2,0)] = -1e-3
      grad_input_2:cmul(gradOutput) --]]

      --print('updateGradInput: cmul input is of size ' .. input_size .. '; gradOutput is of size ' .. gradOutput:size(1) .. ' gradInput is of size ' .. self.gradInput:size(1))

      return self.gradInput
   end
end

