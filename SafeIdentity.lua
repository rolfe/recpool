-- Passes output and gradient through unchanged, unless gradient is nil, in which case gradInput is an Tensor of zeros.  This allows SafeIdentity to be used as an ersatz criterion, after the outputs of actual criteria have been recombined, e.g. using nn.Sum

local SafeIdentity, parent = torch.class('nn.SafeIdentity', 'nn.Module')

function SafeIdentity:updateOutput(input)
   self.output = input
   return self.output
end


function SafeIdentity:updateGradInput(input, gradOutput)
   if gradOutput then -- don't reset gradInput from an empty tensor to a nil
      self.gradInput = gradOutput
   else
      if type(input) == 'number' then
	 self.gradInput:resize(1)
      else
	 self.gradInput:resizeAs(input)
      end
      self.gradInput:zero()
   end
   return self.gradInput
end
