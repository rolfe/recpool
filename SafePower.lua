local SafePower, parent = torch.class('nn.SafePower','nn.Module')

function SafePower:__init(p)
   parent.__init(self)
   self.pow = p
   if not p then
      error('nn.SafePower(power)')
   end
end

function SafePower:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output:pow(self.pow)
   return self.output
end

function SafePower:updateGradInput(input, gradOutput)
   -- nn.Power divides the output by the input to efficiently calculate input^(pow-1).  However, if input == 0, this results in a nan.  We could probably use that more efficient strategy if we could efficiently set all nans in the gradInput to zero, or better yet, go through and set all elements of the gradInput to zero for which the corresponding input is zero (so if the input is actually nan, this is preserved in the gradInput).  The present solution is less efficient, but doesn't require any C code.
   self.gradInput:resizeAs(input):copy(input):pow(self.pow - 1) 
   self.gradInput:cmul(gradOutput):mul(self.pow)
   return self.gradInput
end
