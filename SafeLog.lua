local SafeLog, parent = torch.class('nn.SafeLog','nn.Module')

-- taking the log of 0 results in infinity; instead compute log(x + offset), where x >= 0.  Since in the end we're interested in x*log(x), which -> 0 as x -> 0, this introduces very little error
function SafeLog:__init(offset)
   parent.__init(self)
   self.offset = offset or 1e-6
end

function SafeLog:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output:add(self.offset):log()
   return self.output
end

function SafeLog:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):copy(input)
   self.gradInput:add(self.offset):pow(-1) 
   self.gradInput:cmul(gradOutput)
   return self.gradInput
end
