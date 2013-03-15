local SafeEntropy, parent = torch.class('nn.SafeEntropy','nn.Module')

-- taking the log of 0 results in infinity; instead compute -x*log(x + offset), where x >= 0.  
function SafeEntropy:__init(offset)
   parent.__init(self)
   self.offset = offset or 1e-8
   self.offsetLog = torch.Tensor()
   self.divisor = torch.Tensor()
   self.ratio = torch.Tensor()
end

function SafeEntropy:updateOutput(input)
   self.offsetLog:resizeAs(input):copy(input)
   self.offsetLog:add(self.offset):log()
   self.output:cmul(self.offsetLog, input)
   self.output:mul(-1)
   return self.output
end

function SafeEntropy:updateGradInput(input, gradOutput)
   --self.gradInput:resizeAs(input):copy(input)
   --self.gradInput:add(self.offset):log()
   self.divisor:resizeAs(input):copy(input):add(self.offset)
   self.ratio:resizeAs(input):cdiv(input, self.divisor)
   self.gradInput:resizeAs(input):add(self.offsetLog, self.ratio)
   self.gradInput:cmul(gradOutput)
   self.gradInput:mul(-1)
   return self.gradInput
end
