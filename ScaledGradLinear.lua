local ScaledGradLinear, parent = torch.class('nn.ScaledGradLinear', 'nn.Linear')

function ScaledGradLinear:__init(inputSize, outputSize, learningRateScaling)
   self.learningRateScaling = learningRateScaling -- added by Jason 6/13/12

   parent.__init(self, inputSize, outputSize)
end

function ScaledGradLinear:reset(stdv)
   local dist
   if type(stdv) == "string" and stdv == "identity" then -- added by Jason 6/12/12
      for i=1,self.weight:size(1) do
	 local j = 0
	 self.weight:select(1, i):apply(function()
					   j = j + 1
					   return ((i == j) and 1) or 0
					end)
	 self.bias[i] = 0
      end
      return
   elseif stdv and type(stdv) == 'number' then
	 stdv = stdv * math.sqrt(3)
   else
      if type(stdv) == 'string' and stdv == 'nonnegative' then -- added by Jason 6/27/12
	 dist = 'nonnegative'
      end
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   
   -- we do this so the initialization is exactly
   -- the same than in previous torch versions
   for i=1,self.weight:size(1) do
      self.weight:select(1, i):apply(function()
					if dist == 'nonnegative' then
					   return torch.uniform(0, stdv)
					else
					   return torch.uniform(-stdv, stdv)
					end
                                     end)
      self.bias[i] = torch.uniform(-stdv, stdv)
   end
end

function ScaledGradLinear:updateParameters(learningRate)
   if self.learningRateScaling then 
      learningRate = learningRate * self.learningRateScaling -- added by Jason 6/13/12
   end

   parent.updateParameters(self, learningRate)
end



function ScaledGradLinear:accGradParameters(input, gradOutput, scale)
   if scale and (scale ~= 1) then
      print('Scale for Linear:accGradParameters is ')
      print(scale)
   end

   parent.accGradParameters(self, input, gradOutput, scale)
end
