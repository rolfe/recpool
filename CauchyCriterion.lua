local CauchyCriterion, parent = torch.class('nn.CauchyCriterion', 'nn.Criterion')

function CauchyCriterion:__init(scale)
   parent.__init(self)
   self.scale = scale or 100 --10 
   self.MSEcost = nn.MSECriterion()

   self.abs_dif = torch.Tensor()
   self.abs_dif_temp = torch.Tensor()
end 
 
function CauchyCriterion:updateOutput(input, target)
   --self.output = torch.sum(torch.add(input, -1, target):div(self.scale):pow(2):add(1):mul(math.pi * self.scale):log())
   --print('MSE cost: ' .. math.log(self.MSEcost:updateOutput(input, target)))
   self.abs_dif:resizeAs(input):copy(input):add(-1, target):abs()
   self.abs_dif_temp:resizeAs(self.abs_dif):copy(self.abs_dif)
   self.output = torch.sum(self.abs_dif_temp:add(self.scale):log():mul(-1*self.scale):add(self.abs_dif):mul(self.scale))
   return self.output
end

function CauchyCriterion:updateGradInput(input, target)
   --self.gradInput:resizeAs(input)

   --self.gradInput = torch.add(input, -1, target):mul(2/self.scale):cdiv(torch.add(input, -1, target):div(self.scale):pow(2):add(1))
   --print(self.gradInput:unfold(1,8,8))

   self.gradInput:resizeAs(input):copy(input):add(-1, target)
   self.abs_dif:resizeAs(self.gradInput):copy(self.gradInput):abs()
   --self.abs_dif_temp:resizeAs(self.abs_dif):copy(self.abs_dif)
   self.gradInput:mul(self.scale):cdiv(self.abs_dif:add(self.scale))

   

   return self.gradInput 
end
