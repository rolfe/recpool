local ScaledGradDiag,parent = torch.class('nn.ScaledGradDiag','nn.Diag')

function ScaledGradDiag:__init(nFeature, learningRateScaling)
   self.learningRateScaling = learningRateScaling -- added by Jason 6/13/12

   parent.__init(self, nFeature)
end

function ScaledGradDiag:updateParameters(learningRate)
   if self.learningRateScaling then 
      learningRate = learningRate * self.learningRateScaling -- added by Jason 6/13/12
   end

   parent.updateParameters(self, learningRate)
end
