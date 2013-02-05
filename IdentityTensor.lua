local IdentityTensor, parent = torch.class('nn.IdentityTensor', 'nn.Module')

-- Convert a scaler input into a tensor output (with a single dimension of size 1); pass the gradient through unchanged

function IdentityTensor:__init()
   parent.__init(self)
   self.output = torch.Tensor(1)
end

function IdentityTensor:updateOutput(input)
   self.output[1] = input
   return self.output
end


function IdentityTensor:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end
