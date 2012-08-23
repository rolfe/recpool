local L2Cost, parent = torch.class('nn.L2Cost', 'nn.Module')

function L2Cost:__init(L2_lambda)
   parent.__init(self)
   print('L2Cost: L2_lambda is ', L2_lambda)
   io.read()

   self.L2_lambda = L2_lambda

   -- state
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   --self.output = torch.Tensor(1)
   self.output = 1
end 
  
function L2Cost:updateOutput(input)
   --self.output[1]=input[1]:dist(input[2])
   --self.output:pow(2):mul(1/2); -- we want the sum of squares of the difference, not the square root of the sum of squares
   self.output = self.L2_lambda * 0.5 * math.pow(input[1]:dist(input[2]), 2)
   return self.output
end

function L2Cost:updateGradInput(input, gradOutput)
  self.gradInput[1]:resizeAs(input[1]) 
  self.gradInput[2]:resizeAs(input[2]) 
  self.gradInput[1]:copy(input[1])
  self.gradInput[1]:add(-1, input[2])
  self.gradInput[1]:mul(self.L2_lambda);
  if gradOutput then
     self.gradInput[1]:mul(gradOutput[1]);
  end

  self.gradInput[2]:zero():add(-1, self.gradInput[1])
  return self.gradInput
end
