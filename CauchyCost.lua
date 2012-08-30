-- A sparsifying regularizer L(x) = 0.5 * \sum_i log(1 + x_i^2), similar to L1, which exactly induces pooling when used in a bilinear reconstruction, where the other set of variables is subject to an L2 regularizer, and the weight matrix consists of disjoint collections of all-ones.  

local CauchyCost, parent = torch.class('nn.CauchyCost', 'nn.Module')

function CauchyCost:__init(cauchy_lambda)
   parent.__init(self)
   if type(cauchy_lambda) ~= 'number' then
      error('cauchy_lambda input to CauchyCost is of type ' .. type(cauchy_lambda))
   end
   self.cauchy_lambda = cauchy_lambda

   -- self.gradInput is properly initialized by nn.Module
   self.output = 1
   self.intermediate_1_p_x_sq = torch.Tensor() -- store (1 + x_i^2)
   self.intermediate_log_1_p_x_sq = torch.Tensor() -- store log(1 + x_i^2)
end 
  
function CauchyCost:updateOutput(input)
   -- L(x) = 0.5 * \sum_i log(1 + x_i^2) -- THIS IS ELEMENT-WISE!!!
   -- dL/dx_i = x_i / ( 1 + x_i^2)

   self.intermediate_1_p_x_sq:resizeAs(input):copy(input):pow(2):add(1)
   self.intermediate_log_1_p_x_sq:resizeAs(input):copy(self.intermediate_1_p_x_sq):log()
   self.output = self.cauchy_lambda * 0.5 * torch.sum(self.intermediate_log_1_p_x_sq)
   return self.output
end

function CauchyCost:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):copy(input):cdiv(self.intermediate_1_p_x_sq):mul(self.cauchy_lambda)
   return self.gradInput
end
