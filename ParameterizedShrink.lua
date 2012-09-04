-- A module that performs a "soft"-shrink operation (reduce the magnitude of each output by a constant, with the change bounded at zero), where the magnitude of the shrink is parameterized separately for each input index.
-- if nonnegative_units is true, the outputs are bounded below at zero
-- if ignore_nonnegative_constraint_on_shrink is true, the shrink parameters can go below zero.  This is not computationally sensible, but ensures that estimations of the Jacobian based upon parameter perturbations match those calculated by backpropagation

local ParameterizedShrink, parent = torch.class('nn.ParameterizedShrink', 'nn.Module')

-- EFFICIENCY NOTE: when using non-negative units this could be accomplished more efficiently using an unparameterized, one-sided rectification, just like Glorot, Bordes, and Bengio, along with a non-positive bias in the inverse_dictionary.  However, both nn.SoftShrink and the shrinkage utility method implemented in kex are two-sided.

function ParameterizedShrink:__init(size, nonnegative_units, ignore_nonnegative_constraint_on_shrink)
   parent.__init(self)

   self.shrink_val = torch.Tensor(size):zero() -- this is non-negative
   self.grad_shrink_val = torch.Tensor(size):zero()

   self.negative_shrink_val = torch.Tensor(size):zero() -- precompute shrink_val * -1 used for efficient comparisons
   self.grad_shrink_val_acc = torch.Tensor() -- helper used to compute the value to be accumulated into the gradient
   self.shrunk_indices = torch.ByteTensor() -- indices that have been shrunk to zero, and thus through which the gradient doesn't propagate
   self.shrink_sign = torch.Tensor() -- sign() returns a DoubleTensor

   if type(nonnegative_units) == 'nil' then 
      self.nonnegative_units = true -- use nonnegative units by default
   else 
      self.nonnegative_units = nonnegative_units
   end 

   self.ignore_nonnegative_constraint_on_shrink = ignore_nonnegative_constraint_on_shrink
end

function ParameterizedShrink:repair()
   if not(self.ignore_nonnegative_constraint_on_shrink) then
      self.shrink_val[torch.le(self.shrink_val, 0)] = 0 -- This causes errors to be reported by Jacobian, since the parameter update is not linear in the gradient
   end
   self.negative_shrink_val:mul(self.shrink_val, -1)
end

function ParameterizedShrink:reset(new_shrink_val)
   if type(new_shrink_val) == 'number' then
      self.shrink_val:fill(new_shrink_val)
   elseif type(new_shrink_val) == 'userdata' then
      self.shrink_val:copy(new_shrink_val)
   else
      self.shrink_val:zero()
   end

   self.shrink_val[torch.le(self.shrink_val, 0)] = 0 
   self.negative_shrink_val:mul(self.shrink_val, -1)
end


function ParameterizedShrink:parameters()
   return {self.shrink_val}, {self.grad_shrink_val}
end


-- Koray's shrinkage implementation in kex requires that all elements of a tensor be shrunk by the same amount.  shrinkage_factory produces a shrinkage function which can apply a different shrink magnitude to each element of a tensor, as is required if we have lagrange multipliers on the L1 units or are shrinking the cmul mask

function ParameterizedShrink:updateOutput(input)
   local input_size = input:size() -- necessary, since shrunk_indices is a ByteTensor
   self.output:resize(input_size)
   self.output:copy(input)
   self.shrunk_indices:resize(input_size)

   self.negative_shrink_val:mul(self.shrink_val, -1) -- ONLY NECESSARY FOR JACOBIAN TESTING, since parameters are perturbed directly

   if self.nonnegative_units then
      self.shrunk_indices = torch.le(input, self.shrink_val)
      self.output:add(input, -1, self.shrink_val) -- don't worry about adding to negative values, since they will be set equal to zero by shrunk_indices
   else
      self.shrink_sign:resizeAs(input)
      
      self.shrunk_indices = torch.le(input, self.shrink_val) -- WARNING: this is INEFFICIENT, since torch.le and torch.ge unnecessarily allocate new memory on every iteration
      self.shrunk_indices:cmul(torch.ge(input, self.negative_shrink_val)) -- WARNING: this is INEFFICIENT, since torch.le and torch.ge unnecessarily allocate new memory on every iteration
      self.shrink_sign:sign(input)
      
      self.output:addcmul(-1, self.shrink_val, self.shrink_sign) 
   end
   
   self.output[self.shrunk_indices] = 0
   return self.output
end

-- take this from nonsmooth gradient calculation
function ParameterizedShrink:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput)
   self.gradInput:copy(gradOutput)
   self.gradInput[self.shrunk_indices] = 0 -- this assumes that updateOutput was called before updateGradInput

   return self.gradInput
end

function ParameterizedShrink:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.grad_shrink_val_acc:resizeAs(gradOutput)
   self.grad_shrink_val_acc:mul(gradOutput, -1 * scale) 
   if not(self.nonnegative_units) then
      self.grad_shrink_val_acc:cmul(self.shrink_sign) -- since the shrink_val is nonnegative and multiplied by shrink_sign, a complementary transform must be applied to the gradient
   end
   self.grad_shrink_val_acc[self.shrunk_indices] = 0 -- this assumes that updateOutput was called before updateGradInput and accGradParameters
   self.grad_shrink_val:add(self.grad_shrink_val_acc)
end

function ParameterizedShrink:accUpdateGradParameters(input, gradOutput, lr)
   local this_grad_shrink_val = self.grad_shrink_val
   self.grad_shrink_val = self.shrink_val
   self:accGradParameters(input, gradOutput, -lr)
   self.grad_shrink_val = this_grad_shrink_val

   self:repair()
end

function ParameterizedShrink:updateParameters(learningRate)
   parent.updateParameters(self, learningRate)
   self:repair()
end
