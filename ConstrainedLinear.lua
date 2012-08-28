local ConstrainedLinear, parent = torch.class('nn.ConstrainedLinear', 'nn.Linear')

function ConstrainedLinear:__init(input_size, output_size, desired_constraints, disable_normalized_updates)
   parent.__init(self, input_size, output_size)

   local defined_constraints = {'normalized_columns', 'no_bias', 'non_negative'}
   for i = 1,#defined_constraints do -- set all constraints to false by default; this potentially allows us to do checking later, to ensure that constraints are not undefined
      self[defined_constraints[i]] = false
   end

   for k,v in pairs(desired_constraints) do
      local found_constraint = false
      for i = 1,#defined_constraints do
	 if k == defined_constraints[i] then
	    found_constraint = true
	    break
	 end
      end
      if found_constraint == false then
	 error('constraint ' .. k .. ' is not defined')
      else
	 self[k] = v
      end
   end

   self:repair()

   if disable_normalized_updates then -- when debugging, disable constraints after initialization
      for i = 1,#defined_constraints do -- set all constraints to false by default; this potentially allows us to do checking later, to ensure that constraints are not undefined
	 self[defined_constraints[i]] = false
      end
   end
end

function ConstrainedLinear:updateParameters(learningRate) 
   parent.updateParameters(self, learningRate)
   self:repair()
end

function ConstrainedLinear:accUpdateGradParameters(input, gradOutput, lr) 
   parent.accUpdateGradParameters(self, input, gradOutput, lr) 
   self:repair()
end

-- ensure that the dictionary matrix has normalized columns
local function do_normalize_columns(m)
   for i=1,m:size(2) do
      m:select(2,i):div(m:select(2,i):norm()+1e-12)
   end
end

function ConstrainedLinear:repair() -- after any sort of update or initialization, enforce the desired constraints
   if self.normalized_columns then
      do_normalize_columns(self.weight)
   end
   if self.no_bias then
      self.bias:zero()
   elseif self.non_negative then -- we only need to enforce non-negativity on the bias if it is not already set to zero
      self.bias[torch.lt(self.bias,0)] = 0 -- WARNING: THIS IS UNNECESSARILY INEFFICIENT, since a new tensor is created on each call; reimplement this in C
   end
   if self.non_negative then
      self.weight[torch.lt(self.weight,0)] = 0 -- WARNING: THIS IS UNNECESSARILY INEFFICIENT, since a new tensor is created on each call; reimplement this in C
   end
end 
