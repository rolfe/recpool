-- Specialize the nn.Linear module to have zero bias, non-negative weight matrix, and/or a weight matrix with columns normalized to have L2 norm of one.  We build this on top of nn.Linear rather than defining a potentially more efficient class from scratch so that any changes to nn.Linear automatically propagate to nn.ConstrainedLinear.  

-- desired_constraints is an array of strings specifying the constraints to be imposed.  If disable_normalized_updates is true, then the constraints are only imposed during initialization

local ConstrainedLinear, parent = torch.class('nn.ConstrainedLinear', 'nn.Linear')

function ConstrainedLinear:__init(input_size, output_size, desired_constraints, disable_normalized_updates)
   parent.__init(self, input_size, output_size)

   local defined_constraints = {'normalized_columns', 'normalized_columns_pooling', 'no_bias', 'non_negative', 'threshold_normalized_rows'}
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

   if self.normlized_columns and self.threshold_normalized_rows then
      error('Cannot have both normalized columns and normalized rows')
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
local function do_normalize_columns(m, desired_norm_value)
   desired_norm_value = desired_norm_value or 1
   for i=1,m:size(2) do
      m:select(2,i):div(m:select(2,i):norm()/desired_norm_value + 1e-12)
   end
end

-- ensure that the dictionary matrix has columns of some minimum threshold
local function do_threshold_normalize_rows(m)
   local threshold = 0.05 --0.1 -- this is probably too large; try 0.25
   for i=1,m:size(1) do
      local row_norm = m:select(1,i):norm()
      if row_norm < threshold then
	 m:select(1,i):div(row_norm/threshold + 1e-12) 
      end
   end
end

function ConstrainedLinear:repair() -- after any sort of update or initialization, enforce the desired constraints
   if self.normalized_columns then
      do_normalize_columns(self.weight)
   elseif self.normalized_columns_pooling then
      if(self.weight:size(1) < self.weight:size(2)) then
	 error('Did not expect output dimension to be smaller than input dimension for ConstrainedLinear with normalized_columns_pooling')
      end
      do_normalize_columns(self.weight, math.sqrt(self.weight:size(1)/self.weight:size(2)))
   elseif self.threshold_normalized_rows then
      do_threshold_normalize_rows(self.weight)
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
