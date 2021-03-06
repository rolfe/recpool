-- Specialize the nn.Linear module to have zero bias, non-negative weight matrix, and/or a weight matrix with columns normalized to have L2 norm of one.  We build this on top of nn.Linear rather than defining a potentially more efficient class from scratch so that any changes to nn.Linear automatically propagate to nn.ConstrainedLinear.  

-- desired_constraints is an array of strings specifying the constraints to be imposed.  If disable_normalized_updates is true, then the constraints are only imposed during initialization

local ConstrainedLinear, parent = torch.class('nn.ConstrainedLinear', 'nn.Linear')

local check_for_nans

function ConstrainedLinear:__init(input_size, output_size, desired_constraints, disable_normalized_updates, learning_scale_factor, RUN_JACOBIAN_TEST, learning_scale_factor_scale_factor)
   parent.__init(self, input_size, output_size)

   --disable_normalized_updates = false -- THIS AVOIDS NANS!!!
   self.learning_scale_factor_scale_factor = learning_scale_factor_scale_factor or 1
   self.learning_scale_factor = self.learning_scale_factor_scale_factor * (learning_scale_factor or 1)
   self.RUN_JACOBIAN_TEST = RUN_JACOBIAN_TEST
   
   local defined_constraints = {'normalized_columns', 'normalized_columns_pooling', 'no_bias', 'non_negative', 'normalized_rows', 'normalized_rows_pooling', 'threshold_normalized_rows', 
				'bounded_elements', 'non_negative_diag', 'squared_weight_matrix', 'filter_weight_matrix'}
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


   if self.squared_weight_matrix then
      self.weight_squared = torch.Tensor(self.weight:size())
      self.grad_weight_outer_product = torch.Tensor(self.weight:size())
      if self.RUN_JACOBIAN_TEST then
	 self.stored_weight = torch.Tensor(self.weight:size())
      end
   end
   
   self:repair(true, nil, true)

   if disable_normalized_updates then -- when debugging, disable constraints after initialization
      for i = 1,#defined_constraints do -- set all constraints to false by default; this potentially allows us to do checking later, to ensure that constraints are not undefined
	 if defined_constraints[i] ~= 'squared_weight_matrix' then
	    self[defined_constraints[i]] = false
	 end
      end
   end
end

function ConstrainedLinear:reset_learning_scale_factor(new_scale_factor)
   self.learning_scale_factor = self.learning_scale_factor_scale_factor * new_scale_factor
end

function ConstrainedLinear:get_weight_filter_matrix()
   return self.current_weight_filter_matrix
end

function ConstrainedLinear:set_weight_filter_matrix(new_filter_matrix)
   if self.weight:dim() ~= new_filter_matrix:dim() then
      error('dimension of new filter matrix does not match dimension of weight matrix')
   end
   for i = 1,self.weight:dim() do
      if self.weight:size(i) ~= new_filter_matrix:size(i) then
	 error('the size of dimension' .. i .. ' of the new filter matrix does not match that of the weight matrix')
      end
   end
   self.current_weight_filter_matrix = torch.Tensor():resizeAs(new_filter_matrix):copy(new_filter_matrix)
end

function ConstrainedLinear:initialize_local_connection_weight_filter_matrix(connection_width)
   if (self.weight:dim() ~= 2) or (self.weight:size(1) ~= self.weight:size(2)) or (self.weight:size(1) ~= math.pow(math.floor(math.sqrt(self.weight:size(1))), 2)) then
      print('weight matrix subject to local connection constraint is of size', self.weight:size())
      error('weight matrix subject to local connection constraint is not 2d with side length equal to the square of an integer')
   end

   self.current_weight_filter_matrix = torch.Tensor():resizeAs(self.weight):zero()

   local side_length = math.floor(math.sqrt(self.weight:size(1)))
   for i = 0,side_length-1 do
      for j = 0,side_length-1 do
	 for a = -connection_width,connection_width do
	    for b = -connection_width,connection_width do
	       if (i+a >= 0) and (i+a < side_length) and (j+b >= 0) and (j+b < side_length) then
		  self.current_weight_filter_matrix[{1 + (i+a) + side_length*(j+b), 1 + i + side_length*j}] = 1 
	       end
	    end
	 end
      end
   end
end									    
											    
		  

function ConstrainedLinear:percentage_zeros_per_column(percentage)
   for i = 1,self.weight:size(2) do
      local current_column = self.weight:select(2,i)
      for j = 1,current_column:size(1) do
	 if math.random() < percentage then
	    current_column[j] = 0
	 end
      end
   end
end

function ConstrainedLinear:updateOutput(input)
   -- square the weight matrix, since only the weight matrix itself rather than the squared version is subject to training
   if self.squared_weight_matrix then 
      self.weight_squared:resizeAs(self.weight)
      --self.weight_squared:pow(self.weight, 2)
      self.weight_squared:cmul(self.weight, self.weight)
      
      if self.RUN_JACOBIAN_TEST then -- this is necessary to correctly perform the test all accUpdateGradParameters [shared], which does two accUpdateGradParameters in a row.  If we don't store a copy of the weights, then the first accUpdateGradParameters alters the operation performed by the second, which the test code doesn't expect, even thought it is correct
	 self.stored_weight:resizeAs(self.weight):copy(self.weight) -- DEBUG ONLY!!!
      end
   end

   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      local current_output_scaling_value = 0
      if not(self.no_bias) then
	 self.output:copy(self.bias)
	 current_output_scaling_value = 1
      end
      
      if self.squared_weight_matrix then
	 self.output:addmv(current_output_scaling_value, 1, self.weight_squared, input)	 
      else
	 self.output:addmv(current_output_scaling_value, 1, self.weight, input)
      end

   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)
      
      -- 0*nan = nan, not 0!!!  It's thus necessary to initialize the output using more than just the scaling factor in addr.  It might make sense to create a distinct resize function, and only initialize with :zero after an explicit resize.
      self.output:resize(nframe, nunit):zero() 
      local current_output_scaling_value = 0
      if not(self.no_bias) then
	 self.output:addr(0, 1, input.new(nframe):fill(1), self.bias)
	 current_output_scaling_value = 1
      end

      if self.squared_weight_matrix then
	 self.output:addmm(current_output_scaling_value, 1, input, self.weight_squared:t())
      else
	 self.output:addmm(current_output_scaling_value, 1, input, self.weight:t())
      end
   else
      error('input must be vector or matrix')
   end
   
   return self.output
   
   --return parent.updateOutput(self, input)
end


function ConstrainedLinear:updateGradInput(input, gradOutput)
   if self.squared_weight_matrix then
      if self.gradInput then
	 self.gradInput:resizeAs(input)
	 self.weight_squared:resizeAs(self.weight)
	 --self.weight_squared:pow(self.weight, 2)
	 self.weight_squared:cmul(self.weight, self.weight)
	 if input:dim() == 1 then
	    self.gradInput:addmv(0, 1, self.weight_squared:t(), gradOutput)
	 elseif input:dim() == 2 then
	    self.gradInput:addmm(0, 1, gradOutput, self.weight_squared)
	 else
	    error('input must be a vector or matrix')
	 end
	 
	 return self.gradInput
      end
   else
      return parent.updateGradInput(self, input, gradOutput)
   end
end


function ConstrainedLinear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if self.squared_weight_matrix then 
      self.grad_weight_outer_product:resizeAs(self.weight)

      if input:dim() == 1 then
	 self.grad_weight_outer_product:ger(gradOutput, input)
	 self.gradBias:add(self.learning_scale_factor * scale, gradOutput)      
      elseif input:dim() == 2 then
	 local nframe = input:size(1)
	 local nunit = self.bias:size(1)

	 self.grad_weight_outer_product:mm(gradOutput:t(), input)
	 self.gradBias:addmv(self.learning_scale_factor * scale, gradOutput:t(), input.new(nframe):fill(1))
      else
	 error('input must be a vector or matrix')
      end

      if self.RUN_JACOBIAN_TEST then
	 self.grad_weight_outer_product:cmul(self.stored_weight) -- DEBUG ONLY!!!
      else
	 self.grad_weight_outer_product:cmul(self.weight)
      end

      self.gradWeight:add(self.learning_scale_factor * 2 * scale, self.grad_weight_outer_product)

      --self.gradWeight:cmul(self.weight) -- this only works if gradWeight was zeroed before the call; we can't accumulate into gradWeight
      --self.gradWeight:mul(2)

      --print('using modified accGradParameters')
   else
      return parent.accGradParameters(self, input, gradOutput, self.learning_scale_factor * scale)
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

-- THIS APPEARS TO BE VERY SLOW!  IS IT POSSIBLE TO DO THIS IN C?!?
-- ensure that the dictionary matrix has normalized columns (enforce a maximum norm, but not a minimum, unless full_normalization is true)
local function do_normalize_rows_or_columns(m, desired_norm_value, full_normalization, normalized_dimension, squared_weight_matrix)
   desired_norm_value = desired_norm_value or 1
   if full_normalization then
      -- CONSIDER using pow, which copies to a consistent working tensor, and then sum; this saves us from using a for loop in lua
      for i=1,m:size(normalized_dimension) do -- was 2
	 local selected_vector = m:select(normalized_dimension,i)
	 if squared_weight_matrix then
	    selected_vector:div(selected_vector:norm(4)/math.sqrt(desired_norm_value) + 1e-12) 
	 else
	    local norm_val = math.sqrt(selected_vector:dot(selected_vector))
	    selected_vector:div(norm_val/desired_norm_value + 1e-12)
	 end
	 --m:select(normalized_dimension,i):div(m:select(normalized_dimension,i):norm()/desired_norm_value + 1e-12) -- norm is NOT implemented with BLAS in torch, and so is slow
      end
   else
      for i=1,m:size(normalized_dimension) do
	 local selected_vector = m:select(normalized_dimension,i)
	 if squared_weight_matrix then
	    selected_vector:div(math.max(selected_vector:norm(4)/math.sqrt(desired_norm_value), 1) + 1e-12) 
	 else
	    local norm_val = math.sqrt(selected_vector:dot(selected_vector))
	    selected_vector:div(math.max(norm_val/desired_norm_value, 1) + 1e-12)
	 end
	 --m:select(normalized_dimension,i):div(math.max(m:select(normalized_dimension,i):norm()/desired_norm_value, 1) + 1e-12) -- WARNING!!! THIS WAS MIN RATHER THAN MAX!!!
      end
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

function ConstrainedLinear:repair(full_normalization, desired_norm_value, ignore_uninitialized_weight_filter_matrix) -- after any sort of update or initialization, enforce the desired constraints
   if self.filter_weight_matrix then
      if self.current_weight_filter_matrix then
	 self.weight:cmul(self.current_weight_filter_matrix)
      elseif not(ignore_uninitialized_weight_filter_matrix) then -- repair is called by __init(), at which point the current_weight_matrix should not be expected to be set
	 error('weight filter matrix was not set')
      end
   end
   
   self.repair_already_called_once = true

   if self.non_negative then
      --self.weight[torch.lt(self.weight,0)] = 0 -- WARNING: THIS IS UNNECESSARILY INEFFICIENT, since a new tensor is created on each call; reimplement this in C
      self.weight:maxZero()
   elseif self.non_negative_diag then
      if self.weight:dim() ~= 2 then
	 error('expected two dimensions in tensor to enforce non_negative_diag constraint')
      end

      --local found_neg = false
      -- WARNING: THIS IS UNNECESSARILY INEFFICIENT!  (EFFICIENCY NOTE).  The iteration over the diagonal elements of the weight tensor should be done in C
      for i = 1,math.min(self.weight:size(1), self.weight:size(2)) do --1,torch.Tensor(self.weight:size()):min() do -- this doesn't work, since self.weight:size() is interpreted as the desired sizes
	 --if(self.weight[{i,i}] < 0) then found_neg = true end
	 self.weight[{i,i}] = math.max(0, self.weight[{i,i}])
      end
      --if found_neg then print('correcting diagonal element < 0') end
   end

   if self.normalized_columns then
      do_normalize_rows_or_columns(self.weight, desired_norm_value, full_normalization, 2, self.squared_weight_matrix) -- 2 specificies that the second dimension (columns) should be normalized
   elseif self.normalized_columns_pooling then
      if(self.weight:size(1) < self.weight:size(2)) then
	 error('Did not expect output dimension to be smaller than input dimension for ConstrainedLinear with normalized_columns_pooling')
      end
      -- the L2 norm is the square root of the sum of squares; the L2 norm of a vector of n ones is sqrt(n)
      do_normalize_rows_or_columns(self.weight, math.sqrt(self.weight:size(1)/self.weight:size(2)), full_normalization, 2, self.squared_weight_matrix)
   elseif self.normalized_rows then
      do_normalize_rows_or_columns(self.weight, desired_norm_value, full_normalization, 1, self.squared_weight_matrix) -- 1 specificies that the first dimension (rows) should be normalized
   elseif self.normalized_rows_pooling then
      if(self.weight:size(2) < self.weight:size(1)) then
	 error('Did not expect input dimension to be smaller than output dimension for ConstrainedLinear with normalized_rows_pooling')
      end
      -- the L2 norm is the square root of the sum of squares; the L2 norm of a vector of n ones is sqrt(n)
      do_normalize_rows_or_columns(self.weight, math.sqrt(self.weight:size(2)/self.weight:size(1)), full_normalization, 1, self.squared_weight_matrix) -- 1 specifies that the first dimension (rows) should be normalized
   elseif self.threshold_normalized_rows then
      do_threshold_normalize_rows(self.weight)
      if self.squared_weight_matrix then
	 error('threshold normalized rows is not compatible with squared weight matrix')
      end
   elseif self.bounded_elements then
      if desired_norm_value then
	 self.weight:minN(desired_norm_value)
	 self.weight:maxN(-desired_norm_value)
      end
   end

   if self.no_bias then
      self.bias:zero()
   elseif self.non_negative then -- we only need to enforce non-negativity on the bias if it is not already set to zero
      --self.bias[torch.lt(self.bias,0)] = 0 -- WARNING: THIS IS UNNECESSARILY INEFFICIENT, since a new tensor is created on each call; reimplement this in C
      self.bias:maxZero()
   end
   
   -- DEBUG ONLY!!!
   -- check_for_nans(self)
end 

local function check_for_nans(self)
   local found_a_nan
   local function find_nans(x)
      if x ~= x then
	 found_a_nan = true
      end
   end
   self.weight:apply(find_nans)
   if found_a_nan then
      print('Found a nan in ConstrainedLinear!!!')
      print(self.weight)
      io.read()
   end
end