local MultiplicativeFilter, parent = torch.class('nn.MultiplicativeFilter', 'nn.Module')

function MultiplicativeFilter:__init(input_size, forbid_randomize)
   parent.__init(self)

   --self.gradInput:resize(input_size)
   --self.output:resize(input_size) 
   
   -- the filter cannot be named 'weight' or 'bias', or the neural network modules will attempt to train it
   self.bias_filter = torch.Tensor(1, input_size) -- Cmul only depends upon the number of arguments being the same, so no minibatches is the same as one minibatch
   self.forbid_randomize = forbid_randomize
   self.filter_active = torch.Tensor(1):fill(1)
   self:randomize()
end

function MultiplicativeFilter:randomize()
   local perturbation_type = 'dropout' -- 'continuous_uniform'
   if not(forbid_randomize) then
      if perturbation_type == 'continuous_uniform' then
	 self.bias_filter:copy(torch.rand(self.bias_filter:size(2))):mul(0.1):add(0.95)
      elseif perturbation_type == 'dropout' then 
	 self.bias_filter:copy(torch.rand(self.bias_filter:size(2))) -- the multiplicative bias should be mean-1
	 self.bias_filter:add(-0.1):sign():add(1):mul(0.5)
      end
      --print(self.bias_filter)
   end
end

function MultiplicativeFilter:activate()
   self.filter_active[1] = 1
end

function MultiplicativeFilter:inactivate()
   self.filter_active[1] = 0
end

function MultiplicativeFilter:updateOutput(input)
   self.output:resizeAs(input)
   if self.filter_active[1] == 1 then
      --print(self.bias_filter)
      local expanded_bias = self.bias_filter
      if input:dim() == 2 then
	 expanded_bias = torch.expandAs(self.bias_filter, input)
      end
      self.output:cmul(input, expanded_bias)
   else
      self.output:copy(input)
   end
   return self.output
end


function MultiplicativeFilter:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput)
   if self.filter_active[1] == 1 then
      local expanded_bias = self.bias_filter
      if input:dim() == 2 then
	 expanded_bias = torch.expandAs(self.bias_filter, input)
      end
      self.gradInput:cmul(gradOutput, expanded_bias)
   else
      self.gradInput:copy(gradOutput)
   end
   return self.gradInput
end
