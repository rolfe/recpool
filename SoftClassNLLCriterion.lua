local SoftClassNLLCriterion, parent = torch.class('nn.SoftClassNLLCriterion', 'nn.Criterion')

function SoftClassNLLCriterion:__init(aggregate_incorrect_probs)
   parent.__init(self)
   self.sizeAverage = true
   self.desired_class_prob = 0.6 --0.85 -- we recover the original ClassNLLCriterion if we use self.desired_class_prob = 1
   self.aggregate_incorrect_probs = aggregate_incorrect_probs
   --local desired_prob_tensor = torch.Tensor()

   if aggregate_incorrect_probs then
      self.updateOutput = self.updateOutputAggregate
      self.updateGradInput = self.updateGradInputAggregate
   else
      self.updateOutput = self.updateOutputDirect
      self.updateGradInput = self.updateGradInputDirect
   end
   self.min_aggregate_prob = 1e-5
   self.log_max_desired_prob = math.log(self.desired_class_prob)
   self.min_entropy = -self.desired_class_prob * math.log(self.desired_class_prob) - (1 - self.desired_class_prob) * math.log(1 - self.desired_class_prob)
end


function SoftClassNLLCriterion:aggregate_entropy(log_out_prob)
   if log_out_prob > self.log_max_desired_prob then
      return self.min_entropy
   else
      return -self.desired_class_prob * log_out_prob - (1 - self.desired_class_prob) * math.log(math.max(self.min_aggregate_prob, 1 - math.exp(log_out_prob)))
   end
end

function SoftClassNLLCriterion:aggregate_grad_entropy(log_out_prob)
   if log_out_prob > self.log_max_desired_prob then
      return 0
   end

   local out_prob = math.exp(log_out_prob)
   local output = -self.desired_class_prob
   if 1 - out_prob > self.min_aggregate_prob then
      output = output + (1 - self.desired_class_prob) * out_prob / (1 - out_prob)
   end
   return output
end

function SoftClassNLLCriterion:updateOutputAggregate(input, target)
   if input:dim() == 1 then
      self.output = self:aggregate_entropy(input[target])
   elseif input:dim() == 2 then
      local output = 0
      for i=1,target:size(1) do
         output = output + self:aggregate_entropy(input[i][target[i]])
      end
      if self.sizeAverage then
         output = output / target:size(1)
      end
      self.output = output
   else
      error('matrix or vector expected')
   end
   return self.output
end

function SoftClassNLLCriterion:updateGradInputAggregate(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   
   if input:dim() == 1 then
      self.gradInput[target] = self:aggregate_grad_entropy(input[target])
   else
      for i=1,target:size(1) do
	 self.gradInput[i][target[i]] = self:aggregate_grad_entropy(input[i][target[i]]) * ((self.sizeAverage and 1/target:size(1)) or 1)
      end
   end
   
   return self.gradInput
end



function SoftClassNLLCriterion:updateOutputDirect(input, target)
   if input:dim() == 1 then
      self.gradInput:resizeAs(input):fill(-(1-self.desired_class_prob)/(input:size(1) - 1))
      self.gradInput[target] = -1*self.desired_class_prob
      self.output = self.gradInput:dot(input)
      --self.output = -self.desired_class_prob * input[target]
   elseif input:dim() == 2 then
      self.gradInput:resizeAs(input):fill(-(1-self.desired_class_prob)/(input:size(2) - 1))
      if input:size(1) ~= target:size(1) then
	 error('input size does not match target size in SoftClassNLLCriterion!!!')
      end
      for i=1,target:size(1) do
	 self.gradInput[i][target[i]] = -1*self.desired_class_prob
      end
      if self.sizeAverage then
         self.gradInput:div(target:size(1))
      end
      self.output = self.gradInput:dot(input)
   else
      error('matrix or vector expected')
   end
   return self.output
end

function SoftClassNLLCriterion:updateGradInputDirect(input, target)
   return self.gradInput
end
