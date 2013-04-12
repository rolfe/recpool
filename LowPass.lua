local LowPass, parent = torch.class('nn.LowPass', 'nn.Module')

function LowPass:__init(percentage_new_input, RUN_JACOBIAN_TEST)
   parent.__init(self)
   self.percentage_new_input = percentage_new_input
   self.low_pass = torch.Tensor()
   self.input_average = torch.Tensor()
   self.initialized = false
   self.RUN_JACOBIAN_TEST = RUN_JACOBIAN_TEST
end

function LowPass:updateOutput(input)
   self.output:resizeAs(input) -- This hopefully isn't terribly inefficient, even if the input keeps changing size, since new memory is only allocated when a new maximum is reached
   if not(self.initialized) or self.RUN_JACOBIAN_TEST then
      self.low_pass:resize(1,input:size(input:dim())) -- whether we're using minibatches or not, make the low-pass history (and input average) collapse over the minibatch (the first dimension)
      self.input_average:resize(1,input:size(input:dim())) 
      if input:dim() == 1 then
	 self.low_pass:copy(input)
      elseif input:dim() == 2 then
	 self.low_pass:sum(input, 1):div(input:size(1))
	 if self.RUN_JACOBIAN_TEST then -- Otherwise, the different elements of the minibatch are smooshed together and then re-expanded, and affect the output in complicated ways.  
	    self.low_pass:zero() 
	 end
      else
	 error('input has unexpected number of dimensions: ' .. input:dim())
      end
      self.initialized = true
   end
   
   local expanded_low_pass, current_input_average -- these mask possible changes in whether the input uses minibatches
   self.low_pass:mul(1 - self.percentage_new_input)
   if input:dim() == 1 then
      expanded_low_pass = self.low_pass
      current_input_average = input
   elseif input:dim() == 2 then
      expanded_low_pass = self.low_pass:expandAs(input)
      local num_input_reps = ((input:dim() == 2) and input:size(1)) or 1
      self.input_average:sum(input,1):div(num_input_reps)
      current_input_average = self.input_average
   else 
      error('input has unexpected number of dimensions: ' .. input:dim())
   end
   
   self.output:add(expanded_low_pass, self.percentage_new_input, input) -- low_pass was already scaled down above
   --print('before low_pass update ', self.low_pass) -- low_pass has been scaled down by (1-self.percentage_new_input), but is otherwise unchanged
   --print('adding ', torch.mul(current_input_average, self.percentage_new_input))
   self.low_pass:add(self.percentage_new_input, current_input_average) 
   --print('after low_pass update ', self.low_pass) 

   return self.output
end


function LowPass:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput)
   self.gradInput:copy(gradOutput)

   if not(self.RUN_JACOBIAN_TEST) or (input:dim() == 2) then -- when using RUN_JACOBIAN_TEST with minibatches, accum is zeroed
      self.gradInput:mul(self.percentage_new_input)
   end

   return self.gradInput
end
