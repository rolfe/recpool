local LowPass, parent = torch.class('nn.LowPass', 'nn.Module')

function LowPass:__init(percentage_new_input, RUN_JACOBIAN_TEST)
   parent.__init(self)
   self.percentage_new_input = percentage_new_input
   self.initialized = false
   self.RUN_JACOBIAN_TEST = RUN_JACOBIAN_TEST
end

function LowPass:updateOutput(input)
   if not(self.initialized) or self.RUN_JACOBIAN_TEST then
      self.output:resizeAs(input)
      self.output:copy(input)
      self.initialized = true
   end
   
   self.output:mul(1 - self.percentage_new_input):add(self.percentage_new_input, input)
   return self.output
end


function LowPass:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput)
   self.gradInput:copy(gradOutput)

   if not(self.RUN_JACOBIAN_TEST) then
      self.gradInput:mul(self.percentage_new_input)
   end

   return self.gradInput
end
