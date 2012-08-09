local Hpsd, parent = torch.class('unsup.Hpsd', 'unsup.PSD')

function Hpsd:setParameters(new_parameters)
   local current_parameters = self:parameters() -- this throws away the gradients of the parameters, which is the second argument returned by self:parameters(), and potentially the hessian, which may be the third parameter

   if #new_parameters ~= #current_parameters then
      error('number of required parameters is ' .. #current_parameters .. ' but number of available parameters is ' .. #new_parameters)
   else
      for i = 1,#current_parameters do
	 if not(new_parameters[i]) then
	    print('WARNING: the ' .. i .. ' parameter was not included in the load file')
	    io.read()
	 end
	 current_parameters[i]:set(new_parameters[i]) -- this only works if all parameters are tensors
	 new_parameters[i] = null -- allows garbage collection
      end
      collectgarbage() -- probably a good idea, since all of the storages associated with current_parameters have been rendered irrelevant
   end
end


-- this differs from psd in having an explicit target
function Hpsd:updateOutput(input, target)
   -- pass through encoder
   local prediction = self.encoder:updateOutput(input)
   -- do FISTA
   local fval,h = self.decoder:updateOutput(input, prediction, target)
   -- calculate prediction error
   local perr = self.predcost:updateOutput(prediction, self.decoder.code)
   -- return total cost
   --print('from PSD: ' .. fval .. ' + ' .. perr .. ' * ' .. self.beta .. ' = ' .. fval + perr*self.beta)
   return fval + perr*self.beta, h
end

-- this differs from psd in having an explicit target
function Hpsd:updateGradInput(input, target, gradOutput)
   --print(self.decoder.code:unfold(1,8,8))
   --io.read()

   -- get gradient from decoder
   --local decgrad = decoder:updateGradInput(input, gradOutput)
   -- get grad from prediction cost
   local predgrad = self.predcost:updateGradInput(self.encoder.output, self.decoder.code)
   predgrad:mul(self.beta)
   self.encoder:updateGradInput(input, predgrad)
end

-- this differs from psd in having an explicit target
function Hpsd:accGradParameters(input, target, gradOutput)
   -- update decoder
   self.decoder:accGradParameters(input)

   -- update encoder
   --[[ begin debug only
   print('make sure debug code is removed')
   local found_inf = nil;
   local function testForHuge(x)
      if x > 1e100 then
	 found_inf = 1;
      end
   end
   
   self.predcost.gradInput:apply(testForHuge)
   if found_inf then
      print('in PSD:accGradParameters - Found a term of predcost.gradInput greater than 1e100')
      print('Predicted output is')
      print(self.encoder.output:unfold(1,8,8))
      print('Actual code is')
      print(self.decoder.code:unfold(1,8,8))
      print('Linear weights are')
      print(self.encoder.weight)
      print('Linear bias is')
      print(self.encoder.bias:unfold(1,8,8))
      io.read()
   end
   --]] --end debug only 

   self.encoder:accGradParameters(input,self.predcost.gradInput)

   --[[
   print('L2 norm of decoder weights is ' .. torch.Tensor(self.decoder.cmul_dictionary.gradWeight:storage()):norm() .. ' ' .. torch.Tensor(self.decoder.L1_dictionary.gradWeight:storage()):norm())
   print('L2 norm of encoder weights is ' .. torch.Tensor(self.encoder:get(1).gradWeight:storage()):norm() .. ' ' .. torch.Tensor(self.encoder:get(3).gradWeight:storage()):norm())

   local cmul_L2_accum = 0
   self.encoder:get(1).gradWeight[{{1,200},{}}]:apply(function(x) cmul_L2_accum = cmul_L2_accum + x^2 end)
   cmul_L2_accum = math.sqrt(cmul_L2_accum)
	
   local cmul_L1_accum = 0
   self.encoder:get(1).gradWeight[{{201,400},{}}]:apply(function(x) cmul_L1_accum = cmul_L1_accum + x^2 end)
   cmul_L1_accum = math.sqrt(cmul_L1_accum)
	
   print('L2 norm of cmul encoder weights is ' .. cmul_L2_accum .. ' ' .. cmul_L1_accum)
   --]]
end

