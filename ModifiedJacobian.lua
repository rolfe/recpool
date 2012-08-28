nn.Jacobian = {}

function nn.Jacobian.backward (module, input, param, dparam)
   local doparam = 0
   if param then
      doparam = 1
   end
   param = param or input
   -- output deriv
   module:forward(input)

   local dout = module.output.new():resizeAs(module.output)
   -- 1D view
   local sdout = module.output.new(dout:storage(),1,dout:nElement())
   -- jacobian matrix to calculate
   local jacobian = torch.Tensor(param:nElement(),dout:nElement()):zero()
   local wrapped_dparam = ((type(dparam) == 'table') and dparam) or {dparam} --if parameters are shared between modules, pass in a table of all shared grad_params.  Use wrapping to ensure that both cases can be handeled in the same manner

   for i=1,sdout:nElement() do
      dout:zero()
      sdout[i] = 1
      module:zeroGradParameters()
      local din = module:updateGradInput(input, dout)
      module:accGradParameters(input, dout)
      if doparam == 1 then
	 jacobian:select(2,i):copy(wrapped_dparam[1])
	 for j=2,#wrapped_dparam do
	    jacobian:select(2,i):add(wrapped_dparam[j])
	 end
      else
	 jacobian:select(2,i):copy(din)
      end
   end
   return jacobian
end

function nn.Jacobian.backwardTable (module, input, param, dparam)
   local doparam = 0
   if param then
      doparam = 1
   end
   param = param or input
   -- output deriv
   module:forward(input)

   local jacobian = {}
   local dout = {}

   -- modules like CAddTable return a single tensor, rather than a table of tensors.  Wrap these outputs in a table to allow consistent processing.
   local converted_output = ((type(module.output) == 'number') and torch.Tensor(1):fill(module.output)) or module.output
   local wrapped_output = ((type(converted_output) ~= 'table') and {converted_output}) or converted_output
   if #wrapped_output == 0 then -- if the network includes its criteria and thus produces no output, create a single dummy output, which the network will ignore
      wrapped_output = {torch.Tensor(1):zero()}
   end
   local wrapped_input = ((type(input) ~= 'table') and {input}) or input -- since we also wrap the input, backwardTable can be used even when neither input nor output is a table


   for output_table_index = 1,#wrapped_output do -- create a table of tensors to hold the artificially constructed gradOutput, which is all zeros except for a single one
      dout[output_table_index] = wrapped_output[output_table_index].new():resizeAs(wrapped_output[output_table_index])
      dout[output_table_index]:zero()
   end
   
   local unwrapped_dout = ((type(converted_output) ~= 'table') and dout[1]) or dout

   for output_table_index = 1,#wrapped_output do
      -- 1D view
      local dout_oti = dout[output_table_index]
      local sdout_oti = wrapped_output[output_table_index].new(dout_oti:storage(),1,dout_oti:nElement()) -- create a flattened version of the current artificial gradOutput tensor
      -- jacobian matrix to calculate
      if doparam == 1 then -- the Jacobian of the output with respect to the parameter only requires a single tensor for each output tensor
	 jacobian[output_table_index] = torch.Tensor(param:nElement(),dout_oti:nElement()):zero()
      else -- the Jacobian of the output with respect to the input requires a table of tensors for each output tensor, since each input is also a table of tensors
	 jacobian[output_table_index] = {}
	 for input_table_index = 1,#wrapped_input do
	    jacobian[output_table_index][input_table_index] = torch.Tensor(wrapped_input[input_table_index]:nElement(), wrapped_output[output_table_index]:nElement()):zero()
	 end
      end
      local jacobian_oti = jacobian[output_table_index]
      
      for i=1,sdout_oti:nElement() do
	 sdout_oti:zero()
	 sdout_oti[i] = 1
	 module:zeroGradParameters()
	 local din = module:updateGradInput(input, unwrapped_dout)
	 local wrapped_din = ((type(din) ~= 'table') and {din}) or din
	 module:accGradParameters(input, unwrapped_dout)
	 if doparam == 1 then
	    jacobian_oti:select(2,i):copy(dparam)
	 else
	    for input_table_index = 1,#wrapped_input do
	       jacobian_oti[input_table_index]:select(2,i):copy(wrapped_din[input_table_index])
	    end
	 end
      end -- flattened parameters
      sdout_oti:zero() -- make sure that dout for this output_table_index is reset to all zeros before moving onto the next one
   end -- choice of output table

   return jacobian
end


function nn.Jacobian.backwardUnsup (module, criterion, input, target, param, dparam)
   -- output deriv
   local output = module:forward(input)
   criterion:forward(input, target)
   local jacobian

   criterion:updateGradInput(input, target)
   module:zeroGradParameters()
   local din = module:updateGradInput(input, dout)
   module:accGradParameters(input, dout)
   if type(dparam) == 'table' then
      for j=1,#dparam do
	 jacobian[j] = dparam[j]:clone()
      end
   else
      jacobian = dparam:clone()
   end
   
   return jacobian
end



function nn.Jacobian.backwardUpdate (module, input, param, other_params)

   -- output deriv
   module:forward(input)
   local dout = module.output.new():resizeAs(module.output)
   -- 1D view
   local sdout = module.output.new(dout:storage(),1,dout:nElement())
   -- jacobian matrix to calculate
   local jacobian = torch.Tensor(param:nElement(),dout:nElement()):zero()

   -- original param
   local origparam = param:clone()
   local orig_other_params = {}
   --print('other_params is ', other_params)
   for i=1,#other_params do
      --print('index is', i)
      orig_other_params[i] = other_params[i]:clone()
   end

   for i=1,sdout:nElement() do
      param:copy(origparam)
      for i=1,#other_params do
	 other_params[i]:copy(orig_other_params[i])
      end


      dout:zero()
      sdout[i] = 1
      local din = module:updateGradInput(input, dout)
      module:accUpdateGradParameters(input, dout, 1)
      jacobian:select(2,i):copy(param)
   end

   param:copy(origparam)
   for i=1,#other_params do
      other_params[i]:copy(orig_other_params[i])
   end
   

   return jacobian
end

function nn.Jacobian.forward(module, input, param)
   param = param or input
   -- perturbation amount
   local small = 1e-6
   -- 1D view of input
   --local tst = param:storage()
   local sin = param.new(param):resize(param:nElement())--param.new(tst,1,tst:size())
   -- jacobian matrix to calculate
   local jacobian = torch.Tensor():resize(param:nElement(),module:forward(input):nElement())
   
   local outa = torch.Tensor(jacobian:size(2))
   local outb = torch.Tensor(jacobian:size(2))
   
   for i=1,sin:nElement() do      
      sin[i] = sin[i] - small
      outa:copy(module:forward(input))
      sin[i] = sin[i] + 2*small
      outb:copy(module:forward(input))
      sin[i] = sin[i] - small

      outb:add(-1,outa):div(2*small)
      jacobian:select(1,i):copy(outb)
   end

   return jacobian
end

function nn.Jacobian.forwardTable(module, input, param)
   local doparam = 0
   if param then
      doparam = 1
   end
   param = param or input
   -- perturbation amount
   local small = 1e-6
   -- 1D view of input
   module:forward(input)
   -- modules like CAddTable return a single tensor, rather than a table of tensors.  Wrap these outputs in a table to allow consistent processing.
   -- since we also wrap the input, forwardTable can be used even when neither input nor output is a table
   local converted_output = ((type(module.output) == 'number') and torch.Tensor(1):fill(module.output)) or module.output
   local wrapped_output = ((type(converted_output) ~= 'table') and {converted_output}) or converted_output
   local wrapped_input = ((type(param) ~= 'table') and {param}) or param

   local jacobian = {}
   for output_table_index = 1,#wrapped_output do
      if doparam == 1 then -- the Jacobian of the output with respect to the parameter only requires a single tensor for each output tensor
	 jacobian[output_table_index] = torch.Tensor(param:nElement(),wrapped_output[output_table_index]:nElement()):zero()
      else -- the Jacobian of the output with respect to the input requires a table of tensors for each output tensor, since each input is also a table of tensors
	 jacobian[output_table_index] = {}
	 for input_table_index = 1,#wrapped_input do
	    jacobian[output_table_index][input_table_index] = torch.Tensor(wrapped_input[input_table_index]:nElement(),wrapped_output[output_table_index]:nElement()):zero()
	 end
      end
   end

   local sin_iti -- flattened input, pre-indexed by the input_table_index
   local out_accum -- variable into which we accumulate the gradient

   for input_table_index=1,#wrapped_input do
      sin_iti = wrapped_input[input_table_index].new(wrapped_input[input_table_index]):resize(wrapped_input[input_table_index]:nElement())
      out_accum = {}
      for output_table_index=1,#wrapped_output do
	 out_accum[output_table_index] = torch.Tensor(wrapped_output[output_table_index]:nElement())
      end

      for i=1,sin_iti:nElement() do -- perturb each element of the flattened current input table in turn
	 sin_iti[i] = sin_iti[i] - small
	 module:forward(input)
	 converted_output = ((type(module.output) == 'number') and torch.Tensor(1):fill(module.output)) or module.output
	 wrapped_output = ((type(converted_output) ~= 'table') and {converted_output}) or converted_output
	 for output_table_index=1,#wrapped_output do
	    out_accum[output_table_index]:copy(wrapped_output[output_table_index])
	 end
	 
	 sin_iti[i] = sin_iti[i] + 2*small
	 module:forward(input)
	 converted_output = ((type(module.output) == 'number') and torch.Tensor(1):fill(module.output)) or module.output
	 wrapped_output = ((type(converted_output) ~= 'table') and {converted_output}) or converted_output
	 for output_table_index=1,#wrapped_output do
	    out_accum[output_table_index]:add(-1, wrapped_output[output_table_index]):div(-2*small) -- we subtract the positive perturbation from the negative perturbation, so must multiply by -1
	 end
	 sin_iti[i] = sin_iti[i] - small
	 
	 for output_table_index=1,#wrapped_output do
	    if doparam == 1 then
	       if #wrapped_input ~= 1 then
		  error('Doing forwardTable on params, but chosen input has more than one table element')
	       end
	       jacobian[output_table_index]:select(1,i):copy(out_accum)
	    else
	       jacobian[output_table_index][input_table_index]:select(1,i):copy(out_accum[output_table_index])
	    end
	 end -- output_table_index
      end -- sin element
   end -- input_table_index

   return jacobian
end

function nn.Jacobian.forwardUpdate(module, input, param)
   -- perturbation amount
   local small = 1e-6
   -- 1D view of input
   --local tst = param:storage()
   local sin =  param.new(param):resize(param:nElement())--param.new(tst,1,tst:size())
   -- jacobian matrix to calculate
   local jacobian = torch.Tensor():resize(param:nElement(),module:forward(input):nElement())
   
   local outa = torch.Tensor(jacobian:size(2))
   local outb = torch.Tensor(jacobian:size(2))
   
   for i=1,sin:nElement() do      
      sin[i] = sin[i] - small
      outa:copy(module:forward(input))
      sin[i] = sin[i] + 2*small
      outb:copy(module:forward(input))
      sin[i] = sin[i] - small

      outb:add(-1,outa):div(2*small)
      jacobian:select(1,i):copy(outb)
      jacobian:select(1,i):mul(-1)
      jacobian:select(1,i):add(sin[i])
   end
   return jacobian
end

function nn.Jacobian.testJacobian (module, input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(torch.rand(input:nElement()):mul(inrange):add(minval))
   local jac_fprop = nn.Jacobian.forward(module,input)
   local jac_bprop = nn.Jacobian.backward(module,input)
   local error = jac_fprop-jac_bprop

   print('Max and min of fprop and bprop are ', jac_fprop:max(), jac_bprop:max(), jac_fprop:min(), jac_bprop:min())
   return error:abs():max()
end

function nn.Jacobian.testJacobianTable (module, input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   local wrapped_input = ((type(input) ~= 'table') and {input}) or input
   for input_table_index = 1,#wrapped_input do
      wrapped_input[input_table_index]:copy(torch.rand(wrapped_input[input_table_index]:nElement()):mul(inrange):add(minval))
   end
   local jac_fprop = nn.Jacobian.forwardTable(module,input)
   local jac_bprop = nn.Jacobian.backwardTable(module,input)
   local error = 0
   for output_table_index = 1,#jac_fprop do
      for input_table_index = 1,#(jac_fprop[output_table_index]) do
	 --print('Max and min of fprop and bprop (' .. output_table_index .. ', ' .. input_table_index .. ' are ', 
	 --      jac_fprop[output_table_index][input_table_index]:max(), jac_bprop[output_table_index][input_table_index]:max(), 
	 --      jac_fprop[output_table_index][input_table_index]:min(), jac_bprop[output_table_index][input_table_index]:min())
	 error = math.max(error, (jac_fprop[output_table_index][input_table_index] - jac_bprop[output_table_index][input_table_index]):abs():max())
      end
   end
   return error
end

function nn.Jacobian.testJacobianParameters (module, input, param, dparam, minval, maxval)
   --print('started testJacobianParameters')
   local original_param = param:clone()

   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(torch.rand(input:nElement()):mul(inrange):add(minval))
   param:copy(torch.rand(param:nElement()):mul(inrange):add(minval))
   local jac_bprop = nn.Jacobian.backward(module, input, param, dparam)
   local jac_fprop = nn.Jacobian.forward(module, input, param)
   local error = jac_fprop - jac_bprop

   param:copy(original_param)

   print('Max and min of fprop and bprop are ', jac_fprop:max(), jac_bprop:max(), jac_fprop:min(), jac_bprop:min(), jac_fprop:max() / jac_bprop:max(), jac_fprop:min() /jac_bprop:min())
   return error:abs():max()
end

function nn.Jacobian.testJacobianUpdateParameters (module, input, param, other_params, minval, maxval)
   local original_param = param:clone()

   other_params = other_params or {}
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(torch.rand(input:nElement()):mul(inrange):add(minval))
   param:copy(torch.rand(param:nElement()):mul(inrange):add(minval))
   --print('using for other_params: ', other_params)
   local params_bprop = nn.Jacobian.backwardUpdate(module, input, param, other_params)
   local params_fprop = nn.Jacobian.forwardUpdate(module, input, param)

   local error = params_fprop - params_bprop
   param:copy(original_param)

   print('Max and min of fprop and bprop are ', params_fprop:max(), params_bprop:max(), params_fprop:min(), params_bprop:min(), params_fprop:max() / params_bprop:max(), params_fprop:min() / params_bprop:min())
   return error:abs():max()
end

function nn.Jacobian.testIO(module,input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval

   -- run module
   module:forward(input)
   local go = module.output:clone():copy(torch.rand(module.output:nElement()):mul(inrange):add(minval))
   module:zeroGradParameters()
   module:updateGradInput(input,go)
   module:accGradParameters(input,go)

   local fo = module.output:clone()
   local bo = module.gradInput:clone()

   -- write module
   local f = torch.DiskFile('tmp.bin','w'):binary()
   f:writeObject(module)
   f:close()
   -- read module
   local m = torch.DiskFile('tmp.bin'):binary():readObject()
   m:forward(input)
   m:zeroGradParameters() -- SHOULDN'T THIS BE m, RATHER THAN module?!?
   m:updateGradInput(input,go)
   m:accGradParameters(input,go)
   -- cleanup
   os.remove('tmp.bin')

   local fo2 = m.output:clone()
   local bo2 = m.gradInput:clone()

   local errf = fo - fo2
   local errb = bo - bo2
   return errf:abs():max(), errb:abs():max()
end

function nn.Jacobian.testIOTable(module,input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval

   -- run module
   module:forward(input)
   local converted_output = ((type(module.output) == 'number') and torch.Tensor(1):fill(module.output)) or module.output
   local wrapped_output = ((type(converted_output) == 'table') and converted_output) or {converted_output}
   local go = {}
   local fo = {}
   for i = 1,#wrapped_output do
      fo[i] = wrapped_output[i]:clone()
      go[i] = wrapped_output[i]:clone():copy(torch.rand(wrapped_output[i]:nElement()):mul(inrange):add(minval))
   end
   if type(converted_output) ~= 'table' then -- if we wrapped the output, unwrap the gradOutput before feeding it into updateGradInput, since the wrapping is not expected by the module
      go = go[1]
   end
   
   module:zeroGradParameters()
   module:updateGradInput(input,go)
   module:accGradParameters(input,go)

   local wrapped_gradInput = ((type(module.gradInput) == 'table') and module.gradInput) or {module.gradInput}
   local bo = {}
   for i = 1,#wrapped_gradInput do
      bo[i] = wrapped_gradInput[i]:clone()
   end

   -- write module
   local f = torch.DiskFile('tmp.bin','w'):binary()
   f:writeObject(module)
   f:close()
   -- read module
   local m = torch.DiskFile('tmp.bin'):binary():readObject()
   m:forward(input)
   m:zeroGradParameters() -- SHOULDN'T THIS BE m, RATHER THAN module?!?
   m:updateGradInput(input,go)
   m:accGradParameters(input,go)
   -- cleanup
   os.remove('tmp.bin')


   local errf, errb = 0,0
   local converted_output_2 = ((type(m.output) == 'number') and torch.Tensor(1):fill(m.output)) or m.output
   local wrapped_output_2 = ((type(converted_output_2) == 'table') and converted_output_2) or {converted_output_2}
   local fo2 = {}
   for i = 1,#wrapped_output_2 do
      fo2[i] = wrapped_output_2[i]:clone()
      errf = math.max(errf, (fo[i] - fo2[i]):abs():max())
   end
   
   local wrapped_gradInput_2 = ((type(m.gradInput) == 'table') and m.gradInput) or {m.gradInput}
   local bo2 = {}
   for i = 1,#wrapped_gradInput_2 do
      bo2[i] = wrapped_gradInput_2[i]:clone()
      errb = math.max(errb, (bo[i] - bo2[i]):abs():max())
   end

   return errf, errb
end



function nn.Jacobian.testAllUpdate(module, input, weight, gradWeight)
   local gradOutput
   local lr = torch.uniform(0.1, 1)
   local errors = {}

   -- accGradParameters
   local maccgp = module:clone()
   local weightc = maccgp[weight]:clone()
   maccgp:forward(input)
   gradOutput = torch.rand(maccgp.output:size())
   maccgp:zeroGradParameters()
   maccgp:updateGradInput(input, gradOutput)
   maccgp:accGradParameters(input, gradOutput)
   maccgp:updateParameters(lr)
   errors["accGradParameters"] = (weightc-maccgp[gradWeight]*lr-maccgp[weight]):norm()
   
   -- accUpdateGradParameters
   local maccugp = module:clone()
   maccugp:forward(input)
   maccugp:updateGradInput(input, gradOutput)
   maccugp:accUpdateGradParameters(input, gradOutput, lr)
   errors["accUpdateGradParameters"] = (maccugp[weight]-maccgp[weight]):norm()

   -- shared, accGradParameters
   local macsh1 = module:clone()
   local macsh2 = module:clone()
   macsh2:share(macsh1, weight)
   macsh1:forward(input)
   macsh2:forward(input)
   macsh1:zeroGradParameters()
   macsh2:zeroGradParameters()
   macsh1:updateGradInput(input, gradOutput)
   macsh2:updateGradInput(input, gradOutput)
   macsh1:accGradParameters(input, gradOutput)
   macsh2:accGradParameters(input, gradOutput)
   macsh1:updateParameters(lr)
   macsh2:updateParameters(lr)
   local err = (weightc-maccgp[gradWeight]*(lr*2)-macsh1[weight]):norm()
   err = err + (weightc-maccgp[gradWeight]*(lr*2)-macsh2[weight]):norm()
   errors["accGradParameters [shared]"] = err
   
   -- shared, accUpdateGradParameters
   local macshu1 = module:clone()
   local macshu2 = module:clone()
   macshu2:share(macshu1, weight)
   macshu1:forward(input)
   macshu2:forward(input)
   macshu1:updateGradInput(input, gradOutput)
   macshu2:updateGradInput(input, gradOutput)
   macshu1:accUpdateGradParameters(input, gradOutput, lr)
   macshu2:accUpdateGradParameters(input, gradOutput, lr)
   local err = (weightc-maccgp[gradWeight]*(lr*2)-macshu1[weight]):norm()
   err = err + (weightc-maccgp[gradWeight]*(lr*2)-macshu2[weight]):norm()
   errors["accUpdateGradParameters [shared]"] = err

   return errors
end
