----------------------------------------------------------------------
-- A plain implementation of SGD
--
-- ARGS:
-- opfunc : a function that takes a single input (X), the point of 
--          evaluation, and returns f(X) and df/dX
-- x      : the initial point
-- state  : a table describing the state of the optimizer; after each
--          call the state is modified
--   state.learningRate      : learning rate
--   state.learningRateDecay : learning rate decay
--   state.weightDecay       : weight decay
--   state.L1weightDecay     : L1 weight decay
--   state.momentum          : momentum
--   state.learningRates     : vector of individual learning rates
--
-- RETURN:
-- x     : the new x vector
-- f(x)  : the function, evaluated before the update
--
function optim.sgd_decayed_weight_decay(opfunc, x, state)
   -- (0) get/update state
   local state = state or {}
   local lr = state.learningRate or 1e-3
   local lrd = state.learningRateDecay or 0
   local wd = state.weightDecay or 0
   local mom = state.momentum or 0
   local lrs = state.learningRates
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   state.sign_tensor = state.sign_tensor or torch.Tensor()

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (2) apply momentum
   if mom ~= 0 then
      if not state.dfdx then
         state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
      else
         state.dfdx:mul(mom):add(1-mom, dfdx)
      end
      dfdx = state.dfdx
   end

   -- (2) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)

   -- (3) weight decay -- weight decay is applied *AFTER* learning rate decay, as opposed to before as in the standard optim.sgd implementation
   if wd ~= 0 then
      x:add(-wd*clr, x)
   end

   -- (3.5) L1 weight decay 
   if state.L1weightDecay ~= 0 then
      state.sign_tensor:resizeAs(x)
      state.sign_tensor:sign(x)
      x:add(-clr * state.L1weightDecay, state.sign_tensor)
   end
      
   -- (4) parameter update with single or individual learning rates
   if lrs then
      if not state.deltaParameters then
         state.deltaParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.deltaParameters:copy(lrs):cmul(dfdx)
      x:add(-clr, state.deltaParameters)
   else
      x:add(-clr, dfdx)
   end

   -- (5) update evaluation counter
   if state.learningRateDecay > 0 then -- added by Jason 10/26/12; only counter iterations towards learning rate decay when learning rate decay is active
      state.evalCounter = state.evalCounter + 1
   end

   -- return x*, f(x) before optimization
   return x,{fx}
end
