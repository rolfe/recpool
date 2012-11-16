----------------------------------------------------------------------
-- An implementation of ASGD
--
-- ASGD: 
--     x := x - eta_t df/dx(z,x)
--     The original code had x := (1 - lambda eta_t) x - eta_t df/dx(z,x), but this incorrectly forces weight decay on the system, which is not appropriate for non-SVMs
--     a := a + mu_t [ x - a ]
--
--  eta_t = eta0 / (1 + lambda eta0 t) ^ 0.75
--   mu_t = 1/max(1,t-t0)
-- 
-- implements ASGD algoritm as in L.Bottou's sgd-2.0
--
-- ARGS:
-- opfunc : a function that takes a single input (X), the point of 
--          evaluation, and returns f(X) and df/dX
-- x      : the initial point
-- state  : a table describing the state of the optimizer; after each
--          call the state is modified
--   state.eta0              : learning rate
--   state.lambda            : decay term
--   state.alpha             : power for eta update
--   state.t0                : point at which to start averaging
--
-- RETURN:
-- x     : the new x vector
-- f(x)  : the function, evaluated before the update
-- ax    : the averaged x vector
--
function optim.asgd_no_weight_decay(opfunc, x, state)
   -- (0) get/update state
   local state = state or {}
   state.eta0 = state.eta0 or 1e-4
   state.lambda = state.lambda or 1e-4
   state.alpha = state.alpha or 0.75
   state.t0 = state.t0 or 1e6

   -- (hidden state)
   state.eta_t = state.eta_t or state.eta0
   state.mu_t = state.mu_t or 1
   state.t = state.t or 0
   state.evalCounter = state.t -- used to maintain compatibility with SGD
   state.learningRateDecay = state.lambda * state.eta0

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (2) decay term - NOT REQUIRED OR DESIRED!!!  This is just weight decay
   --x:mul(1 - state.lambda*state.eta_t)

   -- (3) update x
   x:add(-state.eta_t, dfdx)

   -- (4) averaging
   state.ax = state.ax or torch.Tensor():typeAs(x):resizeAs(x):zero()
   state.tmp = state.tmp or torch.Tensor():typeAs(state.ax):resizeAs(state.ax)
   if state.mu_t ~= 1 then
      state.tmp:copy(x)
      state.tmp:add(-1,state.ax):mul(state.mu_t)
      state.ax:add(state.tmp)
   else
      state.ax:copy(x)
   end

   -- (5) update eta_t and mu_t
   state.t = state.t + 1
   state.evalCounter = state.t -- used to maintain compatibility with SGD
   state.eta_t = state.eta0 / math.pow((1 + state.lambda * state.eta0 * state.t), state.alpha)
   state.mu_t = 1 / math.max(1, state.t - state.t0)

   -- return x*, f(x) before optimization, and average(x_t0,x_t1,x_t2,...)
   return x,{fx},state.ax
end
