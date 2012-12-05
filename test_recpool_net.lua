require 'torch'
require 'nn'
dofile('init.lua')
dofile('build_recpool_net.lua')

local mytester = torch.Tester()
local jac

local precision = 1e-5
local expprecision = 1e-4

local rec_pool_test = {}
local run_test = {}
local other_tests = {}

function rec_pool_test.Square()
   local in1 = torch.rand(10,20)
   local module = nn.Square()
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Square()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.Square1D()
   local in1 = torch.rand(10)
   local module = nn.Square()
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local input = torch.Tensor(ini):zero()

   local module = nn.Square()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.Sqrt()
   local in1 = torch.rand(10,20)
   local module = nn.Sqrt()
   local out = module:forward(in1)
   local err = out:dist(in1:sqrt())
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Sqrt()

   local err = jac.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input, 0, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

--[[
function rec_pool_test.DebugSquareSquare()
   local in1 = torch.rand(10,20)
   local module = nn.DebugSquare('square')
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   --local inj = math.random(5,10)
   --local ink = math.random(5,10)
   local input = torch.Tensor(ini):zero()

   local module = nn.DebugSquare('square')

   local err = jac.testJacobian(module, input, -2, 2)
   mytester:assertlt(err, precision, 'error on state ')

   --local ferr, berr = jac.testIO(module, input, -2, 2)
   --mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   --mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.DebugSquareSqrt()
   local in1 = torch.rand(10,20)
   local module = nn.DebugSquare('sqrt')
   local out = module:forward(in1)
   local err = out:dist(in1:sqrt())
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   --local inj = math.random(5,10)
   --local ink = math.random(5,10)
   local input = torch.Tensor(ini):zero()

   local module = nn.DebugSquare('sqrt')

   local err = jac.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   --local ferr, berr = jac.testIO(module, input, 0.1, 2)
   --mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   --mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end
--]]


function rec_pool_test.ConstrainedLinearLinearWeightMatrix()
   local ini = math.random(50,70)
   local inj = math.random(50,70)
   local ink = math.random(10,20)
   local input = torch.Tensor(ink,ini):zero()
   local module = nn.ConstrainedLinear(ini, inj, {no_bias = true, normalized_columns = true, squared_weight_matrix = false}, true, 1, true) 

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err,precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err,precision, 'error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.ConstrainedLinearSquaredWeightMatrix()
   local ini = math.random(50,70)
   local inj = math.random(50,70)
   local ink = math.random(10,20)
   local input = torch.Tensor(ink,ini):zero()
   local module = nn.ConstrainedLinear(ini, inj, {no_bias = true, normalized_columns = true, squared_weight_matrix = true}, true, 1, true) 

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err,precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err,precision, 'error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end



function rec_pool_test.LogSoftMax()
   local ini = math.random(10,20)
   --local inj = math.random(10,20)
   local input = torch.Tensor(ini):zero()
   local module = nn.LogSoftMax()

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, expprecision, 'error on state ') -- THIS REQUIRES LESS PRECISION THAN NORMAL, presumably because the exponential tends to make backpropagation unstable

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.FixedShrink()
   local ini = math.random(10,20)
   --local inj = math.random(10,20)
   local input = torch.Tensor(ini):zero()
   local module = nn.FixedShrink(ini)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ') 

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end



function rec_pool_test.AddConstant()
   local input = torch.rand(10,20)
   local random_addend = math.random()
   local module = nn.AddConstant(input:size(), random_addend)
   local out = module:forward(input)
   local err = out:dist(input:add(random_addend))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.AddConstant(input:size(), random_addend)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.MulConstant()
   local input = torch.rand(10,20)
   local random_addend = math.random()
   local module = nn.MulConstant(input:size(), random_addend)
   local out = module:forward(input)
   local err = out:dist(input:mul(random_addend))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.MulConstant(input:size(), random_addend)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function create_parameterized_shrink_test(require_nonnegative_units)
   local function this_parameterized_shrink_test()
      local size = math.random(10,20)
      local module = nn.ParameterizedShrink(size, require_nonnegative_units, true) -- first, try with units that can be negative; ignore nonnegativity constraint on shrink values
      local shrink_vals = torch.rand(size)
      module:reset(shrink_vals)
      local input = torch.Tensor(size):zero()
      
      local err = jac.testJacobian(module,input)
      mytester:assertlt(err,precision, 'error on state ')
      local err = jac.testJacobianParameters(module, input, module.shrink_val, module.grad_shrink_val)
      mytester:assertlt(err,precision, 'error on shrink val ')
   
      local err = jac.testJacobianUpdateParameters(module, input, module.shrink_val)
      mytester:assertlt(err,precision, 'error on shrink val [direct update]')
      
      for t,err in pairs(jac.testAllUpdate(module, input, 'shrink_val', 'grad_shrink_val')) do
	 mytester:assertlt(err, precision, string.format(
			      'error on shrink val (testAllUpdate) [%s]', t))
      end
      
      local ferr,berr = jac.testIO(module,input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
      mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
   end

   return this_parameterized_shrink_test
end

rec_pool_test.ParameterizedShrinkNonnegative = create_parameterized_shrink_test(true)
rec_pool_test.ParameterizedShrinkUnconstrained = create_parameterized_shrink_test(false)

function rec_pool_test.ParameterizedL1Cost()
   local size = math.random(10,20)
   local initial_lambda = math.random()
   local desired_criterion_value = math.random()
   local learning_rate_scaling_factor = -1 -- necessary since normally the error is maximized with respect to the lagrange multipliers
   local module = nn.ParameterizedL1Cost(size, initial_lambda, desired_criterion_value, learning_rate_scaling_factor, 'full test')
   
   local input = torch.Tensor(size):zero()
   
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')
   
   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update]')
   
   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format('error on weight (testAllUpdate) [%s]', t))
   end
   
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.CAddTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.CAddTable()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.CosineDistance()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.CosineDistance()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.PairwiseDistance()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.PairwiseDistance(2)

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.CMulTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.CMulTable()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.CDivTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.CDivTable()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.SafePower()
   local in1 = torch.rand(10,20)
   local module = nn.SafePower(2)
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local pw = torch.uniform()*math.random(1,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.SafePower(pw)

   local err = nn.Jacobian.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module,input, 0.1, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.SafeCMulTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.SafeCMulTable()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.NormalizeTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(inj,ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.NormalizeTable()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.NormalizeTensor1D()
   local ini = math.random(10,20)
   local input = torch.Tensor(ini):zero()
   local module = nn.NormalizeTensor()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state (non-table) ')

   --local ferr,berr = jac.testIOTable(module,input)
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.NormalizeTensor2D()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local input = torch.Tensor(inj,ini):zero()
   local module = nn.NormalizeTensor()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state (non-table) ')

   --local ferr,berr = jac.testIOTable(module,input)
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end



function rec_pool_test.L2Cost()
   print(' testing L2Cost!!!')
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.L2Cost(math.random(), 2)

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (2 inputs) ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (2 inputs) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (2 inputs) - i/o backward err ')

   input = torch.Tensor(ini):zero()
   module = nn.L2Cost(math.random(), 1)

   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (1 input) ')

   ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (1 input) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (1 input) - i/o backward err ')
end

function rec_pool_test.SoftClassNLLCriterion()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local input = torch.Tensor(ini, inj):zero()
   local module = nn.L1CriterionModule(nn.SoftClassNLLCriterion(), 1)
   local target = torch.Tensor(ini)
   for i = 1,ini do
      target[i] = math.random(input:size(2))
   end
   module:setTarget(target)

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (2 inputs) ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (2 inputs) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (2 inputs) - i/o backward err ')

   input = torch.Tensor(ini):zero()
   module = nn.L1CriterionModule(nn.SoftClassNLLCriterion(), 1)
   module:setTarget(math.random(ini))

   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (1 input) ')

   ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (1 input) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (1 input) - i/o backward err ')
end

function rec_pool_test.HingeClassNLLCriterion()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local input = torch.Tensor(ini, inj):zero()
   local module = nn.L1CriterionModule(nn.HingeClassNLLCriterion(), 1)
   local target = torch.Tensor(ini)
   for i = 1,ini do
      target[i] = math.random(input:size(2))
   end
   module:setTarget(target)

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (2 inputs) ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (2 inputs) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (2 inputs) - i/o backward err ')

   input = torch.Tensor(ini):zero()
   module = nn.L1CriterionModule(nn.HingeClassNLLCriterion(), 1)
   module:setTarget(math.random(ini))

   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state (1 input) ')

   ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (1 input) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (1 input) - i/o backward err ')
end


function rec_pool_test.CauchyCost()
   print(' testing CauchyCost!!!')
   local ini = math.random(10,20)
   --local inj = math.random(10,20)
   --local ink = math.random(10,20)
   local input = torch.Tensor(ini):zero()
   local module = nn.CauchyCost(math.random())

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' (2 inputs) - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' (2 inputs) - i/o backward err ')
end


function rec_pool_test.ParallelIdentity()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.ParallelTable()
   module:add(nn.Identity())
   module:add(nn.Identity())

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function rec_pool_test.CopyTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local table_size = math.random(5,10)
   local input = torch.Tensor(ini):zero()

   -- test CopyTable on a tensor (non-table) input
   local module = nn.Sequential()
   module:add(nn.IdentityTable())
   module:add(nn.CopyTable(1, math.random(1,5)))

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   
   -- test CopyTable on an a table input with many entries
   input = {}
   for i=1,table_size do
      input[i] = torch.Tensor(ini):zero()
   end
   module = nn.CopyTable(math.random(1,table_size), math.random(1,5))

   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end

function rec_pool_test.IdentityTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local table_size = math.random(5,10)
   local input = torch.Tensor(ini):zero()
   local module = nn.IdentityTable()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function rec_pool_test.SelectTable()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local table_size = math.random(5,10)
   local input = {}
   for i=1,table_size do
      input[i] = torch.Tensor(ini):zero()
   end

   -- try using SelectTable to pass through the inputs unchanged
   local module = nn.ParallelDistributingTable()
   for i=1,table_size do
      module:add(nn.SelectTable{i})
   end
   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- try using SelectTable to permute the inputs
   module = nn.ParallelDistributingTable()
   for i=1,table_size-1 do
      module:add(nn.SelectTable{i+1})
   end
   module:add(nn.SelectTable{1})
   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')


   -- try using SelectTable to throw away all but one input
   module = nn.ParallelDistributingTable()
   module:add(nn.SelectTable{math.random(table_size)})
   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')


   -- try using SelectTable to feed into CAddTables
   module = nn.ParallelDistributingTable()
   local current_module
   for i=1,table_size-1 do
      current_module = nn.Sequential()
      current_module:add(nn.SelectTable{i, i+1})
      current_module:add(nn.CAddTable())
      module:add(current_module)
   end

   err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')
end


function rec_pool_test.L1CriterionModule()
   local ini = math.random(10,20)
   local lambda = math.random()
   local input = torch.Tensor(ini):zero()
   local module = nn.L1CriterionModule(nn.L1Cost(), lambda)

   print('L1CriterionModule test')

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


local function test_module(name, module, param, grad_param, parameter_list, model, input, jac, precision, min_param, max_param)
   local function copy_table(t)
      local t2 = {}
      for k,v in pairs(t) do
	 t2[k] = v
      end
      return t2
   end
   local unused_params

   print(name .. ' ' .. param)
   local err = jac.testJacobianParameters(model, input, module[param], module[grad_param], min_param, max_param)
   mytester:assertlt(err,precision, 'error on ' .. name .. ' ' .. param)
   unused_params = copy_table(parameter_list)
   for k,v in pairs(parameter_list) do
      if module[param] == v then
	 --print('removing key ' .. k .. ' for ' .. name .. '.' .. param)
	 table.remove(unused_params, k)
      end
   end
   --table.remove(unused_params, param_number)
   local err = jac.testJacobianUpdateParameters(model, input, module[param], unused_params, min_param, max_param)
   mytester:assertlt(err,precision, 'error on ' .. name .. ' ' .. param .. ' [full processing chain, direct update] ')
end


function rec_pool_test.full_network_test()
   --REMEMBER that all Jacobian tests randomly reset the parameters of the module being tested, and then return them to their original value after the test is completed.  If gradients explode for only one module, it is likely that this random initialization is incorrect.  In particular, the signals passing through the explaining_away matrix will explode if it has eigenvalues with magnitude greater than one.  The acceptable scale of the random initialization will decrease as the explaining_away matrix increases, so be careful when changing layer_size.

   -- recpool_config_prefs are num_ista_iterations, shrink_style, disable_pooling, use_squared_weight_matrix, normalize_each_layer, repair_interval
   local recpool_config_prefs = {}
   recpool_config_prefs.num_ista_iterations = 5
   --recpool_config_prefs.shrink_style = 'ParameterizedShrink'
   recpool_config_prefs.shrink_style = 'FixedShrink' --'ParameterizedShrink'
   --recpool_config_prefs.shrink_style = 'SoftPlus'
   recpool_config_prefs.disable_pooling = false
   if recpool_config_prefs.disable_pooling then
      print('POOLING IS DISABLED!!!')
      io.read()
   end
   recpool_config_prefs.use_squared_weight_matrix = true
   recpool_config_prefs.normalize_each_layer = false
   recpool_config_prefs.repair_interval = 1


   --local layer_size = {math.random(10,20), math.random(10,20), math.random(5,10), math.random(5,10)} 
   --local layer_size = {math.random(10,20), math.random(10,20), math.random(5,10), math.random(10,20), math.random(5,10), math.random(5,10)} 
   --local layer_size = {math.random(10,20), math.random(10,20), math.random(5,10), math.random(10,20), math.random(5,10), math.random(10,20), math.random(5,10), math.random(5,10)} 
   local minibatch_size = 0
   local layer_size = {10, 20, 10, 10}
   local target
   if minibatch_size > 0 then
      target = torch.Tensor(minibatch_size)
      for i = 1,minibatch_size do
	 target[i] = math.random(layer_size[#layer_size])
      end
   else
      target = math.random(layer_size[#layer_size])
   end
   --local target = torch.zeros(layer_size[#layer_size]) -- DEBUG ONLY!!! FOR THE LOVE OF GOD!!!
   --target[math.random(layer_size[#layer_size])] = 1

   local lambdas = {ista_L2_reconstruction_lambda = math.random(), 
		    ista_L1_lambda = math.random(), 
		    pooling_L2_shrink_reconstruction_lambda = math.random(), 
		    pooling_L2_orig_reconstruction_lambda = math.random(), 
		    pooling_L2_shrink_position_unit_lambda = math.random(), 
		    pooling_L2_orig_position_unit_lambda = math.random(), 
		    pooling_output_cauchy_lambda = math.random(), 
		    pooling_mask_cauchy_lambda = math.random()}

   local lagrange_multiplier_targets = {feature_extraction_target = math.random(), pooling_target = math.random(), mask_target = math.random()}
   local lagrange_multiplier_learning_rate_scaling_factors = {feature_extraction_scaling_factor = -1, pooling_scaling_factor = -1, mask_scaling_factor = -1}

   ---[[
   local layered_lambdas = {lambdas} --{lambdas, lambdas}
   local layered_lagrange_multiplier_targets = {lagrange_multiplier_targets} --{lagrange_multiplier_targets, lagrange_multiplier_targets}
   local layered_lagrange_multiplier_learning_rate_scaling_factors = {lagrange_multiplier_learning_rate_scaling_factors} --{lagrange_multiplier_learning_rate_scaling_factors, lagrange_multiplier_learning_rate_scaling_factors}
   --]]

   --[[
   local layered_lambdas = {lambdas, lambdas, lambdas}
   local layered_lagrange_multiplier_targets = {lagrange_multiplier_targets, lagrange_multiplier_targets, lagrange_multiplier_targets}
   local layered_lagrange_multiplier_learning_rate_scaling_factors = {lagrange_multiplier_learning_rate_scaling_factors, lagrange_multiplier_learning_rate_scaling_factors, lagrange_multiplier_learning_rate_scaling_factors}
   --]]
   
   local model =
      build_recpool_net(layer_size, layered_lambdas, 1, layered_lagrange_multiplier_targets, layered_lagrange_multiplier_learning_rate_scaling_factors, recpool_config_prefs, nil, true) -- final true -> NORMALIZATION IS DISABLED!!!
   print('finished building recpool net')

   -- create a list of all the parameters of all modules, so they can be held constant when doing Jacobian tests
   local parameter_list = {}
   for i = 1,#model.layers do
      for k,v in pairs(model.layers[i].module_list) do
	 if v.parameters and v:parameters() then -- if a parameters function is defined
	    local params = v:parameters()
	    for j = 1,#params do
	       table.insert(parameter_list, params[j])
	    end
	 end
      end
   end
   
   for k,v in pairs(model.module_list) do
      if v.parameters and v:parameters() then -- if a parameters function is defined
	 local params = v:parameters()
	 for j = 1,#params do
	    table.insert(parameter_list, params[j])
	 end
      end
   end

   model:set_target(target)
   
   print('Since the model contains a LogSoftMax, use precision ' .. expprecision .. ' rather than ' .. precision)
   local precision = 3*expprecision
   
   local input
   if minibatch_size > 0 then
      input = torch.Tensor(minibatch_size, layer_size[1]):zero()
   else
      input = torch.Tensor(layer_size[1]):zero()
   end

   -- check that we don't always produce nans
   local check_for_non_nans = false
   if check_for_non_nans then
      local test_input = torch.rand(layer_size[1])
      model:updateOutput(test_input)
      print('check that we do not always produce nans')
      print(test_input)
      print(model.module_list.classification_dictionary.output)
      io.read()
   end


   local err = jac.testJacobian(model, input)
   mytester:assertlt(err,precision, 'error on processing chain state ')

   for i = 1,#model.layers do
      test_module('decoding dictionary weight', model.layers[i].module_list.decoding_feature_extraction_dictionary, 'weight', 'gradWeight', parameter_list, model, input, jac, precision)
      test_module('decoding dictionary bias', model.layers[i].module_list.decoding_feature_extraction_dictionary, 'bias', 'gradBias', parameter_list, model, input, jac, precision)

      -- problem here!!!
      test_module('encoding dictionary weight', model.layers[i].module_list.encoding_feature_extraction_dictionary, 'weight', 'gradWeight', parameter_list, model, input, jac, precision)
      test_module('encoding dictionary bias', model.layers[i].module_list.encoding_feature_extraction_dictionary, 'bias', 'gradBias', parameter_list, model, input, jac, precision)

      -- don't allow large weights, or the messages exhibit exponential growth
      test_module('explaining away weight', model.layers[i].module_list.explaining_away, 'weight', 'gradWeight', parameter_list, model, input, jac, precision, -0.6, 0.6)
      test_module('explaining away bias', model.layers[i].module_list.explaining_away, 'bias', 'gradBias', parameter_list, model, input, jac, precision)
      if shrink_style == 'ParameterizedShrink' then
	 test_module('shrink shrink_val', model.layers[i].module_list.shrink, 'shrink_val', 'grad_shrink_val', parameter_list, model, input, jac, precision)
      end
      -- element 8 of the parameter_list is negative_shrink_val

      if not(disable_pooling) then
	 test_module('decoding pooling dictionary weight', model.layers[i].module_list.decoding_pooling_dictionary, 'weight', 'gradWeight', parameter_list, model, input, jac, precision, 0, 2)
	 test_module('decoding pooling dictionary bias', model.layers[i].module_list.decoding_pooling_dictionary, 'bias', 'gradBias', parameter_list, model, input, jac, precision, 0, 2)
	 
	 -- make sure that the random weights assigned to the encoding pooling dictionary for Jacobian testing are non-negative!
	 test_module('encoding pooling dictionary weight', model.layers[i].module_list.encoding_pooling_dictionary, 'weight', 'gradWeight', parameter_list, model, input, jac, precision, 0, 2)
	 test_module('encoding pooling dictionary bias', model.layers[i].module_list.encoding_pooling_dictionary, 'bias', 'gradBias', parameter_list, model, input, jac, precision, 0, 2)
      end      

      if model.layers[i].module_list.feature_extraction_sparsifying_module.weight then
	 test_module('feature extraction sparsifying module', model.layers[i].module_list.feature_extraction_sparsifying_module, 'weight', 'gradWeight', parameter_list, model, input, jac, precision)
      end
      if model.layers[i].module_list.pooling_sparsifying_module.weight then
	 test_module('pooling sparsifying module', model.layers[i].module_list.pooling_sparsifying_module, 'weight', 'gradWeight', parameter_list, model, input, jac, precision)
      end
      if model.layers[i].module_list.mask_sparsifying_module.weight then
	 test_module('mask sparsifying module', model.layers[i].module_list.mask_sparsifying_module, 'weight', 'gradWeight', parameter_list, model, input, jac, precision)
      end
   end
   
   test_module('classification dictionary weight', model.module_list.classification_dictionary, 'weight', 'gradWeight', parameter_list, model, input, jac, precision)
   test_module('classification dictionary bias', model.module_list.classification_dictionary, 'bias', 'gradBias', parameter_list, model, input, jac, precision)


end

function rec_pool_test.ISTA_reconstruction()
   -- check that ISTA actually finds a sparse reconstruction.  decoding_dictionary.output should be similar to test_input, and shrink_copies[#shrink_copies].output should have some zeros

   -- recpool_config_prefs are num_ista_iterations, shrink_style, disable_pooling, use_squared_weight_matrix, normalize_each_layer, repair_interval
   local recpool_config_prefs = {}
   recpool_config_prefs.num_ista_iterations = 50
   --recpool_config_prefs.shrink_style = 'ParameterizedShrink'
   recpool_config_prefs.shrink_style = 'FixedShrink' --'ParameterizedShrink'
   --recpool_config_prefs.shrink_style = 'SoftPlus'
   recpool_config_prefs.disable_pooling = false
   recpool_config_prefs.use_squared_weight_matrix = true
   recpool_config_prefs.normalize_each_layer = false
   recpool_config_prefs.repair_interval = 1


   local minibatch_size = 0
   local layer_size = {10, 60, 10, 10}
   local target
   if minibatch_size > 0 then
      target = torch.Tensor(minibatch_size)
      for i = 1,minibatch_size do
	 target[i] = math.random(layer_size[#layer_size])
      end
   else
      target = math.random(layer_size[#layer_size])
   end


   local lambdas = {ista_L2_reconstruction_lambda = math.random(), 
		    ista_L1_lambda = math.random(), 
		    pooling_L2_shrink_reconstruction_lambda = math.random(), 
		    pooling_L2_orig_reconstruction_lambda = math.random(), 
		    pooling_L2_shrink_position_unit_lambda = math.random(), 
		    pooling_L2_orig_position_unit_lambda = math.random(), 
		    pooling_output_cauchy_lambda = math.random(), 
		    pooling_mask_cauchy_lambda = math.random()}
   local lagrange_multiplier_targets = {feature_extraction_target = math.random(), pooling_target = math.random(), mask_target = math.random()}
   local lagrange_multiplier_learning_rate_scaling_factors = {feature_extraction_scaling_factor = -1, pooling_scaling_factor = -1, mask_scaling_factor = -1}

   local layered_lambdas = {lambdas}
   local layered_lagrange_multiplier_targets = {lagrange_multiplier_targets}
   local layered_lagrange_multiplier_learning_rate_scaling_factors = {lagrange_multiplier_learning_rate_scaling_factors}

   -- create the dataset so the features of the network can be initialized
   local data = nil
   --require 'mnist'
   --local data = mnist.loadTrainSet(500, 'recpool_net') -- 'recpool_net' option ensures that the returned table contains elements data and labels, for which the __index method is overloaded.  

   --Indexing labels returns an index, rather than a tensor
   --data:normalizeL2() -- normalize each example to have L2 norm equal to 1


   local model =
      build_recpool_net(layer_size, layered_lambdas, 1, layered_lagrange_multiplier_targets, layered_lagrange_multiplier_learning_rate_scaling_factors, recpool_config_prefs, nil) -- final true -> NORMALIZATION IS DISABLED!!!

   -- convenience names for easy access
   local shrink_copies = model.layers[1].module_list.shrink_copies
   local shrink = model.layers[1].module_list.shrink
   local explaining_away_copies = model.layers[1].module_list.explaining_away_copies
   local explaining_away = model.layers[1].module_list.explaining_away
   local decoding_feature_extraction_dictionary = model.layers[1].module_list.decoding_feature_extraction_dictionary

   local test_input 
   if minibatch_size > 0 then
      test_input = torch.rand(minibatch_size, layer_size[1]) --torch.Tensor(minibatch_size, layer_size[1]):zero()
   else
      test_input = torch.rand(layer_size[1])
   end

   model:set_target(target)

   model:updateOutput(test_input)
   print('test input', test_input)
   print('reconstructed input', model.layers[1].module_list.decoding_feature_extraction_dictionary.output)
   print('shrink output', shrink_copies[#shrink_copies].output)

   local test_gradInput = torch.zeros(model.output:size())
   model:updateGradInput(test_input, test_gradInput)


   if shrink_style == 'ParameterizedShrink' then
      -- confirm that parameter sharing is working properly
      for i = 1,#shrink_copies do
	 if (shrink_copies[i].shrink_val:storage() ~= shrink.shrink_val:storage()) or (shrink_copies[i].grad_shrink_val:storage() ~= shrink.grad_shrink_val:storage()) then
	    print('ERROR!!!  shrink_copies[' .. i .. '] does not share parameters with base shrink!!!')
	    io.read()
	 end
	 --print('shrink_copies[' .. i .. '] gradInput', shrink_copies[i].gradInput)
	 --print('shrink_copies[' .. i .. '] output', shrink_copies[i].output)
      end
   end
   
   for i = 1,#explaining_away_copies do
      if (explaining_away_copies[i].weight:storage() ~= explaining_away.weight:storage()) or (explaining_away_copies[i].bias:storage() ~= explaining_away.bias:storage()) then
	 print('ERROR!!!  explaining_away_copies[' .. i .. '] does not share parameters with base explaining_away!!!')
	 io.read()
      end
      --print('explaining_away_copies[' .. i .. '] gradInput', explaining_away_copies[i].gradInput)
      --print('explaining_away_copies[' .. i .. '] output', explaining_away_copies[i].output)
   end


   ---[[
   if minibatch_size == 0 then
      local shrink_output_tensor = torch.Tensor(decoding_feature_extraction_dictionary.output:size(1), #shrink_copies)
      for i = 1,#shrink_copies do
	 shrink_output_tensor:select(2,i):copy(decoding_feature_extraction_dictionary:updateOutput(shrink_copies[i].output))
      end
      print(shrink_output_tensor)
   else
      local index_list = {1, 2, 3, 4, 5, 6, 7}
      local num_shrink_output_tensor_elements = #index_list -- model.layers[1].module_list.shrink.output:size(1)
      local shrink_output_tensor = torch.Tensor(num_shrink_output_tensor_elements, 1 + #model.layers[1].module_list.shrink_copies)
      
      for j = 1,#index_list do
	 shrink_output_tensor[{j, 1}] = model.layers[1].module_list.shrink.output[{1, index_list[j]}] -- minibatch_size >= 1, so we need to select the minibatch from which to draw the state
      end
      
      for i = 1,#model.layers[1].module_list.shrink_copies do
	 --shrink_output_tensor:select(2,i):copy(model.layers[1].module_list.shrink_copies[i].output)
	 for j = 1,#index_list do
	    shrink_output_tensor[{j, i+1}] = model.layers[1].module_list.shrink_copies[i].output[{1, index_list[j]}] -- minibatch_size >= 1, so we need to select the minibatch from which to draw the state
	 end
      end

      print('full evolution of index_list for the first element of the minibatch', shrink_output_tensor)
   end
   
   --]]

end







--local num_tests = 0 
--for name in pairs(rec_pool_test) do num_tests = num_tests + 1; print('test ' .. num_tests .. ' is ' .. name) end
--print('number of tests: ', num_tests)
math.randomseed(os.clock())

mytester:add(rec_pool_test)
--mytester:add(run_test)
--mytester:add(other_tests)

jac = nn.Jacobian
mytester:run()

