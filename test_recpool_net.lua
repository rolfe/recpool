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



function rec_pool_test.full_network_test()
   --REMEMBER that all Jacobian tests randomly reset the parameters of the module being tested, and then return them to their original value after the test is completed.  If gradients explode for only one module, it is likely that this random initialization is incorrect.  In particular, the signals passing through the explaining_away matrix will explode if it has eigenvalues with magnitude greater than one.  The acceptable scale of the random initialization will decrease as the explaining_away matrix increases, so be careful when changing layer_size.

   local layer_size = {math.random(10,20), math.random(10,20), math.random(10,20), math.random(10,20)} 
   --local layer_size = {10, 20, 10, 10}
   local target = math.random(layer_size[4])
   local lambdas = {ista_L2_reconstruction_lambda = math.random(), 
		    ista_L1_lambda = math.random(), 
		    pooling_L2_reconstruction_lambda = math.random(), 
		    pooling_L2_position_unit_lambda = math.random(), 
		    pooling_output_cauchy_lambda = math.random(), 
		    pooling_mask_cauchy_lambda = math.random()}
   local model, criteria_list, encoding_dictionary, decoding_dictionary, encoding_pooling_dictionary, decoding_pooling_dictionary, classification_dictionary, explaining_away, shrink, explaining_away_copies, shrink_copies = 
      build_recpool_net(layer_size, lambdas, 5, true) -- NORMALIZATION IS DISABLED!!!

   local parameter_list = {decoding_dictionary.weight, decoding_dictionary.bias, encoding_dictionary.weight, encoding_dictionary.bias, explaining_away.weight, explaining_away.bias, shrink.shrink_val, shrink.negative_shrink_val, decoding_pooling_dictionary.weight, decoding_pooling_dictionary.bias, encoding_pooling_dictionary.weight, encoding_pooling_dictionary.bias, classification_dictionary.weight, classification_dictionary.bias}

   model:set_target(target)
   
   print('Since the model contains a LogSoftMax, use precision ' .. expprecision .. ' rather than ' .. precision)
   local precision = 3*expprecision

   local function copy_table(t)
      local t2 = {}
      for k,v in pairs(t) do
	 t2[k] = v
      end
      return t2
   end
   local unused_params

   local input = torch.Tensor(layer_size[1]):zero()


   local err = jac.testJacobian(model, input)
   mytester:assertlt(err,precision, 'error on processing chain state ')

   print('decoding dictionary weight')
   local err = jac.testJacobianParameters(model, input, decoding_dictionary.weight, decoding_dictionary.gradWeight)
   mytester:assertlt(err,precision, 'error on decoding dictionary weight ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 1)
   local err = jac.testJacobianUpdateParameters(model, input, decoding_dictionary.weight, unused_params)
   mytester:assertlt(err,precision, 'error on decoding dictionary weight [full processing chain, direct update] ')

   print('decoding dictionary bias')
   local err = jac.testJacobianParameters(model, input, decoding_dictionary.bias, decoding_dictionary.gradBias)
   mytester:assertlt(err,precision, 'error on decoding dictionary bias ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 2)
   local err = jac.testJacobianUpdateParameters(model, input, decoding_dictionary.bias, unused_params)
   mytester:assertlt(err,precision, 'error on decoding dictionary bias [full processing chain, direct update] ')
   
   print('encoding dictionary weight')
   local err = jac.testJacobianParameters(model, input, encoding_dictionary.weight, encoding_dictionary.gradWeight)
   mytester:assertlt(err,precision, 'error on encoding dictionary weight ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 3)
   local err = jac.testJacobianUpdateParameters(model, input, encoding_dictionary.weight, unused_params)
   mytester:assertlt(err,precision, 'error on encoding dictionary weight [full processing chain, direct update] ')
   
   print('encoding dictionary bias')
   local err = jac.testJacobianParameters(model, input, encoding_dictionary.bias, encoding_dictionary.gradBias)
   mytester:assertlt(err,precision, 'error on encoding dictionary bias ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 4)
   local err = jac.testJacobianUpdateParameters(model, input, encoding_dictionary.bias, unused_params)
   mytester:assertlt(err,precision, 'error on encoding dictionary bias [full processing chain, direct update] ')

   -- explaining_away uses shared weights.  We can only test the gradient using testJacobianParameters if we sum the backwards gradient at all shared copies, since the parameter perturbation in forward affects all shared copies
   local explaining_away_gradWeight_array = {}
   for i,ea in ipairs(explaining_away_copies) do
      explaining_away_gradWeight_array[i] = ea.gradWeight
   end
   print('explaining away weight')
   --local err = jac.testJacobianParameters(model, input, explaining_away.weight, explaining_away_gradWeight_array, -0.6, 0.6) -- don't allow large weights, or the messages exhibit exponential growth
   local err = jac.testJacobianParameters(model, input, explaining_away.weight, explaining_away.gradWeight, -0.6, 0.6) -- don't allow large weights, or the messages exhibit exponential growth
   mytester:assertlt(err,precision, 'error on explaining away weight ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 5)
   local err = jac.testJacobianUpdateParameters(model, input, explaining_away.weight, unused_params, -0.6, 0.6) -- don't allow large weights, or the messages exhibit exponential growth
   mytester:assertlt(err,precision, 'error on explaining away weight [full processing chain, direct update] ')

   local explaining_away_gradBias_array = {}
   for i,ea in ipairs(explaining_away_copies) do
      explaining_away_gradBias_array[i] = ea.gradBias
   end
   print('explaining away bias')
   --local err = jac.testJacobianParameters(model, input, explaining_away.bias, explaining_away_gradBias_array)
   local err = jac.testJacobianParameters(model, input, explaining_away.bias, explaining_away.gradBias)
   mytester:assertlt(err,precision, 'error on explaining away bias ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 6)
   local err = jac.testJacobianUpdateParameters(model, input, explaining_away.bias, unused_params)
   mytester:assertlt(err,precision, 'error on explaining away bias [full processing chain, direct update] ')

   local shrink_grad_shrink_val_array = {}
   for i,sh in ipairs(shrink_copies) do
      shrink_grad_shrink_val_array[i] = sh.grad_shrink_val
   end
   print('shrink shrink_val')
   --local err = jac.testJacobianParameters(model, input, shrink.shrink_val, shrink_grad_shrink_val_array)
   local err = jac.testJacobianParameters(model, input, shrink.shrink_val, shrink.grad_shrink_val)
   mytester:assertlt(err,precision, 'error on shrink shrink_val ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 7)
   local err = jac.testJacobianUpdateParameters(model, input, shrink.shrink_val, unused_params)
   mytester:assertlt(err,precision, 'error on shrink shrink_val [full processing chain, direct update] ')

   --[[
   print(encoding_pooling_dictionary.weight)
   print(encoding_pooling_dictionary.bias)
   print(decoding_pooling_dictionary.weight)
   print(decoding_pooling_dictionary.bias)
   --]]

   print('decoding pooling dictionary weight')
   local err = jac.testJacobianParameters(model, input, decoding_pooling_dictionary.weight, decoding_pooling_dictionary.gradWeight)
   mytester:assertlt(err,precision, 'error on decoding pooling dictionary weight ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 9)
   local err = jac.testJacobianUpdateParameters(model, input, decoding_pooling_dictionary.weight, unused_params)
   mytester:assertlt(err,precision, 'error on decoding pooling dictionary weight [full processing chain, direct update] ')

   print('decoding pooling dictionary bias')
   local err = jac.testJacobianParameters(model, input, decoding_pooling_dictionary.bias, decoding_pooling_dictionary.gradBias)
   mytester:assertlt(err,precision, 'error on decoding pooling dictionary bias ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 10)
   local err = jac.testJacobianUpdateParameters(model, input, decoding_pooling_dictionary.bias, unused_params)
   mytester:assertlt(err,precision, 'error on decoding pooling dictionary bias [full processing chain, direct update] ')

   -- make sure that the random weights assigned to the encoding pooling dictionary for Jacobian testing are non-negative!
   print('encoding pooling dictionary weight')
   local err = jac.testJacobianParameters(model, input, encoding_pooling_dictionary.weight, encoding_pooling_dictionary.gradWeight, 0, 2)
   mytester:assertlt(err,precision, 'error on encoding pooling dictionary weight ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 11)
   local err = jac.testJacobianUpdateParameters(model, input, encoding_pooling_dictionary.weight, unused_params, 0, 2)
   mytester:assertlt(err,precision, 'error on encoding pooling dictionary weight [full processing chain, direct update] ')
   
   print('encoding pooling dictionary bias')
   local err = jac.testJacobianParameters(model, input, encoding_pooling_dictionary.bias, encoding_pooling_dictionary.gradBias, 0, 2)
   mytester:assertlt(err,precision, 'error on encoding pooling dictionary bias ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 12)
   local err = jac.testJacobianUpdateParameters(model, input, encoding_pooling_dictionary.bias, unused_params, 0, 2)
   mytester:assertlt(err,precision, 'error on encoding pooling dictionary bias [full processing chain, direct update] ')

   print('classification pooling dictionary weight')
   local err = jac.testJacobianParameters(model, input, classification_dictionary.weight, classification_dictionary.gradWeight)
   mytester:assertlt(err,precision, 'error on classification dictionary weight ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 13)
   local err = jac.testJacobianUpdateParameters(model, input, classification_dictionary.weight, unused_params)
   mytester:assertlt(err,precision, 'error on classification dictionary weight [full processing chain, direct update] ')
   
   print('classification dictionary bias')
   local err = jac.testJacobianParameters(model, input, classification_dictionary.bias, classification_dictionary.gradBias)
   mytester:assertlt(err,precision, 'error on encoding pooling dictionary bias ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 14)
   local err = jac.testJacobianUpdateParameters(model, input, classification_dictionary.bias, unused_params)
   mytester:assertlt(err,precision, 'error on classification dictionary bias [full processing chain, direct update] ')
end

function rec_pool_test.ISTA_reconstruction()
   -- check that ISTA actually finds a sparse reconstruction.  decoding_dictionary.output should be similar to test_input, and shrink_copies[#shrink_copies].output should have some zeros
   local layer_size = {10, 60, 10, 10}
   local target = math.random(layer_size[4])
   local lambdas = {ista_L2_reconstruction_lambda = math.random(), 
		    ista_L1_lambda = math.random(), 
		    pooling_L2_reconstruction_lambda = math.random(), 
		    pooling_L2_position_unit_lambda = math.random(), 
		    pooling_output_cauchy_lambda = math.random(), 
		    pooling_mask_cauchy_lambda = math.random()}
   local model, criteria_list, encoding_dictionary, decoding_dictionary, encoding_pooling_dictionary, decoding_pooling_dictionary, classification_dictionary, explaining_away, shrink, explaining_away_copies, shrink_copies = 
      build_recpool_net(layer_size, lambdas, 50) -- normalization is not disabled

   local test_input = torch.rand(layer_size[1])
   local target = math.random(layer_size[4])
   model:set_target(target)

   model:updateOutput(test_input)
   print(test_input)
   print(decoding_dictionary.output)
   print(shrink_copies[#shrink_copies].output)

   local test_gradInput = torch.zeros(model.output:size())
   model:updateGradInput(test_input, test_gradInput)



   -- confirm that parameter sharing is working properly
   for i = 1,#shrink_copies do
      if shrink_copies[i].shrink_val:storage() ~= shrink.shrink_val:storage() then
	 print('ERROR!!!  shrink_copies[' .. i .. '] does not share parameters with base shrink!!!')
	 io.read()
      end
      --print('shrink_copies[' .. i .. '] gradInput', shrink_copies[i].gradInput)
      --print('shrink_copies[' .. i .. '] output', shrink_copies[i].output)
   end

   for i = 1,#explaining_away_copies do
      if (explaining_away_copies[i].weight:storage() ~= explaining_away.weight:storage()) or (explaining_away_copies[i].bias:storage() ~= explaining_away.bias:storage()) then
	 print('ERROR!!!  explaining_away_copies[' .. i .. '] does not share parameters with base explaining_away!!!')
	 io.read()
      end
      --print('explaining_away_copies[' .. i .. '] gradInput', explaining_away_copies[i].gradInput)
      --print('explaining_away_copies[' .. i .. '] output', explaining_away_copies[i].output)
   end


   --[[
   local shrink_output_tensor = torch.Tensor(decoding_dictionary.output:size(1), #shrink_copies)
   for i = 1,#shrink_copies do
      shrink_output_tensor:select(2,i):copy(decoding_dictionary:updateOutput(shrink_copies[i].output))
   end
   print(shrink_output_tensor)
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

