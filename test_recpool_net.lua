require 'torch'
require 'nn'
dofile('init.lua')

local mytester = torch.Tester()
local jac

local precision = 1e-5
local expprecision = 1e-4

local rec_pool_test = {}
local other_tests = {}

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

function rec_pool_test.L2Cost()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = {torch.Tensor(ini):zero(), torch.Tensor(ini):zero()}
   local module = nn.L2Cost()

   local err = jac.testJacobianTable(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIOTable(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
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
   dofile('build_recpool_net.lua')
   local layer_size = {10, 10} --{math.random(10,20), math.random(10,20)}
   local L1_lambda = math.random()
   local L2_lambda = math.random()
   local model, criteria_list, forward_dictionary, inverse_dictionary, explaining_away, shrink, explaining_away_copies, shrink_copies = build_recpool_net(layer_size, L2_lambda, L1_lambda, 5, true) -- NORMALIZATION IS DISABLED!!!
   local parameter_list = {forward_dictionary.weight, forward_dictionary.bias, inverse_dictionary.weight, inverse_dictionary.bias, explaining_away.weight, explaining_away.bias, shrink.shrink_val, shrink.negative_shrink_val}

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
   
   print('forward dictionary weight')
   local err = jac.testJacobianParameters(model, input, forward_dictionary.weight, forward_dictionary.gradWeight)
   mytester:assertlt(err,precision, 'error on forward dictionary weight ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 1)
   local err = jac.testJacobianUpdateParameters(model, input, forward_dictionary.weight, unused_params)
   mytester:assertlt(err,precision, 'error on forward dictionary weight [full processing chain, direct update] ')
   
   print('forward dictionary bias')
   local err = jac.testJacobianParameters(model, input, forward_dictionary.bias, forward_dictionary.gradBias)
   mytester:assertlt(err,precision, 'error on forward dictionary bias ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 2)
   local err = jac.testJacobianUpdateParameters(model, input, forward_dictionary.bias, unused_params)
   mytester:assertlt(err,precision, 'error on forward dictionary bias [full processing chain, direct update] ')
   
   print('inverse dictionary weight')
   local err = jac.testJacobianParameters(model, input, inverse_dictionary.weight, inverse_dictionary.gradWeight)
   mytester:assertlt(err,precision, 'error on inverse dictionary weight ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 3)
   local err = jac.testJacobianUpdateParameters(model, input, inverse_dictionary.weight, unused_params)
   mytester:assertlt(err,precision, 'error on inverse dictionary weight [full processing chain, direct update] ')
   
   print('inverse dictionary bias')
   local err = jac.testJacobianParameters(model, input, inverse_dictionary.bias, inverse_dictionary.gradBias)
   mytester:assertlt(err,precision, 'error on inverse dictionary bias ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 4)
   local err = jac.testJacobianUpdateParameters(model, input, inverse_dictionary.bias, unused_params)
   mytester:assertlt(err,precision, 'error on inverse dictionary bias [full processing chain, direct update] ')

   -- explaining_away uses shared weights.  We can only test the gradient using testJacobianParameters if we sum the backwards gradient at all shared copies, since the parameter perturbation in forward affects all shared copies
   local explaining_away_gradWeight_array = {}
   for i,ea in ipairs(explaining_away_copies) do
      explaining_away_gradWeight_array[i] = ea.gradWeight
   end
   print('explaining away weight')
   local err = jac.testJacobianParameters(model, input, explaining_away.weight, explaining_away_gradWeight_array)
   mytester:assertlt(err,precision, 'error on explaining away weight ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 5)
   local err = jac.testJacobianUpdateParameters(model, input, explaining_away.weight, unused_params)
   mytester:assertlt(err,precision, 'error on explaining away weight [full processing chain, direct update] ')
   
   local explaining_away_gradBias_array = {}
   for i,ea in ipairs(explaining_away_copies) do
      explaining_away_gradBias_array[i] = ea.gradBias
   end
   print('explaining away bias')
   local err = jac.testJacobianParameters(model, input, explaining_away.bias, explaining_away_gradBias_array)
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
   local err = jac.testJacobianParameters(model, input, shrink.shrink_val, shrink_grad_shrink_val_array)
   mytester:assertlt(err,precision, 'error on shrink shrink_val ')   
   unused_params = copy_table(parameter_list)
   table.remove(unused_params, 7)
   local err = jac.testJacobianUpdateParameters(model, input, shrink.shrink_val, unused_params)
   mytester:assertlt(err,precision, 'error on shrink shrink_val [full processing chain, direct update] ')

   layer_size[2] = 20
   model, criteria_list, forward_dictionary, inverse_dictionary, explaining_away, shrink, explaining_away_copies, shrink_copies = build_recpool_net(layer_size, L2_lambda, L1_lambda, 500) 
   local test_input = torch.rand(layer_size[1])
   model:updateOutput(test_input)
   print(test_input)
   print(forward_dictionary.output)

   --[[
   local shrink_output_tensor = torch.Tensor(forward_dictionary.output:size(1), #shrink_copies)
   for i = 1,#shrink_copies do
      shrink_output_tensor:select(2,i):copy(forward_dictionary:updateOutput(shrink_copies[i].output))
   end
   print(shrink_output_tensor)
   --]]
end







--local num_tests = 0 
--for name in pairs(rec_pool_test) do num_tests = num_tests + 1; print('test ' .. num_tests .. ' is ' .. name) end
--print('number of tests: ', num_tests)

mytester:add(rec_pool_test)
--mytester:add(other_tests)

jac = nn.Jacobian
mytester:run()

