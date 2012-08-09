require 'unsup'
require 'image'
require 'gnuplot'
require 'mnist'

dofile('init.lua')

-- NOTE: I have disabled hessian.lua in the kex init.  Otherwise, it overwrites a bunch of methods in the standard nn package

if not arg then arg = {} end

cmd = torch.CmdLine()

cmd:text()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on MNIST')
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-name', '', 'name of directory in which experiments are saved')
cmd:option('-dir','outputs', 'subdirectory in which to save experiments')
cmd:option('-load_file','', 'file from which to load experiments')
cmd:option('-test_mode','no', 'test rather than train')
cmd:option('-record_lagrange_hist','yes', 'generate a graph with the full history of the lagrange multipliers')
cmd:option('-seed', 123211, 'initial random seed')
cmd:option('-dictionarysize', 200, 'number of dictionary elements') -- 200
cmd:option('-lambda', 1.2e-2, 'sparsity coefficient')  --1.2e-2 --50 --5e-2 with L2 disabled -- 1e-3 with L2 enabled --4e-4
cmd:option('-beta', 1, 'prediction error coefficient')
--cmd:option('-datafile', 'tr-berkeley-N5K-M56x56-lcn.bin','Data set file')
-- for use with fixed L1 dictionary
--cmd:option('-eta',300,'learning rate') --200 with L2 enabled --500 --100  -- 0.00002 with LinearFistaL1 -- 0.0001
-- for use with variable L1 dictionary
--cmd:option('-eta_encoder',100,'encoder learning rate') --0.2 with L2 disabled  -- 2 with L2 enabled   --0.0001 with 100 quasi-cauchy --0.0005 with 30 quasi-cauchy --0.001 with 10 quasi-cauchy --0.000005 with mse
cmd:option('-eta',0.5,'learning rate')  -- 100 before removal of kex.nnhacks-- 50 -- 100 -- 300
cmd:option('-eta_encoder',0.05,'encoder learning rate') -- 10 before removal of kex.nnhacks -- 100; 60; 30 --0.2 with L2 disabled  -- 2 with L2 enabled   --0.0001 with 100 quasi-cauchy --0.0005 with 30 quasi-cauchy --0.001 with 10 quasi-cauchy --0.000005 with mse
cmd:option('-momentum',0,'gradient momentum')
cmd:option('-decay',0,'weight decay')
cmd:option('-datasetsize',5000,'number of elements loaded from dataset')
cmd:option('-maxiter',1000000,'max number of updates')
cmd:option('-textstatinterval',1000,'interval for displaying stats and models') -- 1000
cmd:option('-diskstatinterval',250000,'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
cmd:text()

local params = cmd:parse(arg)

print('test mode is set to: ' .. params.test_mode)

if params.name ~= '' then params.name = ('_' .. params.name) end
local rundir = cmd:string('psd_mnist' .. params.name, params, {dir=true, name=true, load_file=true, test_mode=true, record_lagrange_hist=true})
params.rundir = params.dir .. '/' .. rundir

if paths.dirp(params.rundir) then
   error('This experiment is already done!!!')
end

os.execute('mkdir -p ' .. params.rundir)
cmd:log(params.rundir .. '/log', params)
os.execute('cp demo_psd_mnist.lua ' .. params.rundir)
os.execute('cp ~/Code/torch/share/torch/lua/unsup/LinearFactoredFistaL1Auto.lua ' .. params.rundir)
os.execute('cp ~/torch_install/share/torch/lua/unsup/LinearFactoredFistaL1Auto.lua ' .. params.rundir)


-- init random number generator
torch.manualSeed(params.seed)

-- create the dataset
if params.test_mode == 'support' then -- don't load the full dataset if we're not going to use it
   params.datasetsize = 1
end
data = mnist.loadTrainSet(params.datasetsize)
data:normalizeL2() -- normalize each example to have L2 norm equal to 1

-- create unsup stuff
local input_code_size = data[1][1]:size(1);
local target_code_size = data[1][2]:size(1) -- look at the label vector of the first example to determine the size of the target code
print('Input size is ' .. input_code_size .. ' ; target size is ' .. target_code_size)
fista_params = {maxiter = 50, encoderType = 'tanh_shrink_parallel'}; -- was using tanh_shrink
mlp = unsup.HLinearPsd(input_code_size, params.dictionarysize, target_code_size, params.lambda, params.beta, fista_params)

-- run a unit test to make sure that everything is working properly
print('Before unit tests')
--test_factored_sparse_coder_main_chain(mlp.decoder)
local unit_test_example_index = math.random(params.datasetsize)
local unit_test_example = data[unit_test_example_index]
test_factored_sparse_coder_ista(mlp, unit_test_example[1], unit_test_example[2])
print('After unit tests')
io.read()



-- load parameters if desired
if params.load_file ~= '' then
   print('loading parameters from ' .. params.load_file)
   local mf = torch.DiskFile(params.load_file,'r'):binary()
   local saved_parameters = mf:readObject()
   mlp:setParameters(saved_parameters)
   mf:close()
end
   

-- learning rates
if params.eta_encoder == 0 then params.eta_encoder = params.eta end
params.eta = torch.Tensor({params.eta_encoder, params.eta})

-- do learning rate hacks
--kex.nnhacks() -- this is defined in kex/stochasticrates.lua, and appears to scale the learning rate of every nn.Linear module by the length of its column.  As a result, top_level_classifier updates have been much larger than those of the other matrices

local num_correct_classifications = 0  --self.num_correct_classifications or 0
local num_incorrect_classifications = 0 --self.num_incorrect_classifications or 0

local function count_correct_classifications(classifier_output, target)
   --local classifier_output = self.top_level_classifier:updateOutput(concat_code)
   local calc_max, calc_max_ind = torch.max(classifier_output, 1)
   local true_max, true_max_ind = torch.max(target, 1)
   if calc_max_ind[1] ~= true_max_ind[1] then
      --print('Incorrect prediction')
      --print(torch.cat(classifier_output:unfold(1,10,10), target:unfold(1,10,10), 1))
      num_incorrect_classifications = num_incorrect_classifications + 1
      return false
   else
      num_correct_classifications = num_correct_classifications + 1
      return true
   end
end 


function train(module,dataset)

   local avTrainingError = torch.FloatTensor(math.ceil(params.maxiter/params.textstatinterval)):zero()
   local avFistaIterations = torch.FloatTensor(math.ceil(params.maxiter/params.textstatinterval)):zero()
   local currentLearningRate = params.eta
   local lagrange_history_L1_units = nil
   local lagrange_history_cmul_mask = nil
   
   
   module:zeroGradParameters() -- do this once to initialize the system; hereafter, only zero gradParameters after a parameter update

   local function updateSample(input, target, eta, calculate_param_update, do_param_update)
      -- if calculate_param_update or do_param_update are not specified, make them true
      if calculate_param_update == nil then calculate_param_update = true end
      if do_param_update == nil then do_param_update = true end

      local err,h = module:updateOutput(input, target)      

      if calculate_param_update then 
	 --print('calculate param updates')
	 module:updateGradInput(input, target)
	 module:accGradParameters(input, target)
      end

      local print_gradient_magnitudes = false
      if print_gradient_magnitudes then 
	 local accum_cmul = 0
	 module.decoder.cmul_dictionary.gradWeight:apply(function (x) accum_cmul = accum_cmul + x^2 end)
	 accum_cmul = math.sqrt(accum_cmul)
	 
	 local accum_L1 = 0
	 if mlp.decoder.use_L1_dictionary then 
	    module.decoder.L1_dictionary.gradWeight:apply(function (x) accum_L1 = accum_L1 + x^2 end)
	 end
	 accum_L1 = math.sqrt(accum_L1)
	 
	 local accum_classifier = 0
	 module.decoder.top_level_classifier_dictionary.gradWeight:apply(function (x) accum_classifier = accum_classifier + x^2 end)
	 accum_classifier = math.sqrt(accum_classifier)
	 
	 print('gradient mags are - cmul: ' .. accum_cmul .. ' L1: ' .. accum_L1 .. ' classifier: ' .. accum_classifier)
      end

      if do_param_update then
	 --print('do param update')
	 module:updateParameters(eta)
	 module:zeroGradParameters()
      end
      return err, #h, h[#h].F - h[#h-1].F
   end

   local function generate_incorrect_target(correct_target)
      local incorrect_target = torch.zeros(correct_target:size(1))
      local correct_target_location = 0
      for i = 1,correct_target:size(1) do 
	 if correct_target[i] == 1 then 
	    correct_target_location = i
	    break
	 end 
      end
      
      local potential_index = math.random(incorrect_target:size(1) - 1)
      potential_index = potential_index + ((potential_index >= correct_target_location and 1) or 0)
      incorrect_target[potential_index] = 1

      --[[
      print('For sleep, changing between: ')
      print(torch.cat(correct_target, incorrect_target, 2):t())
      --]]
      return incorrect_target
   end
	 
   local function plot_training_error(t)
      gnuplot.pngfigure(params.rundir .. '/error.png')
      gnuplot.plot(avTrainingError:narrow(1,1,math.max(t/params.textstatinterval,2)))
      gnuplot.title('Training Error')
      gnuplot.xlabel('# iterations / ' .. params.textstatinterval)
      gnuplot.ylabel('Cost')
      -- plot number of fista iterations required for training
      gnuplot.pngfigure(params.rundir .. '/iter.png')
      gnuplot.plot(avFistaIterations:narrow(1,1,math.max(t/params.textstatinterval,2)))
      gnuplot.title('Fista Iterations')
      gnuplot.xlabel('# iterations / ' .. params.textstatinterval)
      gnuplot.ylabel('Fista Iterations')
      -- plot evolution of L1 lagrange multipliers
      if mlp.decoder.use_lagrange_multiplier_L1_units and lagrange_history_L1_units then
	 local lagrange_L1_units_fig_data = {}
	 for i = 1,lagrange_history_L1_units:size(1) do
	    lagrange_L1_units_fig_data[i] = {'L1 unit ' .. i, torch.linspace(1,lagrange_history_L1_units:size(2),lagrange_history_L1_units:size(2)), lagrange_history_L1_units:select(1,i), '-'}
	 end
	 gnuplot.pngfigure(params.rundir .. '/lagrange_multipliers_L1_units.png')
	 gnuplot.plot(lagrange_L1_units_fig_data)
	 gnuplot.title('Lagrange L1 unit evolution')
	 gnuplot.xlabel('# iterations')
	 gnuplot.ylabel('Lagrange L1 unit value')
      end

      if mlp.decoder.use_lagrange_multiplier_cmul_mask and lagrange_history_cmul_mask then
	 local lagrange_cmul_mask_fig_data = {}
	 for i = 1,lagrange_history_cmul_mask:size(1) do
	    lagrange_cmul_mask_fig_data[i] = {'cmul mask ' .. i, torch.linspace(1,lagrange_history_cmul_mask:size(2),lagrange_history_cmul_mask:size(2)), lagrange_history_cmul_mask:select(1,i), '-'}
	 end
	 gnuplot.pngfigure(params.rundir .. '/lagrange_multipliers_cmul_mask.png')
	 gnuplot.plot(lagrange_cmul_mask_fig_data)
	 gnuplot.title('Lagrange cmul mask evolution')
	 gnuplot.xlabel('# iterations')
	 gnuplot.ylabel('Lagrange cmul mask value')
      end

      -- clean up plots
      gnuplot.plotflush()
      gnuplot.closeall()
   end

   local function plot_filters(t)
      local db, dc, dd, de, df
      if torch.typename(mlp) == 'unsup.HLinearPsd' then
	 --dd = image.toDisplayTensor{input=mlp.decoder.D.weight:transpose(1,2):unfold(2,28,28),padding=1,nrow=10,symmetric=true}
	 dd = image.toDisplayTensor{input=mlp.decoder.cmul_dictionary.weight:transpose(1,2):unfold(2,28,28),padding=1,nrow=10,symmetric=true}
	 if mlp.decoder.use_L1_dictionary then
	    --print(mlp.decoder.L1_dictionary.weight)
	    dc = image.toDisplayTensor{input=mlp.decoder.L1_dictionary.weight:transpose(1,2):unfold(2,10,10),padding=1,nrow=10,symmetric=true}
	    db = image.toDisplayTensor{input=torch.mm(mlp.decoder.cmul_dictionary.weight, mlp.decoder.L1_dictionary.weight):transpose(1,2):unfold(2,28,28),padding=1,nrow=10,symmetric=true}
	 else
	    db = image.toDisplayTensor{input=mlp.decoder.cmul_dictionary.weight:transpose(1,2):unfold(2,28,28),padding=1,nrow=10,symmetric=true}
	    print('skipping plot of L1_dictionary')
	 end
	 if mlp.decoder.use_top_level_classifier then
	    df = image.toDisplayTensor{input=mlp.decoder.top_level_classifier_dictionary.weight:transpose(1,2):unfold(2,10,10),padding=1,nrow=10,symmetric=true}
	 end
	 if not fista_params or fista_params.encoderType == 'linear' then 
	    de = image.toDisplayTensor{input=mlp.encoder.weight:unfold(2,28,28),padding=1,nrow=10,symmetric=true}
	 elseif fista_params.encoderType == 'tanh_shrink_parallel' then
	    --print(mlp.encoder:get(2):get(2):get(1).weight:unfold(2,28,28):size())
	    de = image.toDisplayTensor{input=torch.cat(mlp.encoder:get(2):get(1):get(1).weight:unfold(2,28,28), mlp.encoder:get(2):get(2):get(1).weight:unfold(2,28,28), 1),padding=1,nrow=10,symmetric=true}
	 else
	    de = image.toDisplayTensor{input=mlp.encoder:get(1).weight:unfold(2,28,28),padding=1,nrow=10,symmetric=true}
	 end
      else
	 de = image.toDisplayTensor{input=mlp.encoder.weight,padding=1,nrow=8,symmetric=true}
	 dd = image.toDisplayTensor{input=mlp.decoder.D.weight,padding=1,nrow=8,symmetric=true}
      end
      --gnuplot.imagesc(dd)
      image.savePNG(params.rundir .. '/filters_dec_' .. t .. '.png',dd)
      image.savePNG(params.rundir .. '/filters_enc_' .. t .. '.png',de)
      if mlp.decoder.use_L1_dictionary then
	 image.savePNG(params.rundir .. '/filters_L1_dec_' .. t .. '.png',dc)
	 image.savePNG(params.rundir .. '/filters_L1_proj_dec_' .. t .. '.png',db)
      end
      if mlp.decoder.use_top_level_classifier then 
	 image.savePNG(params.rundir .. '/filters_class_' .. t .. '.png',df)
      end
   end



   -- Train the network

   local err = 0
   local iter = 0

   for t = 1,params.maxiter do
      -- WARNING: THIS GOES THROUGH THE DATASET IN ORDER!  WE SHOULD PROBABLY RANDOMIZE THIS!!!
      local index = (t % params.datasetsize) + 1

      local example = dataset[index]
      -- Plot the current example, to make sure that the dataset is constructed properly
      --gnuplot.imagesc(example[1]:unfold(1,28,28))
      --print('norm is ' .. example[1]:norm())
      --print(example[2])
      --io.read()

      local serr, siter, serr_change

      if t == 100001 then -- an initial period in which the L1_dictionary is untrained seems to be *critical* to good performance.  Make sure that classification performance plateaus before L1_dictionary training is enabled.
	 -- THIS SHOULD PROBABLY BE 1 NOW!!!  Note that if this is not 1, then the energy function minimized by the dynamics is *NOT* the same as the energy function minimized by training, so the partial derivative is not the total derivative
	 module.decoder.L1_dictionary_learning_rate_scaling = 1 --3e-1 -- even after cmul_dictionary pretraining, the L1_dictionary seems to change disproportionately and dangerously quickly
      end
      

      if t % 200 == 0 then -- CONTINUE HERE!!!  We should probably only test ista after the network has trained for a bit, or use a saved network
	 test_factored_sparse_coder_ista(mlp, example[1], example[2])
	 print('After unit tests')
	 io.read()
      end


      if module.decoder.use_top_level_classifier then -- do a wake/sleep update if the decoder has a top-level classifier
	 --[[local sleep_learning_rate = currentLearningRate:clone()
	 sleep_learning_rate[{{1,1}}] = 0
	 sleep_learning_rate[{{2,2}}]:mul(-0.1) --]]

	 	 
	 -- sleep stage
	 module.decoder.wake_sleep_stage = 'sleep'
	 --local sleep_stage_target = generate_incorrect_target(example[2])
	 --updateSample(example[1], sleep_stage_target, currentLearningRate, true, true)
	 updateSample(example[1], example[2], currentLearningRate, true, true)
	 
	 -- evaluate performance
	 module.decoder.wake_sleep_stage = 'test'
	 updateSample(example[1], example[2], currentLearningRate, false, false)
	 -- presently, the top level classifier is run on every iteration of test mode (unnecessarily, but probably not at great cost).  Importantly, the top-level classification error is *not* incorporated into the returned values; this screws up the line search in fista
	 count_correct_classifications(module.decoder.top_level_classifier.output, example[2])

	 -- wake stage
	 module.decoder.wake_sleep_stage = 'wake'
	 serr, siter, serr_change = updateSample(example[1], example[2], currentLearningRate, true, true)

      else -- otherwse, just do a standard update
	 serr, siter, serr_change = updateSample(example[1], example[2], currentLearningRate, true, true)
      end

      err = err + serr
      iter = iter + siter

      if (params.record_lagrange_hist == 'yes') and mlp.decoder.use_lagrange_multiplier_L1_units then -- REMOVE THIS to avoid the inefficiency of tracking the L1 lagrange multiplier
	 lagrange_history_L1_units = (lagrange_history_L1_units and torch.cat(lagrange_history_L1_units, mlp.decoder.lagrange_multiplier_L1_units:narrow(1,1,10), 2)) or mlp.decoder.lagrange_multiplier_L1_units:narrow(1,1,10)
      end
      
      if (params.record_lagrange_hist == 'yes') and mlp.decoder.use_lagrange_multiplier_cmul_mask then -- REMOVE THIS to avoid the inefficiency of tracking the L1 lagrange multiplier
	 lagrange_history_cmul_mask = (lagrange_history_cmul_mask and torch.cat(lagrange_history_cmul_mask, mlp.decoder.lagrange_multiplier_cmul_mask:narrow(1,1,10), 2)) or mlp.decoder.lagrange_multiplier_cmul_mask:narrow(1,1,10)
      end

	 

      if t == 1 then 
	 plot_filters(t) -- just to confirm that the parameter load worked correctly
      end

      if math.fmod(t , params.textstatinterval) == 0 then
	 avTrainingError[t/params.textstatinterval] = err/params.textstatinterval
	 avFistaIterations[t/params.textstatinterval] = iter/params.textstatinterval

	 -- report
	 --print('# iter=' .. t .. ' eta = ( ' .. currentLearningRate[1] .. ', ' .. currentLearningRate[2] .. ' ) current error = ' .. serr .. ' with ' .. siter .. ' iters ' .. serr_change .. ' last change')
	 print('# iter=' .. t .. ' current error = ' .. serr .. ' with ' .. siter .. ' iters ' .. serr_change .. ' last change')


	 print(num_correct_classifications .. ' correct test classifications out of ' .. num_correct_classifications + num_incorrect_classifications) 
	 num_correct_classifications, num_incorrect_classifications = 0,0
	 
	 print('Code: ')
	 print(mlp.decoder.extract_L2_from_concat(mlp.decoder.code):unfold(1,10,10))
	 print(mlp.decoder.cmul.output:unfold(1,10,10))
	 print(mlp.decoder.extract_L1_from_concat(mlp.decoder.code):unfold(1,10,10))
	 print(mlp.decoder.top_level_classifier.output:unfold(1,10,10))
	 print(example[2]:unfold(1,10,10))
	 if mlp.decoder.use_lagrange_multiplier_L1_units then
	    print('Average L1 units lambda is ' .. torch.mean(mlp.decoder.lagrange_multiplier_L1_units))
	 end
	 if mlp.decoder.use_lagrange_multiplier_cmul_mask then
	    print('Average cmul mask lambda is ' .. torch.mean(mlp.decoder.lagrange_multiplier_cmul_mask))
	 end
	 if mlp.decoder.use_L1_dictionary then
	    print('Average magnitude of L1 dictionary is ' .. torch.mean(torch.abs(mlp.decoder.L1_dictionary.weight)) .. ' ; max element is ' .. torch.max(torch.abs(mlp.decoder.L1_dictionary.weight)))
	 end


	 -- plot training error
	 plot_training_error(t) --(avTrainingError, avFistaIterations, lagrange_history_L1_units, mlp, params, t)
	 

	 -- plot filters
	 plot_filters(t) --(mlp, params, t)
	 
	 -- write training error
	 local tf = torch.DiskFile(params.rundir .. '/error.mat','w'):binary()
	 tf:writeObject(avTrainingError:narrow(1,1,t/params.textstatinterval))
	 tf:close()
	 
	 -- write # of iterations
	 local ti = torch.DiskFile(params.rundir .. '/iter.mat','w'):binary()
	 ti:writeObject(avFistaIterations:narrow(1,1,t/params.textstatinterval))
	 ti:close()

	 -- update learning rate with decay
	 currentLearningRate = params.eta/(1+(t/params.textstatinterval)*params.decay)
	 err = 0
	 iter = 0
      end -- textstatinterval

      if math.fmod(t, params.diskstatinterval) == 0 then
	 -- store model
	 print('starting to store model')
	 local mf = torch.DiskFile(params.rundir .. '/model_' .. t .. '.bin','w'):binary()
	 print('about to writeObject')
	 print(module:parameters())
	 mf:writeObject(module:parameters())
	 print('about to close')
	 mf:close()
	 print('finished storing model')
      end

   end -- loop over iterations
end -- function train


function test(module, dataset)
   local desired_output_fig = 1
   local reconstructed_output_fig = 2
   for t = 1,params.maxiter do
      -- WARNING: THIS GOES THROUGH THE DATASET IN ORDER!  WE SHOULD PROBABLY RANDOMIZE THIS!!!
      local index = (t % params.datasetsize) + 1

      local example = dataset[index]

      local serr, siter, serr_change

      -- It's necessary to learn the correct lagrange multipliers before we begin evaluating the network!!!  Really, the lagrange multipliers should be saved along with the dictionary weights

      -- evaluate performance
      module.decoder.wake_sleep_stage = 'test'
      local err,h = module:updateOutput(example[1], example[2])
      -- presently, the top level classifier is run on every iteration of test mode (unnecessarily, but probably not at great cost).  Importantly, the top-level classification error is *not* incorporated into the returned values; this screws up the line search in fista
      local right_class = count_correct_classifications(module.decoder.top_level_classifier.output, example[2])

      -- Plot the current example
      gnuplot.figure(desired_output_fig)
      gnuplot.imagesc(example[1]:unfold(1,28,28))
      print(example[2]:unfold(1,10,10))

      gnuplot.figure(reconstructed_output_fig)
      gnuplot.imagesc(module.decoder.cmul_dictionary.output:unfold(1,28,28))
      print(module.decoder.top_level_classifier.output:unfold(1,10,10))
      
      print('# iter=' .. t)
      print(num_correct_classifications .. ' correct sleep classifications out of ' .. num_correct_classifications + num_incorrect_classifications) 
      num_correct_classifications, num_incorrect_classifications = 0,0
      
      print('Code: ')
      print(mlp.decoder.extract_L2_from_concat(mlp.decoder.code):unfold(1,10,10))
      print(mlp.decoder.cmul.output:unfold(1,10,10))
      print(mlp.decoder.extract_L1_from_concat(mlp.decoder.code):unfold(1,10,10))
      print(mlp.decoder.top_level_classifier.output:unfold(1,10,10))
      print(example[2]:unfold(1,10,10))
      if mlp.decoder.use_lagrange_multiplier_L1_units then
	 print('Average L1 units lambda is ' .. torch.mean(mlp.decoder.lagrange_multiplier_L1_units))
      end
      if mlp.decoder.use_lagrange_multiplier_cmul_mask then
	 print('Average cmul mask lambda is ' .. torch.mean(mlp.decoder.lagrange_multiplier_cmul_mask))
      end
      if mlp.decoder.use_L1_dictionary then
	 print('Average magnitude of L1 dictionary is ' .. torch.mean(torch.abs(mlp.decoder.L1_dictionary.weight)) .. ' ; max element is ' .. torch.max(torch.abs(mlp.decoder.L1_dictionary.weight)))
      end

      if right_class then 
	 print('Correct')
      else
	 print('Incorrect')
      end


      io.read()
      
   end -- loop over iterations
end -- function test


function display_L1_dictionary_support(module)
   local L1_weights = mlp.decoder.L1_dictionary.weight
   local cmul_weights = mlp.decoder.cmul_dictionary.weight
   local max_num_connections = 10
   local folded_output = torch.Tensor(max_num_connections*L1_weights:size(2),28,28):zero() -- first dim is index of weights; second and third dim are height and width.  Initially, put the desired array of images in a line of images, and use toDisplayTensor to unroll the line of images into a 2d array of images with appropriate spacing

   -- WE MAY WANT TO TAKE THE ABSOLUTE VALUE BEFORE SORTING, IF WEIGHTS CAN BE NEGATIVE!!!
   local connection_mags, connection_mags_index = torch.sort(L1_weights:t()) -- torch.sort sorts each row independently, so transpose before and after the sort to operate on each column
   connection_mags = connection_mags:t()
   connection_mags_index = connection_mags_index:t()

   --print(L1_weights:select(2,1):unfold(1,10,10))
   --print(connection_mags:select(2,1):unfold(1,10,10))
   --print(connection_mags_index:select(2,1):unfold(1,10,10))
   --io.read()

   --gnuplot.figure()

   for L1_unit_index = 1,connection_mags:size(2) do
      for unmod_connection_index = 1,max_num_connections do
	 local connection_index = connection_mags_index:size(1) - unmod_connection_index + 1 -- connections are sorted from smallest-to-largest; convert index to largest-to-smallest

	 if connection_mags[{connection_index, L1_unit_index}] > 5e-2 then -- if the ith connection from this L1 unit is non-zero
	    folded_output:select(1, (L1_unit_index-1)*max_num_connections + unmod_connection_index):copy(cmul_weights:select(2, connection_mags_index[{connection_index, L1_unit_index}]):unfold(1,28,28))
	 end
      end
   end

   print(folded_output:size())
   --[[
   for i = 1,folded_output:size(1) do
      print(i)
      gnuplot.imagesc(folded_output:select(1,i):t())
      io.read()
   end
   --]]
      
   --dd = image.toDisplayTensor{input=mlp.decoder.cmul_dictionary.weight:transpose(1,2):unfold(2,28,28),padding=1,nrow=10,symmetric=true}
   final_image = image.toDisplayTensor{input=folded_output,padding=1,nrow=max_num_connections,symmetric=true}
   --gnuplot.imagesc(image.toDisplayTensor{input=folded_output,padding=1,nrow=max_num_connections,symmetric=true})
   image.savePNG(params.rundir .. '/support_of_L1_dict.png', final_image)
   
   for i = 1,connection_mags:size(2) do
      print(connection_mags:select(2,i):unfold(1,10,10))
      io.read()
   end
   
end



if params.test_mode == 'no' then
   train(mlp, data)
elseif params.test_mode == 'yes' then
   test(mlp, data)
elseif params.test_mode == 'support' then
   display_L1_dictionary_support(mlp)
end
