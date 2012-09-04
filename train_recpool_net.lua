----------------------------------------------------------------------
-- This script demonstrates how to define a training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It shows how to:
--   + construct mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to several optmization
--     methods: SGD, L-BFGS.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

local RecPoolTrainer = torch.class('nn.RecPoolTrainer')

function RecPoolTrainer:__init(model, opt)
   -- set default options
   if not opt then
      opt = {}
   end
   self.opt = {}

   self.opt.log_directory = opt.log_directory or 'recpool_results' -- subdirectory in which to save/log experiments
   self.opt.visualize = opt.visualize or false -- visualize input data and weights during training
   self.opt.plot = opt.plot or false -- live plot
   self.opt.optimization = opt.optimization or 'SGD' -- optimization method: SGD | ASGD | CG | LBFGS
   self.opt.learning_rate = opt.learning_rate or 1e-3 -- learning rate at t=0
   self.opt.batch_size = opt.batch_size or 1 -- mini-batch size (1 = pure stochastic)
   self.opt.weight_decay = opt.weight_decay or 0 -- weight decay (SGD only)
   self.opt.momentum = opt.momentum or 0 -- momentum (SGD only)
   self.opt.t0 = opt.t0 or 1 -- start averaging at t0 (ASGD only), in number (?!?) of epochs -- WHAT DOES THIS MEAN?
   self.opt.max_iter = opt.max_iter or 2 -- maximum nb of iterations for CG and LBFGS
   
   -- allowed output classes
   self.classes = {'1','2','3','4','5','6','7','8','9','0'}
   
   -- This matrix records the current confusion across classes
   self.confusion = optim.ConfusionMatrix(self.classes)
   self.loss_hist = {}   
   self.grad_loss_hist = {}
   for i = 1,#model.criteria_list.criteria do
      self.loss_hist[i] = 0
      self.grad_loss_hist[i] = 0
   end
   self.num_zero_hist = {}
   
   -- Log results to files
   self.train_logger = optim.Logger(paths.concat(self.opt.log_directory, 'train.log'))
   --self.test_logger = optim.Logger(paths.concat(self.opt.log_directory, 'test.log'))

   -- Flatten the parameters (and gradParameters) into a single giant storage.  Each parameter and gradParameter tensor then views an offset into the common storage.  Shared parameters are only stored once, since Module:share() already sets the associated tensors to point to a common storage.
   if model then
      self.parameters,self.gradParameters = model:getParameters() 
   else
      error('RecPoolTrainer requires a model')
   end
   self.model = model

   self.minibatch_inputs = {} -- these allow communication between the train function and the feval closure
   self.minibatch_targets = {}

   -- note that feval takes only current_params as input, whereas make_feval takes self as input; the self provided to make_feval is accessible to feval through the closure
   self.feval = self:make_feval() 
   
   self.epoch = 0
end


-- create closure to evaluate f(X) and df/dX; the closure is necessary so minibatch_inputs and self.minibatch_targets are correct.  
-- self is provided by the column notation.  Thereafter, feval can be called without self as an argument, and the closure provides access to the (implicit) self argument of make_feval
function RecPoolTrainer:make_feval()
   --[[
   local internal_counter = 1
   local found_a_nan = false

   local function find_nans(x)
      if x ~= x then
	 found_a_nan = true
      end
   end
   --]]
      
   local feval = function(current_params)
      -- enforce all constraints on parameters, since the parameters are updated manually, rather than through updateParameters as generally expected
      self.model:repair() -- THIS DOUBLES THE TIME REQUIRED PER ITERATION!  THE COMPONENT OPERATIONS SHOULD BE IMPLEMENTED IN C!!!

      -- get new parameters
      if current_params ~= self.parameters then 
	 self.parameters:copy(current_params)
	 print('copying parameters in feval') -- does this ever actually run?
      end
      
      -- Reset gradients.  This is more efficient than self.model:zeroGradParameters(), since gradParameters has all gradients flattened into a single storage, viewed by the many parameter tensors.  As a result, when parameters are shared by multiple modules, they are only zeroed once by this procedure.
      self.gradParameters:zero() 
      
      -- total_err is the average of the error over the entire minibatch
      local total_err = 0
      
      -- evaluate function for complete minibatch
      for i = 1,#self.minibatch_inputs do
	 -- estimate total_err
	 self.model:set_target(self.minibatch_targets[i])
	 local err = self.model:updateOutput(self.minibatch_inputs[i])
	 local output = self.model:get_classifier_output() -- while the model is a nn.Sequential, it terminates in a set of criteria
	 total_err = total_err + err[1] -- the err returned by updateOutput is a tensor with one element, to maintain compatibility with ModfifiedJacobian
	 
	 -- estimate the gradient of the error with respect to the parameters: d total_err / dW
	 self.model:updateGradInput(self.minibatch_inputs[i]) -- gradOutput is not required, since all computation streams terminate in a criterion; implicitly pass nil
	 self.model:accGradParameters(self.minibatch_inputs[i])
	 
	 -- update the confusion matrix.  This keeps track of the predicted output (maximum output conditional posterior probability) for each true output class
	 self.confusion:add(output, self.minibatch_targets[i])

	 -- track the evolution of sparsity and reconstruction errors
	 for j = 1,#(self.model.criteria_list.criteria) do
	    self.loss_hist[j] = self.model.criteria_list.criteria[j].output + self.loss_hist[j]
	    --print(self.model.criteria_list.names[j], self.model.criteria_list.criteria[j].gradInput)
	    if type(self.model.criteria_list.criteria[j].gradInput) == 'table' then
	       for k = 1,#self.model.criteria_list.criteria[j].gradInput do
		  self.grad_loss_hist[j] = self.model.criteria_list.criteria[j].gradInput[k]:norm() + self.grad_loss_hist[j]
	       end
	    else
	       self.grad_loss_hist[j] = self.model.criteria_list.criteria[j].gradInput:norm() + self.grad_loss_hist[j]
	    end
	 end
      end
      
      -- normalize gradients and f(X)
      self.gradParameters:div(#self.minibatch_inputs)
      total_err = total_err / #self.minibatch_inputs
      
      -- return f and df/dX
      return total_err, self.gradParameters
   end
   
   return feval
end


function RecPoolTrainer:train(train_data)
   self.epoch = self.epoch + 1
   
   -- local vars
   local time = sys.clock()
   
   -- shuffle at each epoch
   local shuffle = torch.randperm(data:size()) --was trsize

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. self.epoch .. ' [batch_size = ' .. self.opt.batch_size .. ']')
   for t = 1, train_data:size(), self.opt.batch_size do
      -- disp progress
      xlua.progress(t, train_data:size())
      
      -- create mini batch.  The minibatch_inputs and minibatch_targets elements of a RecPoolTrainer are viewed directly by the feval made by make_feval()
      self.minibatch_inputs = {}
      self.minibatch_targets = {}
      for i = t,math.min(t+self.opt.batch_size-1,train_data:size()) do
         -- load new sample
         local this_input = train_data.data[shuffle[i]]:double()
         local this_target = train_data.labels[shuffle[i]]
         table.insert(self.minibatch_inputs, this_input)
         table.insert(self.minibatch_targets, this_target)
      end
      
      -- optimize on current mini-batch
      if self.opt.optimization == 'CG' then
         self.config = self.config or {maxIter = self.opt.max_iter}
         optim.cg(self.feval, self.parameters, self.config)
	 
      elseif self.opt.optimization == 'LBFGS' then
         self.config = self.config or {learningRate = self.opt.learning_rate,
                             maxIter = self.opt.max_iter,
                             nCorrection = 10}
         optim.lbfgs(self.feval, self.parameters, self.config)
	 
      elseif self.opt.optimization == 'SGD' then
         self.config = self.config or {learningRate = self.opt.learning_rate,
                             weightDecay = self.opt.weight_decay,
                             momentum = self.opt.momentum,
                             learningRateDecay = 5e-7}
         optim.sgd(self.feval, self.parameters, self.config)
	 
      elseif self.opt.optimization == 'ASGD' then
         self.config = self.config or {eta0 = self.opt.learning_rate,
                             t0 = trsize * self.opt.t0}
         _,_,average = optim.asgd(self.feval, self.parameters, self.config)
	 
      else
         error('unknown optimization method')
      end
   end
   
   -- time taken for the current epoch (each call to train() only runs one epoch)
   time = sys.clock() - time
   time = time / train_data:size()
   print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')
   
   print(self.confusion) -- print the confusion matrix for the current epoch
   
   -- update logger/plot
   self.train_logger:add{['% mean class accuracy (train set)'] = self.confusion.totalValid * 100}
   if self.opt.plot then
      self.train_logger:style{['% mean class accuracy (train set)'] = '-'}
      self.train_logger:plot()
   end

   self.confusion:zero()
   for i = 1,#self.model.criteria_list.criteria do
      print('Criterion: ' .. self.model.criteria_list.names[i] .. ' = ' .. self.loss_hist[i]/train_data:size() .. '; grad = ' .. self.grad_loss_hist[i]/train_data:size())
      self.loss_hist[i] = 0
      self.grad_loss_hist[i] = 0
   end
   
   print('final shrink output', self.model.shrink_copies[#self.model.shrink_copies].output:unfold(1,10,10))
   print('pooling reconstruction', self.model.decoding_pooling_dictionary.output:unfold(1,10,10))
   print('pooling position units', self.model.L2_pooling_units.output:unfold(1,10,10))
   print('pooling output', self.model.pooling_seq.output[1]:unfold(1,10,10))
   -- display filters!  Also display reconstructions minus originals, so we can see how the reconstructions improve with training!
   -- check that without regularization, filters are meaningless.  Confirm that trainable pooling has an effect on the pooled filters.
   plot_reconstructions(self.opt, train_data.data[shuffle[train_data:size()]]:double(), self.model.decoding_feature_extraction_dictionary.output)


   --[[
   -- save/log current net
   local filename = paths.concat(self.opt.log_directory, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, self.model)
   --]]

end



	 --[[
	 if internal_counter % 100 == 1 then
	    print(output:unfold(1,10,10))
	    print(self.model.encoding_feature_extraction_dictionary.weight[{1,{1,10}}]:unfold(1,10,10))
	    print(self.model.explaining_away.weight[{1,{1,10}}]:unfold(1,10,10))
	    print(self.model.shrink.shrink_val[{{1,10}}]:unfold(1,10,10))
	    print(self.model.encoding_pooling_dictionary.weight[{1,{1,10}}]:unfold(1,10,10))
	    print(self.model.classification_dictionary.weight[{1,{1,10}}]:unfold(1,10,10))
	 end
	 internal_counter = internal_counter + 1

	 found_a_nan = false
	 output:apply(find_nans)
	 if found_a_nan then
	    print('outputs')
	    print(output:unfold(1,10,10))
	    print(self.model.encoding_feature_extraction_dictionary.output:unfold(1,10,10))
	    print(self.model.encoding_pooling_dictionary.output:unfold(1,10,10))
	    print(self.model.ista_sparsifying_loss_seq.output[1]:unfold(1,10,10))
	    print(self.model.pooling_seq.output[1]:unfold(1,10,10)) -- one nan is present
	    print(self.model.pooling_L2_loss_seq.output[1]:unfold(1,10,10))
	    print(self.model.pooling_sparsifying_loss_seq.output[1]:unfold(1,10,10))
	    print(self.model.classification_dictionary.output:unfold(1,10,10))
	    
	    print('weights')
	    print(self.model.encoding_feature_extraction_dictionary.weight[{1,{1,10}}]:unfold(1,10,10))
	    print(self.model.explaining_away.weight[{1,{1,10}}]:unfold(1,10,10))
	    print(self.model.shrink.shrink_val[{{1,10}}]:unfold(1,10,10))
	    print(self.model.encoding_pooling_dictionary.weight[{1,{1,10}}]:unfold(1,10,10))
	    print(self.model.classification_dictionary.weight)
	    io.read()
	 end
	 --]]
	    
