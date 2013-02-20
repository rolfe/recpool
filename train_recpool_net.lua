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

local check_for_nans
local output_gradient_magnitudes

function RecPoolTrainer:__init(model, new_opt, layered_lambdas, track_criteria_outputs, receptive_field_builder)
   self.layered_lambdas = layered_lambdas
   self.track_criteria_outputs = track_criteria_outputs or false
   self.receptive_field_builder = receptive_field_builder -- if nil, then receptive fields are not built

   self:reset_options(new_opt)
   
   -- allowed output classes
   if new_opt.num_classification_categories == 10 then
      self.classes = {'1','2','3','4','5','6','7','8','9','0'}
   elseif new_opt.num_classification_categories == 2 then
      self.classes = {'right', 'left'}
   else
      error('unexpected number of classification categories: ' .. new_opt.num_classification_categories '; expected 10 or 2')
   end
   
   -- This matrix records the current confusion across classes
   self.confusion = optim.ConfusionMatrix(self.classes)
   self.current_performance = 0
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
      -- this must *NOT* be called twice; each call allocates new storage and unlinks the modules from the old storage(s)
      self.flattened_parameters,self.flattened_grad_parameters = model:getParameters()  
   else
      error('RecPoolTrainer requires a model')
   end
   self.model = model

   self.minibatch_inputs = torch.Tensor() -- these allow communication between the train function and the feval closure
   self.minibatch_targets = torch.Tensor()

   -- note that feval takes only current_params as input, whereas make_feval takes self as input; the self provided to make_feval is accessible to feval through the closure
   self.feval = self:make_feval() 
   self.epoch = 0
end

function RecPoolTrainer:reset_options(new_opt)
   -- set default options
   if not new_opt then
      new_opt = {}
   end
   self.opt = {}

   -- only copy the desired components from new_opt, and initialize any unspecified components to sensible defaults
   self.opt.log_directory = new_opt.log_directory or 'recpool_results' -- subdirectory in which to save/log experiments
   self.opt.visualize = new_opt.visualize or false -- visualize input data and weights during training
   self.opt.plot = new_opt.plot or false -- live plot
   self.opt.optimization = new_opt.optimization or 'SGD' -- optimization method: SGD | ASGD | CG | LBFGS
   self.opt.learning_rate = new_opt.learning_rate or 1e-3 -- learning rate at t=0
   self.opt.learning_rate_decay = new_opt.learning_rate_decay or 5e-7 -- should be adjusted based upon minibatch size in run_recpool_net
   self.opt.init_eval_counter = new_opt.init_eval_counter or 0
   self.opt.batch_size = new_opt.batch_size or 0 -- mini-batch size (0 = pure stochastic)
   self.opt.weight_decay = new_opt.weight_decay or 0 -- weight decay (SGD only)
   self.opt.L1_weight_decay = new_opt.L1_weight_decay or 0 -- weight decay (SGD only)
   self.opt.momentum = new_opt.momentum or 0 -- momentum (SGD only)
   self.opt.t0 = new_opt.t0 or 1 -- start averaging at t0 (ASGD only), where t0 is measured ASGD calls
   self.opt.max_iter = new_opt.max_iter or 2 -- maximum nb of iterations for CG and LBFGS
end

function RecPoolTrainer:reset_learning_rate(new_learning_rate)
   self.opt.learning_rate = new_learning_rate
   print('Resetting learning rate to ' .. new_learning_rate)
end

function RecPoolTrainer:get_flattened_parameters() -- flattened_parameters are more sensibly handled by the model, rather than the trainer
   return self.flattened_parameters
end

function RecPoolTrainer:get_output_flattened_parameters() -- flattened_parameters are more sensibly handled by the model, rather than the trainer
   if (self.opt.optimization == 'ASGD') and self.average_parameters then
      print('using averaged parameters')
      return self.average_parameters
   else
      return self.flattened_parameters
   end
end


-- create closure to evaluate f(X) and df/dX; the closure is necessary so minibatch_inputs and self.minibatch_targets are correct.  
-- self is provided by the column notation.  Thereafter, feval can be called without self as an argument, and the closure provides access to the (implicit) self argument of make_feval
function RecPoolTrainer:make_feval()
   --[[
   local internal_counter = 1
   --]]

   --[[ DEBUG CODE to check if minibatches produce the same gradients as pure SGD
   local batched_update = torch.Tensor()
   local batched_update_counter = 0
   --]]
      
   local feval = function(current_params)
      -- enforce all constraints on parameters, since the parameters are updated manually, rather than through updateParameters as generally expected
      if (self.opt.optimization ~= 'SGD') and (self.opt.optimization ~= 'ASGD') then -- if we only do one feval call per parameter update, then it is safe to repair once after the update
	 print('consider the need to repair on each iteration')
	 self.model:repair() -- THIS IS INEFFICIENT!  THIS DOUBLES THE TIME REQUIRED PER ITERATION!  THE COMPONENT OPERATIONS SHOULD BE IMPLEMENTED IN C!!!
      end

      -- get new parameters
      if current_params ~= self.flattened_parameters then 
	 self.flattened_parameters:copy(current_params)
	 print('copying parameters in feval') -- does this ever actually run?
      end
      
      -- Reset gradients.  This is more efficient than self.model:zeroGradParameters(), since gradParameters has all gradients flattened into a single storage, viewed by the many parameter tensors.  As a result, when parameters are shared by multiple modules, they are only zeroed once by this procedure.
      self.flattened_grad_parameters:zero() 
      
      -- total_err is the average of the error over the entire minibatch
      local total_err = 0
      
      -- evaluate function for complete minibatch
      --for i = 1,#self.minibatch_inputs do
      -- estimate total_err
      self.model:set_target(self.minibatch_targets)
      local err = self.model:updateOutput(self.minibatch_inputs)
      local output = self.model:get_classifier_output() -- while the model is a nn.Sequential, it terminates in a set of criteria
      total_err = total_err + err[1] -- the err returned by updateOutput is a tensor with one element, to maintain compatibility with ModfifiedJacobian
      
      --check_for_nans(self, output, 'outputs')
      
      -- estimate the gradient of the error with respect to the parameters: d total_err / dW
      self.model:updateGradInput(self.minibatch_inputs) -- gradOutput is not required, since all computation streams terminate in a criterion; implicitly pass nil
      self.model:accGradParameters(self.minibatch_inputs)
      
      -- update the confusion matrix.  This keeps track of the predicted output (maximum output conditional posterior probability) for each true output class
      if (self.minibatch_inputs:nDimension() == 1) and (type(self.minibatch_targets) == 'number') then -- minibatch contains a single element, so its components are of reduced dimensionality
	 self.confusion:add(output, self.minibatch_targets) 
      elseif (self.minibatch_inputs:nDimension() == 2) and (self.minibatch_targets:nDimension() == 1) then -- minibatch contains many elements, over which we must iterate
	 for j = 1,self.minibatch_inputs:size(1) do
	    self.confusion:add(output:select(1,j), self.minibatch_targets[j]) 
	 end
      else
	 error('dimensions of minibatch_inputs and minibatch_targets were not consistent')
      end
      
      -- track the evolution of sparsity and reconstruction errors
      if self.track_criteria_outputs then
	 for j = 1,#(self.model.criteria_list.criteria) do
	    if self.model.criteria_list.criteria[j].output then -- this need not be defined if we've disabled pooling or other layers
	       self.loss_hist[j] = self.model.criteria_list.criteria[j].output + self.loss_hist[j]
	       --print(self.model.criteria_list.names[j], self.model.criteria_list.criteria[j].gradInput)
	       if type(self.model.criteria_list.criteria[j].gradInput) == 'table' then
		  for k = 1,#self.model.criteria_list.criteria[j].gradInput do
		     self.grad_loss_hist[j] = self.model.criteria_list.criteria[j].gradInput[k]:norm() + self.grad_loss_hist[j]
		  end
	       else
		  self.grad_loss_hist[j] = self.model.criteria_list.criteria[j].gradInput:norm() + self.grad_loss_hist[j]
	       end
	    end -- if critera output exists
	 end -- for all criteria
      end --if track_criteria_outputs
      --end

      if self.receptive_field_builder then
	 self.receptive_field_builder:accumulate_shrink_weighted_inputs(self.minibatch_inputs, self.model.layers[1].module_list.shrink, self.model.layers[1].module_list.shrink_copies, 
									self.minibatch_targets)
      end

      -- normalize gradients and f(X)
      if self.minibatch_inputs:nDimension() == 2 then
	 -- we do not average gradients across a minibatch; rather we simply add them together.  This saves us from having to scale up the learning rate in proportion to the size of the minibatch
	 --self.flattened_grad_parameters:div(self.minibatch_inputs:size(1)) -- minibatches are stored along the rows; each row is a different minibatch element
	 total_err = total_err / self.minibatch_inputs:size(1)
      end

      --[[  DEBUG CODE to check if minibatches produce the same gradients as pure SGD
      if self.config.learningRate then
	 batched_update:resizeAs(self.flattened_grad_parameters)
	 if batched_update_counter == 0 then
	    batched_update:zero()
	 end
	 --batched_update:add(self.config.learningRate, self.flattened_grad_parameters)
	 batched_update:add(self.flattened_grad_parameters)
	 batched_update_counter = batched_update_counter + 1
	 if batched_update_counter == 10 then
	    batched_update_counter = 0
	    print('batched update magnitude is ' .. torch.norm(batched_update))
	 end
         --print('update magnitude is ' .. torch.norm(torch.mul(self.flattened_grad_parameters, self.config.learningRate)))
	 print('update magnitude is ' .. torch.norm(self.flattened_grad_parameters))
      end
      --]]

       
      --check_for_nans(self, self.flattened_grad_parameters, 'gradParameters')
      -- return f and df/dX
      return total_err, self.flattened_grad_parameters
   end
   
   return feval
end


function RecPoolTrainer:train(train_data, epoch_type) 
   -- epoch_type == 'validation' if we're just testing the network on the validation data, in which case nothing should be altered (in particular, learning should not be performed
   -- epoch_type == 'display' if we're generating figures, in which case we want to keep the order consistent
   if not(epoch_type) then
      self.epoch = self.epoch + 1
   end

   local this_epoch_batch_size = ((epoch_type == 'validation') and self.opt.test_batch_size) or self.opt.batch_size
   
   -- local vars
   local time = sys.clock()
   
   -- shuffle at each epoch
   local shuffle 
   if epoch_type == 'display' then
      shuffle = torch.range(1,train_data:nExample())
   else
      shuffle = torch.randperm(train_data:nExample()) --was trsize
   end

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. self.epoch .. ' [batch_size = ' .. this_epoch_batch_size .. ']')

   for t = 1, train_data:nExample(), math.max(1, this_epoch_batch_size) do
      -- disp progress
      if (t % 200 == 1) or (t == train_data:nExample()) then
	 xlua.progress(t, train_data:nExample())
      end
      
      -- create mini batch.  The minibatch_inputs and minibatch_targets elements of a RecPoolTrainer are viewed directly by the feval made by make_feval()
      -- previously, minibatches consisted of a table, the elements of which were the components of the minibatch.  However, this is not directly compatible with the torch facilities for efficiently processing minibatches.  Now, minibatches consist of tensors, with the elements along the rows.  An exception is made for minibatches of size one, which use tensors (or numbers) of reduced dimensionality, to take advantage of more efficient matrix-vector rather than matrix-matrix calculations.
      if this_epoch_batch_size == 0 then
	 self.minibatch_inputs = train_data.data[shuffle[t]]:double() -- This doesn't copy memory if the type is already correct
	 self.minibatch_targets = train_data.labels[shuffle[t]]
      else -- unlike with minibatches of size 1, we *ALWAYS* copy the data in forming larger minibatches.  This is necessary because tensors consist of regular strides within contiguous blocks of memory, rather than arbitrary arrays of pointers.
	 self.minibatch_inputs:resize(this_epoch_batch_size, train_data:dataSize())
	 if train_data:labelSize() == 1 then
	    self.minibatch_targets:resize(this_epoch_batch_size)
	 else
	    self.minibatch_targets:resize(this_epoch_batch_size, train_data:labelSize())
	 end
	 
	 -- load new samples for the batch
	 for i = 1,this_epoch_batch_size do 
	    --ensure that batches are full by wrapping around to the beginning if necessary; since the order of the dataset is reshuffled on each epoch, this shouldn't cause a big problem
	    local shuffle_index = ((t+i-2) % train_data:nExample()) + 1
	    self.minibatch_inputs:select(1,i):copy(train_data.data[shuffle[shuffle_index]]:double())
	    if train_data:labelSize() == 1 then
	       self.minibatch_targets[i] = train_data.labels[shuffle[shuffle_index]]
	    else
	       self.minibatch_targets:select(1,i):copy(train_data.labels[shuffle[shuffle_index]])
	    end
	 end
      end

      
      -- optimize on current mini-batch
      if ((epoch_type == 'validation') or (epoch_type == 'display')) then
	 self.feval(self.flattened_parameters) -- just run the network on the minibatch_inputs and minibatch_targets to generate the confusion matrix, without doing any training
	 -- this is necessary since some of these value are accessed by run_recpool_net.  Ideally, we should eliminate this break in the abstraction barrier
	 self.config = self.config or {evalCounter = self.opt.init_eval_counter or 0, 
				       learningRate = self.opt.learning_rate,
				       weightDecay = self.opt.weight_decay,
				       L1weightDecay = self.opt.L1_weight_decay,
				       momentum = self.opt.momentum,
				       learningRateDecay = self.opt.learning_rate_decay}
      elseif self.opt.optimization == 'CG' then
         self.config = self.config or {maxIter = self.opt.max_iter}
         optim.cg(self.feval, self.flattened_parameters, self.config)
	 
      elseif self.opt.optimization == 'LBFGS' then
         self.config = self.config or {learningRate = self.opt.learning_rate,
                             maxIter = self.opt.max_iter,
                             nCorrection = 10}
         optim.lbfgs(self.feval, self.flattened_parameters, self.config)
	 
      elseif self.opt.optimization == 'SGD' then
         self.config = self.config or {evalCounter = self.opt.init_eval_counter or 0,
				       learningRate = self.opt.learning_rate,
				       weightDecay = self.opt.weight_decay,
				       L1weightDecay = self.opt.L1_weight_decay,
				       momentum = self.opt.momentum,
				       learningRateDecay = self.opt.learning_rate_decay} -- 5e-7
	 self.config.learningRate = self.opt.learning_rate -- make sure that the sgd learning rate reflects any resets
         optim.sgd_decayed_weight_decay(self.feval, self.flattened_parameters, self.config)
	 
      elseif self.opt.optimization == 'ASGD' then
         self.config = self.config or {t = self.opt.init_eval_counter or 0,
				       eta0 = self.opt.learning_rate,
				       lambda = self.opt.learning_rate_decay / self.opt.learning_rate, -- matches the decay to that in SGD
				       t0 = self.opt.t0} -- measured in calls to ASGD
	 self.config.eta0 = self.opt.learning_rate -- make sure that asgd learning rate reflects any resets
	 self.config.lambda = self.opt.learning_rate_decay / self.opt.learning_rate
         _,_,self.average_parameters = optim.asgd_no_weight_decay(self.feval, self.flattened_parameters, self.config)
	 
      else
         error('unknown optimization method')
      end

      -- repair the parameters one final time
      if not((epoch_type == 'validation') or (epoch_type == 'display')) then self.model:repair() end -- EFFICIENCY NOTE: Keep in mind that this is the most time consuming part of the operation!!!
   end -- loop over the current epoch
   
   -- time taken for the current epoch (each call to train() only runs one epoch)
   time = sys.clock() - time
   time = time / train_data:nExample()
   print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')
   
   print(self.confusion) -- print the confusion matrix for the current epoch
   self.current_performance = self.confusion.totalValid * 100 -- extract the current performance from the ConfusionMatrix object

   -- update logger/plot
   self.train_logger:add{['% mean class accuracy (train set)'] = self.confusion.totalValid * 100}
   if self.opt.plot then
      self.train_logger:style{['% mean class accuracy (train set)'] = '-'}
      self.train_logger:plot()
   end

   self.confusion:zero()

   for i = 1,#self.model.criteria_list.criteria do
      if self.track_criteria_outputs then
	 print('Criterion: ' .. self.model.criteria_list.names[i] .. ' = ' .. self.loss_hist[i]/train_data:nExample() .. '; grad = ' .. self.grad_loss_hist[i]/train_data:nExample())
	 self.loss_hist[i] = 0
	 self.grad_loss_hist[i] = 0
      end

      if (self.model.criteria_list.names[i] == 'pooling L2 shrink reconstruction loss') and not(self.model.disable_pooling) then
	 print('performing additional testing on ' .. self.model.criteria_list.names[i] .. ' with value ' .. self.model.criteria_list.criteria[i].output)
	 local alpha = self.layered_lambdas[1].pooling_L2_shrink_position_unit_lambda / self.layered_lambdas[1].pooling_L2_shrink_reconstruction_lambda
	 local shrink_output = self.model.layers[1].module_list.shrink_copies[#self.model.layers[1].module_list.shrink_copies].output
	 local theoretical_shrink_reconstruction_loss = alpha^2 * math.pow(torch.norm(torch.cdiv(shrink_output, 
										torch.add(torch.pow(self.model.layers[1].module_list.decoding_pooling_dictionary.output, 2), alpha))), 2)
	 --print('dividing ', torch.mul(self.model.layers[1].module_list.shrink_copies[#self.model.layers[i].module_list.shrink_copies].output, alpha):unfold(1,10,10))
	 --print('by ', torch.add(torch.pow(self.model.layers[1].module_list.decoding_pooling_dictionary.output, 2), alpha):unfold(1,10,10))
	 local careful_reconstruction_loss = math.pow(torch.norm(torch.add(shrink_output, -1,
									   torch.cdiv(torch.cmul(shrink_output, torch.pow(self.model.layers[1].module_list.decoding_pooling_dictionary.output, 2)), 
										      torch.add(torch.pow(self.model.layers[1].module_list.decoding_pooling_dictionary.output, 2), alpha)))), 2)
	 
	 local theoretical_shrink_position_loss = math.pow(torch.norm(torch.cdiv(torch.cmul(shrink_output, self.model.layers[1].module_list.decoding_pooling_dictionary.output), 
										 torch.add(torch.pow(self.model.layers[1].module_list.decoding_pooling_dictionary.output, 2), alpha))), 2)
	 
	 local combined_loss_careful = self.layered_lambdas[1].pooling_L2_shrink_reconstruction_lambda * theoretical_shrink_reconstruction_loss + 
	    self.layered_lambdas[1].pooling_L2_shrink_position_unit_lambda * theoretical_shrink_position_loss
	 
	 -- this should not equal the exact combined loss, since (a + b)^2 ~= a^2 + b^2!!!
	 local combined_loss = self.layered_lambdas[1].pooling_L2_shrink_position_unit_lambda * 
	    math.pow(torch.norm(torch.cdiv(torch.cmul(shrink_output, 
						      torch.add(self.model.layers[1].module_list.decoding_pooling_dictionary.output, math.sqrt(alpha))), 
					      torch.add(torch.pow(self.model.layers[1].module_list.decoding_pooling_dictionary.output, 2), alpha))), 2)
	 local better_combined_loss = self.layered_lambdas[1].pooling_L2_shrink_position_unit_lambda * 
	    torch.sum(torch.cdiv(torch.pow(shrink_output, 2), 
				 torch.add(torch.pow(self.model.layers[1].module_list.decoding_pooling_dictionary.output, 2), alpha)))
	    -- THIS IS NO LONGER CORRECT, SINCE L1 SPARSITY AND L2 RECONSTRUCTION CRITERIA ARE REPEATED IN LISTA STACK
	 local exact_combined_loss = self.model.criteria_list.criteria[4].output + self.model.criteria_list.criteria[6].output -- this already includes the lambdas
	 print('*combined ratio is ' .. combined_loss / exact_combined_loss .. ' better: ' .. better_combined_loss / exact_combined_loss .. ' careful: ' .. combined_loss_careful / exact_combined_loss)
	    

	 print('shrink reconstruction ratio is ' .. (self.layered_lambdas[1].pooling_L2_shrink_reconstruction_lambda * theoretical_shrink_reconstruction_loss) / self.model.criteria_list.criteria[i].output .. ' careful version ' .. (self.layered_lambdas[1].pooling_L2_shrink_reconstruction_lambda * careful_reconstruction_loss) / self.model.criteria_list.criteria[i].output .. ' with criteria output ' .. self.model.criteria_list.criteria[i].output)
	 
	 print('shrink position ratio is ' .. (self.layered_lambdas[1].pooling_L2_shrink_position_unit_lambda * theoretical_shrink_position_loss) / self.model.criteria_list.criteria[6].output)


	 
	 -- version using alternative position loss ||sqrt(P*s)*x||^2
	 local theoretical_alt_shrink_reconstruction_loss = alpha^2 * math.pow(torch.norm(torch.cdiv(shrink_output, 
										torch.add(self.model.layers[1].module_list.decoding_pooling_dictionary.output, alpha))), 2)
	 local careful_alt_reconstruction_loss = math.pow(torch.norm(torch.add(shrink_output, -1,
								    torch.cdiv(torch.cmul(shrink_output, self.model.layers[1].module_list.decoding_pooling_dictionary.output), 
									       torch.add(self.model.layers[1].module_list.decoding_pooling_dictionary.output, alpha)))), 2)
	 local theoretical_alt_shrink_position_loss = math.pow(torch.norm(torch.cdiv(torch.cmul(shrink_output, torch.pow(self.model.layers[1].module_list.decoding_pooling_dictionary.output, 0.5)), 
										     torch.add(self.model.layers[1].module_list.decoding_pooling_dictionary.output, alpha))), 2)
	 local better_alt_combined_loss = self.layered_lambdas[1].pooling_L2_shrink_position_unit_lambda * 
	    torch.sum(torch.cdiv(torch.pow(shrink_output, 2), 
				 torch.add(self.model.layers[1].module_list.decoding_pooling_dictionary.output, alpha)))

	 print('*alt combined ratio is ' .. better_alt_combined_loss / exact_combined_loss)
	 print('alt shrink reconstruction ratio is ' .. (self.layered_lambdas[1].pooling_L2_shrink_reconstruction_lambda * theoretical_alt_shrink_reconstruction_loss) / self.model.criteria_list.criteria[i].output .. ' careful version ' .. (self.layered_lambdas[1].pooling_L2_shrink_reconstruction_lambda * careful_alt_reconstruction_loss) / self.model.criteria_list.criteria[i].output .. ' with criteria output ' .. self.model.criteria_list.criteria[i].output)
	 
	 print('alt shrink position ratio is ' .. (self.layered_lambdas[1].pooling_L2_shrink_position_unit_lambda * theoretical_alt_shrink_position_loss) / self.model.criteria_list.criteria[6].output)

	 

      end  -- pooling is enabled
   end	-- loop over all criteria 

   if not(self.model.disable_pooling) then
      local alpha = (self.layered_lambdas[1].pooling_L2_shrink_position_unit_lambda + self.layered_lambdas[1].pooling_L2_orig_position_unit_lambda) /
	 (self.layered_lambdas[1].pooling_L2_shrink_reconstruction_lambda + self.layered_lambdas[1].pooling_L2_orig_reconstruction_lambda)
      
      local theoretical_orig_position_loss = math.pow(torch.norm(torch.cdiv(torch.cmul(self.model.layers[1].module_list.decoding_feature_extraction_dictionary_transpose.output,
										       self.model.layers[1].module_list.decoding_pooling_dictionary.output), 
									    torch.add(torch.pow(self.model.layers[1].module_list.decoding_pooling_dictionary.output, 2), alpha))), 2)
      --[[
      local theoretical_orig_reconstruction_loss = math.pow(torch.norm(torch.add(INPUT, -1,
										 torch.cdiv(torch.cmul(shrink_output, torch.pow(self.model.layers[1].module_list.decoding_pooling_dictionary.output, 2)),
											    torch.add(torch.pow(self.model.layers[1].module_list.decoding_pooling_dictionary.output, 2), alpha)))), 2)
      --]]

      print('orig position ratio is ' .. (self.layered_lambdas[1].pooling_L2_orig_position_unit_lambda * theoretical_orig_position_loss) / self.model.criteria_list.criteria[7].output)
      --print('orig reconstruction ratio is ' .. (self.layered_lambdas[1].pooling_L2_orig_reconstruction_lambda * theoretical_orig_reconstruction_loss) / self.model.criteria_list.criteria[5].output)
   end 
      
   print('Explaining away diag is', torch.diag(self.model.layers[1].module_list.explaining_away.weight):unfold(1,10,10))
   

   for i = 1,#self.model.layers do
      --print('feature reconstruction ', self.model.layers[i].module_list.decoding_feature_extraction_dictionary.output:unfold(1,10,10))
      --print('shrink magnitude ', self.model.layers[i].module_list.shrink.shrink_val:norm())
      --print('all shrink', self.model.layers[i].module_list.shrink.shrink_val:unfold(1,10,10))
      --print('explaining away diag', torch.diag(self.model.layers[i].module_list.explaining_away.weight):unfold(1,10,10))
      local single_shrink_output, single_offset_shrink_output, single_pooling_output
      if this_epoch_batch_size == 0 then
	 single_shrink_output = self.model.layers[i].module_list.shrink_copies[#self.model.layers[i].module_list.shrink_copies].output
	 if self.model.layers[i].module_list.offset_shrink then
	    single_offset_shrink_output = self.model.layers[i].module_list.offset_shrink.output
	 end
	 if not(self.model.disable_pooling) then 
	    single_pooling_output = self.model.layers[i].debug_module_list.pooling_seq.output[1]
	 end
      else -- only display the first element of the minibatch
	 single_shrink_output = self.model.layers[i].module_list.shrink_copies[#self.model.layers[i].module_list.shrink_copies].output:select(1,1)
	 if self.model.layers[i].module_list.offset_shrink then
	    single_offset_shrink_output = self.model.layers[i].module_list.offset_shrink.output:select(1,1)
	 end
	 if not(self.model.disable_pooling) then 
	    single_pooling_output = self.model.layers[i].debug_module_list.pooling_seq.output[1]:select(1,1)
	 end
      end

      print('final shrink output', single_shrink_output:unfold(1,10,10))
      if single_offset_shrink_output then 
	 print('offset shrink output', single_offset_shrink_output:unfold(1,10,10))
      end
      if not(self.model.disable_pooling) then -- this may not be defined if we've disabled pooling
	 print('pooling reconstruction', self.model.layers[i].module_list.decoding_pooling_dictionary.output:unfold(1,10,10))
      end
      -- these two outputs are from the middle of the processing chain, rather than the parameterized modules
      --print('pooling position units', self.model.layers[i].debug_module_list.compute_shrink_position_units.output:unfold(1,10,10))
      if not(self.model.disable_pooling) then -- this may not be defined if we've disabled pooling
	 print('pooling output', single_pooling_output:unfold(1,10,10))
      end
      -- since the sparsifying modules can be parameterized by lagrange multipliers, they are in the main module list
      if self.model.layers[i].module_list.feature_extraction_sparsifying_module.weight then
	 print('feature extraction L1', self.model.layers[i].module_list.feature_extraction_sparsifying_module.weight:unfold(1,10,10))
      end
      if self.model.layers[i].module_list.pooling_sparsifying_module.weight then
	 print('pooling L1', self.model.layers[i].module_list.pooling_sparsifying_module.weight:unfold(1,10,10))
      end
      if self.model.layers[i].module_list.mask_sparsifying_module.weight then
	 print('mask L1', self.model.layers[i].module_list.mask_sparsifying_module.weight:unfold(1,10,10))
      end
      --print('normalized output', self.model.layers[i].debug_module_list.normalize_output.output[1]:unfold(1,10,10))

      --[[
      local m = self.model.layers[i].module_list.decoding_feature_extraction_dictionary.weight
      --local m = self.model.layers[i].module_list.decoding_pooling_dictionary.weight
      local norms = torch.Tensor(m:size(2))
      for j = 1,m:size(2) do
	 norms[j] = m:select(2,j):norm()
      end
      print('FE dec col norms are ', norms:unfold(1,10,10))

      print('FE dec matrix is ', m)
      --]]

      --print('dec matrix is', self.model.layers[1].module_list.decoding_feature_extraction_dictionary.weight)
      --print('enc matrix is', self.model.layers[1].module_list.encoding_feature_extraction_dictionary.weight:t())
      --print('enc bias is', self.model.layers[1].module_list.encoding_feature_extraction_dictionary.bias:unfold(1,10,10))

      local m = self.model.layers[i].module_list.encoding_feature_extraction_dictionary.weight
      --local m = self.model.layers[i].module_list.encoding_pooling_dictionary.weight
      local norms = torch.Tensor(m:size(1))
      for j = 1,m:size(1) do
	 norms[j] = m:select(1,j):norm()
      end
      print('FE row norms are ', norms:unfold(1,10,10))

      --print('FE biases are ', self.model.layers[i].module_list.encoding_feature_extraction_dictionary.bias:unfold(1,10,10))

      --[[
      local m = self.model.layers[i].module_list.encoding_pooling_dictionary.weight
      local norms = torch.Tensor(m:size(1))
      for j = 1,m:size(1) do
	 norms[j] = torch.pow(m:select(1,j), 2):norm()
      end
      print('P row norms are ', norms:unfold(1,10,10))
      --]]
	 
      if self.model.layers[i].debug_module_list.normalize_pooled_output then
	 print('Pooling normalization is ', self.model.layers[i].debug_module_list.normalize_pooled_output.norm)
      end
     

      ---[[
      local m = self.model.layers[i].module_list.explaining_away.weight
      local norms = torch.Tensor(m:size(1))
      for j = 1,m:size(1) do
	 norms[j] = m:select(1,j):norm()
      end
      print('EXP row norms are ', norms:unfold(1,10,10))

      --print(m:select(1,101):unfold(1,10,10))
      --]]

      

      --[[
      print('shrink values', self.model.layers[i].module_list.shrink.shrink_val:unfold(1,10,10))
      --print('shrink values', torch.add(self.model.layers[i].module_list.shrink.shrink_val, -1e-5):unfold(1,10,10))
      --print('negative_shrink values', torch.add(self.model.layers[i].module_list.shrink.negative_shrink_val, 1e-5):unfold(1,10,10))
      -- display filters!  Also display reconstructions minus originals, so we can see how the reconstructions improve with training!
      -- check that without regularization, filters are meaningless.  Confirm that trainable pooling has an effect on the pooled filters.

      print('encoding_feature_extraction_dictionary output')
      print(self.model.layers[i].module_list.encoding_feature_extraction_dictionary.output:unfold(1,10,10))

      print('explaining_away outputs')
      local desired_size = 20 --self.model.layers[i].module_list.decoding_feature_extraction_dictionary.output:size(1)
      local explaining_away_output_tensor = torch.Tensor(desired_size, #self.model.layers[i].module_list.explaining_away_copies)
      for j = 1,#self.model.layers[i].module_list.explaining_away_copies do 	 
	 explaining_away_output_tensor:select(2,j):copy(self.model.layers[i].module_list.explaining_away_copies[j].output:narrow(1,1,desired_size))
      end
      print(explaining_away_output_tensor)

      print('raw shrink outputs')
      local desired_size = 20 --self.model.layers[i].module_list.decoding_feature_extraction_dictionary.output:size(1)
      local shrink_output_tensor = torch.Tensor(desired_size, #self.model.layers[i].module_list.shrink_copies)
      for j = 1,#self.model.layers[i].module_list.shrink_copies do 	 
	 shrink_output_tensor:select(2,j):copy(self.model.layers[i].module_list.shrink_copies[j].output:narrow(1,1,desired_size))
      end
      print(shrink_output_tensor)

      print('reconstructed shrink outputs')
      local desired_size = 200 --self.model.layers[i].module_list.decoding_feature_extraction_dictionary.output:size(1)
      local shrink_output_tensor = torch.Tensor(desired_size, #self.model.layers[i].module_list.shrink_copies)
      for j = 1,#self.model.layers[i].module_list.shrink_copies do 	 
	 shrink_output_tensor:select(2,j):copy(self.model.layers[i].module_list.decoding_feature_extraction_dictionary:updateOutput(
						  self.model.layers[i].module_list.shrink_copies[j].output):narrow(1,1,desired_size))
      end
      print(shrink_output_tensor)
      --]]
      
      local plot_recs = false
      if plot_recs then
	 if i == 1 then
	    plot_reconstructions(self.opt, self.minibatch_inputs, self.model.layers[i].module_list.decoding_feature_extraction_dictionary.output)
	 else
	    plot_reconstructions(self.opt, self.model.layers[i-1].debug_module_list.normalize_output.output[1], self.model.layers[i].module_list.decoding_feature_extraction_dictionary.output)
	 end
      end
   end -- loop over layers

   local m = self.model.module_list.classification_dictionary.weight
   local norms = torch.Tensor(m:size(1))
   local c_size = m:size(1)
   for j = 1,m:size(1) do
      norms[j] = m:select(1,j):norm()
   end
   print('C row norms are ', norms:unfold(1,c_size,c_size)) -- unfold so the tensor is presented horizontally rather than vertically
   
   if self.model.module_list.logsoftmax.output:nElement() > 0 then -- make sure the logsoftmax is used before trying to display its output
      if this_epoch_batch_size == 0 then
	 print('logsoftmax output is ', self.model.module_list.logsoftmax.output:unfold(1,c_size,c_size))
	 print('target is ' .. self.model.current_target)
      else
	 print('logsoftmax output is ', self.model.module_list.logsoftmax.output:select(1,1):unfold(1,c_size,c_size))
	 print('target is ', self.model.current_target[1])
      end
   end
   print('classification bias is ', self.model.module_list.classification_dictionary.bias:unfold(1,c_size,c_size))

   output_gradient_magnitudes(self)

   local index_list = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
   --local index_list = {11, 12, 13, 14, 15, 16, 4, 7, 8, 10, 42, 52, 63, 64, 67, 78}
   --local index_list = {1, 2, 3, 11, 12, 13, 4, 9, 21, 32, 72, 83, 88}
   --local index_list = {32, 34, 41, 58, 69, 70, 91, 103, 114, 121, 123, 138, 171, 201, 203, 213, 217, 238, 244, 290, 304, 327, 1, 2, 3, 4, 5, 6, 7, 8} -- for 11/30 400 units
   --local index_list = {30, 47, 53, 55, 77, 119, 172, 192, 196, 199, 206, 209, 232, 241, 255, 269, 290, 302, 303, 305, 311, 313, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10} -- for 12/12 400 units manual
   --local index_list = {3, 7, 11, 16, 51, 77, 88, 107, 109, 127, 129, 1, 2, 4, 5, 8} -- 12/13 400 units 11 ista iterations 5e-3 training rate
   local num_shrink_output_tensor_elements = #index_list -- self.model.layers[1].module_list.shrink.output:size(1)
   local shrink_output_tensor = torch.Tensor(num_shrink_output_tensor_elements, 1 + #self.model.layers[1].module_list.shrink_copies)

   for j = 1,#index_list do
      -- automatically choose whether to use 1D or 2D indexing into shrink.output, depending upon the size of the minibatch
      shrink_output_tensor[{j, 1}] = self.model.layers[1].module_list.shrink.output[((this_epoch_batch_size == 0) and {index_list[j]}) or {1,index_list[j]}]
   end
   
   for i = 1,#self.model.layers[1].module_list.shrink_copies do
      --shrink_output_tensor:select(2,i):copy(self.model.layers[1].module_list.shrink_copies[i].output)
      for j = 1,#index_list do
	 -- automatically choose whether to use 1D or 2D indexing into shrink_copies[i].output, depending upon the size of the minibatch
	 shrink_output_tensor[{j, i+1}] = self.model.layers[1].module_list.shrink_copies[i].output[((this_epoch_batch_size == 0) and {index_list[j]}) or {1,index_list[j]}]
      end
   end
   print('evolution of selected shrink elements', shrink_output_tensor)

   local exp_away_grad_mags = torch.Tensor(#self.model.layers[1].module_list.explaining_away_copies)
   for j = 1,#self.model.layers[1].module_list.explaining_away_copies do 	 
      exp_away_grad_mags[j] = self.model.layers[1].module_list.explaining_away_copies[j].gradInput:norm()
   end
   print('explaining away grad mags are ', exp_away_grad_mags)


   --[[
   -- save/log current net
   local filename = paths.concat(self.opt.log_directory, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, self.model)
   --]]

end


function output_gradient_magnitudes(self)
   for i = 1,#self.model.layers do
      if self.model.layers[i].module_list.shrink.grad_shrink_val then  -- THIS IS A TOTAL HACK!!!
	 print('layer ' .. i, 'encoding FE dict', self.model.layers[i].module_list.encoding_feature_extraction_dictionary.gradWeight:norm(), 
	       'decoding FE dict', self.model.layers[i].module_list.decoding_feature_extraction_dictionary.gradWeight:norm(),
	       'shrink', self.model.layers[i].module_list.shrink.grad_shrink_val:norm(), 'explaining away', self.model.layers[i].module_list.explaining_away.gradWeight:norm(),
	       'encoding P dict', self.model.layers[i].module_list.encoding_pooling_dictionary.gradWeight:norm(), 
	       'decoding P dict', self.model.layers[i].module_list.decoding_pooling_dictionary.gradWeight:norm())
      else
	 local agg_norm = 1
	 print('layer ' .. i, 'encoding FE dict', self.model.layers[i].module_list.encoding_feature_extraction_dictionary.gradWeight:norm(agg_norm) /
	       self.model.layers[i].module_list.encoding_feature_extraction_dictionary.gradWeight:nElement(), 
	       'decoding FE dict', self.model.layers[i].module_list.decoding_feature_extraction_dictionary.gradWeight:norm(agg_norm) / 
		  self.model.layers[i].module_list.decoding_feature_extraction_dictionary.gradWeight:nElement(),
	       'explaining away', self.model.layers[i].module_list.explaining_away.gradWeight:norm(agg_norm) / 
		  self.model.layers[i].module_list.explaining_away.gradWeight:nElement())
	 if not(self.model.disable_pooling) then
	    print('encoding P dict', self.model.layers[i].module_list.encoding_pooling_dictionary.gradWeight:norm(agg_norm) / 
		  self.model.layers[i].module_list.encoding_pooling_dictionary.gradWeight:nElement(), 
		  'decoding P dict', self.model.layers[i].module_list.decoding_pooling_dictionary.gradWeight:norm(agg_norm) / 
		     self.model.layers[i].module_list.decoding_pooling_dictionary.gradWeight:nElement())
	 end
      end
   end
   print('classification layer', 'class dict', self.model.module_list.classification_dictionary.gradWeight:norm())

   --print(self.model.layers[1].module_list.encoding_pooling_dictionary.gradWeight:unfold(1,10,10))
   --print(self.model.layers[1].debug_module_list.ista_sparsifying_loss_seq.output[1]:unfold(1,10,10))
   --print(self.model.layers[1].debug_module_list.pooling_L2_loss_seq.gradInput[1]:unfold(1,10,10))
   --print('max value is: ', torch.pow(self.model.layers[1].debug_module_list.ista_sparsifying_loss_seq.output[1], 2):max(), self.model.layers[1].debug_module_list.pooling_L2_loss_seq.gradInput[1]:max())
   
end

function check_for_nans(self, output, name)
   local found_a_nan = false
   
   local function find_nans(x)
      if x ~= x then
	 found_a_nan = true
      end
   end

   local function find_nans_in_table(x)
      for k,v in pairs(x) do
	 found_a_nan = false
	 v:apply(find_nans)
	 if found_a_nan then
	    print('found a nan in entry ' .. k)
	    print(v:unfold(1,10,10))
	 end
      end
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
   --]]
      
   output:apply(find_nans)
   if found_a_nan then
      for i = 1,#self.model.layers do
	 print('checking for nans in ' .. name .. ' layer ' .. i)
	 io.read()
	 print('outputs')
	 --print(output:unfold(1,10,10))
	 print(self.model.layers[i].module_list.encoding_feature_extraction_dictionary.output:unfold(1,10,10))
	 print(self.model.layers[i].module_list.encoding_pooling_dictionary.output:unfold(1,10,10))
	 print(self.model.layers[i].debug_module_list.ista_sparsifying_loss_seq.output[1]:unfold(1,10,10))
	 print(self.model.layers[i].debug_module_list.pooling_seq.output[1]:unfold(1,10,10)) -- one nan is present
	 print(self.model.layers[i].debug_module_list.pooling_L2_loss_seq.output[1]:unfold(1,10,10))
	 print(self.model.layers[i].debug_module_list.pooling_sparsifying_loss_seq.output[1]:unfold(1,10,10))
	 print(self.model.layers[i].module_list.decoding_pooling_dictionary.output:unfold(1,10,10))
      end
      print(self.model.module_list.classification_dictionary.output:unfold(1,10,10))
      io.read()

      for i = 1,#self.model.layers do
	 print('gradInputs test in layer ' .. i)

	 print('shrink_reconstruction')
	 find_nans_in_table(self.model.layers[i].debug_module_list.compute_shrink_reconstruction_loss_seq.gradInput)
	 print('orig_reconstruction')
	 find_nans_in_table(self.model.layers[i].debug_module_list.compute_orig_reconstruction_loss_seq.gradInput)
	 print('shrink position_loss')
	 find_nans_in_table(self.model.layers[i].debug_module_list.compute_shrink_position_loss_seq.gradInput)
	 print('orig position_loss')
	 find_nans_in_table(self.model.layers[i].debug_module_list.compute_orig_position_loss_seq.gradInput)
	 io.read()
	 
	 print('gradInputs second test')
	 print('shrink_rec_numerator')
	 find_nans_in_table(self.model.layers[i].debug_module_list.construct_shrink_rec_numerator_seq.gradInput)
	 --print(self.model.construct_shrink_rec_numerator_seq.output:unfold(1,10,10))
	 print('shink_pos_numerator_seq')
	 find_nans_in_table(self.model.layers[i].debug_module_list.construct_shrink_pos_numerator_seq.gradInput)
	 --print(self.model.construct_shrink_pos_numerator_seq.output:unfold(1,10,10))
	 print('orig_rec_numerator_seq')
	 find_nans_in_table(self.model.layers[i].debug_module_list.construct_orig_rec_numerator_seq.gradInput)
	 --print(self.model.construct_orig_rec_numerator_seq.output:unfold(1,10,10))
	 print('orig_pos_numerator_seq')
	 find_nans_in_table(self.model.layers[i].debug_module_list.construct_orig_pos_numerator_seq.gradInput)
	 --print(self.model.construct_orig_pos_numerator_seq.output:unfold(1,10,10))
	 print('denominator_seq')
	 find_nans_in_table(self.model.layers[i].debug_module_list.construct_denominator_seq.gradInput)
	 --print(self.model.construct_denominator_seq.output:unfold(1,10,10))
	 io.read()


	 --[[
	 print('gradInputs')
	 print(self.model.encoding_feature_extraction_dictionary.gradInput:unfold(1,10,10)) -- all nans
	 print(self.model.encoding_pooling_dictionary.gradInput:unfold(1,10,10)) -- all nans
	 print(self.model.ista_sparsifying_loss_seq.gradInput[1]:unfold(1,10,10)) -- all nans
	 print(self.model.pooling_seq.gradInput[1]:unfold(1,10,10)) -- all nans
	 print('pooling L2 loss input 1')
	 print(self.model.pooling_L2_loss_seq.gradInput[1]:unfold(1,10,10)) -- all nans
	 print('pooling L2 loss input 2')
	 print(self.model.pooling_L2_loss_seq.gradInput[2]:unfold(1,10,10)) -- all nans
	 print('pooling L2 loss input 3')
	 print(self.model.pooling_L2_loss_seq.gradInput[3]:unfold(1,10,10)) -- all nans
	 print(self.model.pooling_sparsifying_loss_seq.gradInput[1]:unfold(1,10,10))
	 print(self.model.pooling_sparsifying_loss_seq.gradInput[2]:unfold(1,10,10))
	 print(self.model.classification_dictionary.gradInput:unfold(1,10,10))
	 io.read()
	 
	 print('weights')
	 print(self.model.encoding_feature_extraction_dictionary.weight[{1,{1,10}}]:unfold(1,10,10))
	 print(self.model.explaining_away.weight[{1,{1,10}}]:unfold(1,10,10))
	 print(self.model.shrink.shrink_val[{{1,10}}]:unfold(1,10,10))
	 print(self.model.encoding_pooling_dictionary.weight[{1,{1,10}}]:unfold(1,10,10))
	 print(self.model.classification_dictionary.weight)
	 io.read()
	 --]]
      end -- loop over model layers
   end -- if found_a_nan
end
	    
