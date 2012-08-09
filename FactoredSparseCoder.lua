local FactoredSparseCoder, parent = torch.class('unsup.FactoredSparseCoder','unsup.UnsupModule')


-- inputSize   : size of input
-- cmul_code_size  : size of each half of the internal combined_code
-- L1_code_size : size of the code underlying the L1 half of combined_code
-- L2_code_size : size of the code underlying the L2 half of combined_code; currently disabled and assumed equal to cmul_code_size
-- lambda      : sparsity coefficient
-- params      : optim.FistaLS parameters

-- it's probably easiest to fully split the code after factorization, and then just copy into a concatenated tensor for sharing with fista.  Each module stores output and gradInput internally, so these cannot easily be defined as views of a larger shared Storage
function FactoredSparseCoder:__init(input_size, cmul_code_size, L1_code_size, target_size, lambda, params)

   parent.__init(self)
   self.cmul_code_size = cmul_code_size -- while we get away without explicitly referring to these elsewhere in FactoredSparseCoder, they're useful for outside functions like the tester
   self.L1_code_size = L1_code_size
   self.target_size = target_size

   self.use_L1_dictionary = true
   self.use_L1_dictionary_training = true
   self.use_lagrange_multiplier_cmul_mask = true
   self.use_lagrange_multiplier_L1_units = true
   -- L1 norming the L1 dictionary, in conjunction with shrink_cmul_mask, makes the fista dynamics identical to an L1 norm on the units themselves, since the scaling of each unit provided by the dictionary is fixed equal to one.
   self.L1_norm_L1_dictionary = false
   self.bound_L1_dictionary_dimensions = false -- WARNING: this contains unnecessarily INEFFICIENT operations
   self.shrink_L1_dictionary = false -- in some sense, this enforces the desiderata of low-dimensional subspaces, but only indirectly.  Rather than controlling the sparsity of the true cmul mask, which is the dot product of the L1 units with the L1_dictionary, it only addresses the L1 dictionary.  This allows very non-sparse masks to minimize the energy, so the gradient of the reconstruction error with respect to the weights at the local minima of the energy will tend to make the L1_dictionary dense.  Given the strong pressure from the reconstruction error to use dense L1_dictionary, it seems plausible that the strength of a direct L1 sparsity on the L1_dictionary will be difficult to calibrate.  
   self.shrink_cmul_mask = true
   self.shrink_L1_units = true
   self.use_L2_code = true

   self.use_top_level_classifier = true --false
   if self.use_top_level_classifier then
      self.wake_sleep_stage = 'wake'
      self.sleep_stage_use_classifier = true
      self.sleep_stage_learning_rate_scaling_factor = -0.4 -- -0.2 -- scale down the learning rate for all parameters in the sleep stage except the lagrange multipliers, which are only trained in the sleep stage
      -- keep in mind that the maximal values of the digit ID target are likely to be larger than those of the pixel input, potentially leading to larger gradients
      -- note that if the digit ID is always correctly reconstructed in the sleep stage because the top-level classification error gradient is larger than that due to all other errors, the top_level_classifier_dictionary will not be effectively trained, since concat_code will reflect the digit ID rather than the pixel input
      self.top_level_classifier_scaling_factor = 2.5e-2 --1e-1 --5e-2 -- scale the gradient due to the top-level classifier, so it is comparable to that due to the reconstruction error; otherwise, the concat code is driven primarily by the classification, and it's difficult to learn good reconstructions.  Keep in mind that the relatively indirect connection to the reconstruction relative to the classification from concat_code tends to make the reconstruction gradient small
   end


   -- sizes of internal codes
   local L2_code_size = self.cmul_code_size
   if not(self.use_L1_dictionary) then
      self.L1_code_size = self.cmul_code_size
   end

   -- sparsity coefficient
   -- it seems to be best if L1_lambda = L2_lambda; this seems to ensure that the maximal L2 units are about 0.5.  We initialize all L2 units to 0.3, which is halfway between 0 and the maximal value
   local chosen_lambda = 0
   if not(self.use_L2_code) then 
      chosen_lambda = lambda*4
   --elseif self.L1_norm_L1_dictionary then
   --   chosen_lambda = lambda*2/3
   --elseif self.use_L1_dictionary_training then 
   --   chosen_lambda = lambda/10
   else
      chosen_lambda = lambda
   end

   self.L1_lambda = chosen_lambda / 10 -- NOTE that this only changes the initialization if we are using L1 lagrange multipliers on the L1 units
   self.L2_lambda = chosen_lambda
   
   -- This was effectively scaled down by a factor of 100 or 200 when we removed kex.nnhacks, which scaled up the learning rate of nn.Linear units by their in-degree (nnhacks scaled down, so removing it scaled up), necessitating a complementary 1/200 scaling of the general learning rate.  However, keep in mind that the newly scaled down learning rate only affects the training of the L1_dictionary weights, but not the sparsification of the L1 units during normal network dynamics.  It is thus probably not safe to scale L1_lambda_cmul up by a factor of 200 to fully counteract the effective change in learning rate
   self.L1_lambda_cmul = chosen_lambda / 30 -- /10
   --self.L1_L2_lambda = chosen_lambda
   self.L1_dictionary_lambda = 1e-6 --5e-8 --1e-8 --1e-10 --1e-7
   
   if self.use_lagrange_multiplier_L1_units or self.use_lagrange_multiplier_cmul_mask then 
      self.L1_lagrange_decay = 0.985 --0.98 -- 0.99
      -- in principle, the L1_dictionary_learning_rate_scaling should be either 0 or 1; otherwise, the dynamics and the training minimize different energy functions, so the partial derivative of the training-energy with respect to the unit activities is not zero, and the total derivative of the unit activities with respect to the parameters contributes to the total derivative of the training-energy with respect to the parameters
      self.L1_dictionary_learning_rate_scaling = 1 --2e-1 --1e-1 --0.01 --0.1 
   else
      -- scale down the speed at which the L1 dictionary learns 
      -- with 0.1, they change *VERY* slowly
      -- with 1, they become unstable, changing more quickly than the L1 lagrange multipliers, so a small number of L1 units become highly active and the rest are ignored
      self.L1_dictionary_learning_rate_scaling = 1e-1 --2e-1 --1e-1 --0.01 --0.1 
   end


   -- this doesn't seem to work well after kex.nnhacks() was removed, since the effect of the cmul regularizer on training is substantially increased relative to its effect on the dynamics.  The amount of regularization the network can sustain without activity collapsing substantially increases as the cmul_dictionary is trained.  The level of cmul regularization that the network can sustain with a random cmul_dictionary is insufficient to sustain the sparsity of the L1_dictionary when it is made variable.
   if self.use_lagrange_multiplier_L1_units and not(self.use_lagrange_multiplier_cmul_mask) then 
      self.lagrange_target_value_cmul_mask = 0
      self.lagrange_target_value_L1_units = 1.0e-2/(1 - self.L1_lagrange_decay) --scale the actual target by 1/(1 - L1_lagrange_decay) 
      self.lagrange_multiplier_L1_units_learning_rate_scaling = 200*1e-6-- was 1e-6 before removal of kex.nnhacks --4e-7 seems stable --5e-7 --2e-7 --1e-6 --1e-7 
      self.lagrange_multiplier_cmul_mask_learning_rate_scaling = 0

      self.L1_lambda = 0 --chosen_lambda / 30
      self.L1_lambda_cmul = (200/60)*chosen_lambda --/ 5 --/ 60 --75 --60
      --self.L1_lambda_cmul = chosen_lambda / 60 --75 --60
      -- an initial period in which the L1_dictionary is untrained seems to be *critical* to good performance.  Make sure that classification performance plateaus before L1_dictionary training is enabled
      -- the lagrange_multipliers should change slowly, so the L1 constraint is enforced on the average activity over time, rather than on short runs of activity.  However, if it changes too slowly, instabilities can arise in the L1 units, with some units considerably more active than others
      -- if lagrange_multiplier_learning_rate_scaling is too small, individual L1_dictionary elements will be quickly trained to account for large portions of the dataset when the L1 regularizer for the unit is too small, and then be ignored as it is set larger, and a better candidate is found.  Ideally, the L1_dictionary elements should be trained evenly, reflecting the fact that they are all active an equal percentage of the time
      -- the lagrange_multiplier_*_learning_rate_scaling should not affect the energy function minimized in theory, since their effect is only guaranteed at fixed points of training (gradient equal to zero)
   elseif self.use_lagrange_multiplier_cmul_mask and not(self.use_lagrange_multiplier_L1_units) then
      -- target may need to be smaller to avoid oscillations with use_lagrange_multiplier_cmul_mask
      -- it is *ESSENTIAL* that the lagrange multipliers, and thus the quantity they control, are stable.  I've turned up the lagrange_multiplier_learning_rate_scaling when the cmul mask is controlled and the L1_dictionary is trained, since otherwise the lagrange multipliers execute large, sweeping trajectories, suggesting that they are not enforcing the desired constraints on cmul mask activity.  When the L1_dictionary is fixed, these constraints seem easier to enforce, so the lagrange_multiplier_learning_rate_scaling can be turned back down.
      
      -- presumably, L1 dictionary elements become non-sparse when trained with cmul mask controlled by lagrange multipliers because many of the lagrange multipliers are set equal to 0
      -- the difficulty of controlling the lagrange multipliers with cmul mask targets seems connected to the fact that the lagrange multipliers try to go negative
      
      -- I think part of the problem with using lagrange multipliers on the cmul mask is that it's very easy to reduce the L1 norm of the cmul mask by setting entries of the L1_dictionary to zero.  As a result, the L1_dictionary is quickly driven to be very sparse when the system needs to reduce the L1 norm of a cmul mask entry to meet a lagrange-enforced target.  
      --self.L1_lagrange_decay = 0.98 -- 0.99
      self.lagrange_target_value_cmul_mask = 1.0e-2/(1 - self.L1_lagrange_decay) --0.5e-2/(1 - self.L1_lagrange_decay)
      self.lagrange_target_value_L1_units = 0
      self.lagrange_multiplier_cmul_mask_learning_rate_scaling = 200*1e-6-- was 1e-6 before removal of kex.nnhacks --4e-7 seems stable --5e-7 --2e-7 --1e-6 --1e-7 
      self.lagrange_multiplier_L1_units_learning_rate_scaling = 0

      self.L1_lambda_cmul = chosen_lambda / 300
      self.L1_lambda = chosen_lambda / 10
      --self.L1_lambda = 0 --chosen_lambda / 50
      --self.L1_lambda_cmul = chosen_lambda / 5
   elseif self.use_lagrange_multiplier_cmul_mask and self.use_lagrange_multiplier_L1_units then
      -- the L1 unit lagrange multipliers evolve too quickly when both lagrange multipliers update at the same rate
      -- if targets are too low, lagrange multipliers go to zero and are uncontrolled.  There's an odd dependence between the two targets; making one target more stringent reduces the pressure on the other set of lagrange multipliers.  Fine-tuning the balance between the two targets seems difficult.  Can these two constraints be integrated?
      -- NOTE that if a cmul mask lagrange multiplier goes to zero, L1_dictionary weights projecting to it are *NOT* sparsified!!!

      -- increasing the sparsity seems to have dramatically reduced performance.  Keep in mind that the full digits that tend to appear in cmul_dictionary rather than strokes are likely indicative of too much sparsity
      --self.lagrange_target_value_cmul_mask = 0.75e-2/(1 - self.L1_lagrange_decay) --0.9e-2/(1 - self.L1_lagrange_decay) --0.5e-2/(1 - self.L1_lagrange_decay)
      self.lagrange_target_value_cmul_mask = 0.9e-2/(1 - self.L1_lagrange_decay) --0.5e-2/(1 - self.L1_lagrange_decay)
      --self.lagrange_target_value_L1_units = 0.6e-2/(1 - self.L1_lagrange_decay) --0.75e-2/(1 - self.L1_lagrange_decay) --1.0e-2/(1 - self.L1_lagrange_decay)
      self.lagrange_target_value_L1_units = 0.75e-2/(1 - self.L1_lagrange_decay) --1.0e-2/(1 - self.L1_lagrange_decay)
      self.lagrange_multiplier_L1_units_learning_rate_scaling = 200*1e-6-- was 1e-6 before removal of kex.nnhacks --4e-7 seems stable --5e-7 --2e-7 --1e-6 --1e-7 
      self.lagrange_multiplier_cmul_mask_learning_rate_scaling = 200*1e-6-- was 1e-6 before removal of kex.nnhacks --4e-7 seems stable --5e-7 --2e-7 --1e-6 --1e-7 


      self.L1_lambda_cmul = chosen_lambda / 300
      self.L1_lambda = chosen_lambda / 10
   end


   print('L2_code_size = ' .. L2_code_size .. ', L1_code_size = ' .. self.L1_code_size .. ', cmul_code_size = ' .. self.cmul_code_size .. ', L1_lambda = ' .. self.L1_lambda .. ', L1_lambda_cmul = ' .. self.L1_lambda_cmul .. ', L2_lambda = ' .. self.L2_lambda)

   -- dictionaries are trainable linear layers; cmul combines factored representations
   if self.use_L1_dictionary then
      self.L1_dictionary = nn.ScaledGradLinear(self.L1_code_size, self.cmul_code_size)
   else
      if self.L1_code_size ~= self.cmul_code_size then
	 error('L1_code_size must match cmul_code_size when not using an L1_dictionary')
      end
      self.L1_dictionary_identity_matrix = torch.eye(self.cmul_code_size)
   end
   self.cmul = nn.InputCMul(self.cmul_code_size)
   self.cmul_dictionary = nn.ScaledGradLinear(self.cmul_code_size, input_size)
   -- L2 reconstruction cost
   self.input_reconstruction_cost = nn.MSECriterion()
   self.input_reconstruction_cost.sizeAverage = false

   self.concat_code_L2_cost = nn.MSECriterion()
   self.concat_code_L2_cost.sizeAverage = false
   self.concat_constant_zeros_L2 = torch.zeros(L2_code_size)
   --self.concat_code_L2_side_L1_supplement = nn.L1Cost()

   if self.L1_L2_lambda and (self.L1_L2_lambda ~= 0) then 
      print('constructing L1_L2 norm: ' .. self.L1_L2_lambda)
      self.concat_code_L1_L2_cost = nn.MSECriterion()
      self.concat_code_L1_L2_cost.sizeAverage = false
      self.concat_constant_zeros_L1_L2 = torch.zeros(self.L1_code_size)
   end
   -- L1 sparsity cost
   if self.shrink_cmul_mask then 
      self.factored_code_L1_cost = nn.L1Cost()
   end
   if self.shrink_L1_units then 
      self.concat_code_L1_cost = nn.L1Cost()
   end
   


   -- top-level classification for wake/sleep training
   if self.use_top_level_classifier then 
      --self.top_level_classification_cost = nn.MSECriterion()
      self.top_level_classification_cost = nn.ClassNLLCriterion()
      self.top_level_classification_cost.sizeAverage = false
   end


   -- To generate two parallel chains that process different inputs, first Replicate the input, feed it into a Parallel container, and Narrow within each module of the Parallel container to select the desired inputs.  Neither Replicate nor Narrow actually copy memory - they just change the tensor view of the underlying storage - so both are safe.
   self.processing_chain = nn.Sequential()
   self.processing_chain:add(nn.Replicate(2))
   self.L2_processing_chain = nn.Narrow(1,1,L2_code_size)
   self.L1_processing_chain = nn.Sequential()
   self.L1_pc_narrow = nn.Narrow(1,L2_code_size+1,self.L1_code_size)
   self.L1_processing_chain:add(self.L1_pc_narrow)
   if self.use_L1_dictionary then
      self.L1_processing_chain:add(self.L1_dictionary)
   end
   self.factored_processor = nn.Parallel(1,1)
   self.factored_processor:add(self.L2_processing_chain)
   self.factored_processor:add(self.L1_processing_chain)
   self.processing_chain:add(self.factored_processor)
   self.processing_chain:add(self.cmul)
   self.processing_chain:add(self.cmul_dictionary)

   if self.use_top_level_classifier then
      self.top_level_classifier = nn.Sequential()
      self.L1_tlc_narrow = nn.Narrow(1,L2_code_size+1,self.L1_code_size)
      self.top_level_classifier_dictionary = nn.ScaledGradLinear(self.L1_code_size, self.target_size)
      self.top_level_classifier_log_softmax = nn.LogSoftMax()
      self.top_level_classifier:add(self.L1_tlc_narrow)
      self.top_level_classifier:add(self.top_level_classifier_dictionary)
      self.top_level_classifier:add(self.top_level_classifier_log_softmax)
   end

   -- this is going to be set at each forward call.
   self.input = nil
   self.target = nil

   --self.factored_code = torch.Tensor(2*self.cmul_code_size):fill(0)
   self.concat_code = torch.Tensor(self.L1_code_size + L2_code_size):fill(0)
   self.code = self.concat_code -- we create this variable solely because unsup.PSD expects it
   
   -- this is going to be passed to unsup.FistaLS
   --self.grad_concat_code_smooth = torch.Tensor(self.L1_code_size + L2_code_size):fill(0) -- REMOVE THIS - SHOULD BE UNNECESSARY
   --self.grad_concat_code_nonsmooth = torch.Tensor(self.L1_code_size + L2_code_size):fill(0) -- REMOVE THIS - SHOULD BE UNNECESSARY

   self.extract_L2_from_concat = function(this_concat_code) return this_concat_code:narrow(1,1,L2_code_size) end
   self.extract_L1_from_concat = function(this_concat_code) return this_concat_code:narrow(1,L2_code_size+1,self.L1_code_size) end

   self.extract_L2_from_factored_code = function(this_factored_code) return this_factored_code:narrow(1,1,self.cmul_code_size) end
   self.extract_L1_from_factored_code = function(this_factored_code) return this_factored_code:narrow(1,self.cmul_code_size+1,self.cmul_code_size) end

   local zeros_storage = torch.Tensor(1):fill(0) 
   if self.use_lagrange_multiplier_L1_units then 
      local lagrange_size_L1_units = self.extract_L1_from_concat(self.concat_code):size()
      self.lagrange_multiplier_L1_units = torch.Tensor(lagrange_size_L1_units):fill(3*self.L1_lambda) -- L1_lambda seems to be too small
      self.lagrange_history_L1_units = torch.Tensor(lagrange_size_L1_units):fill(self.lagrange_target_value_L1_units) -- keeps a running average of the L1 activity for comparison with the target
      self.lagrange_grad_L1_units = torch.Tensor(lagrange_size_L1_units):zero()
      self.abs_calc_L1_units = torch.Tensor(lagrange_size_L1_units)
      self.lagrange_multiplier_L1_units_zeros = torch.Tensor(zeros_storage:storage(), zeros_storage:storageOffset(), lagrange_size_L1_units, torch.LongStorage{0}) -- a tensor of all zeros, equal in size to the L1_dictionary, but with only one actual element
   end
   
   if self.use_lagrange_multiplier_cmul_mask then
      local lagrange_size_cmul_mask = torch.LongStorage{self.cmul_code_size}
      self.lagrange_multiplier_cmul_mask = torch.Tensor(lagrange_size_cmul_mask):fill(3*self.L1_lambda_cmul) 
      self.lagrange_history_cmul_mask = torch.Tensor(lagrange_size_cmul_mask):fill(self.lagrange_target_value_cmul_mask) -- keeps a running average of the L1 activity for comparison with the target
      self.lagrange_grad_cmul_mask = torch.Tensor(lagrange_size_cmul_mask):zero()
      self.abs_calc_cmul_mask = torch.Tensor(lagrange_size_cmul_mask)
      self.lagrange_multiplier_cmul_mask_zeros = torch.Tensor(zeros_storage:storage(), zeros_storage:storageOffset(), lagrange_size_cmul_mask, torch.LongStorage{0}) -- a tensor of all zeros, equal in size to the L1_dictionary, but with only one actual element
      
   end


   self.cost_output_counter = 0
   
   init_fsc_cost_functions(self)
   
   -- shrinkage of the L1 units due to the L1 regularizer both directly on the L1 units and on the cmul mask
   if (self.shrink_L1_units and self.use_lagrange_multiplier_L1_units) or self.shrink_cmul_mask then 
      self.L1_shrinkage = self.shrinkage_factory() -- DO enforce nonnegative values when shrinking the L1 units
   end

   -- shrinkage of the L1_dictionary due to the L1 regularizer on the cmul mask
   if self.shrink_cmul_mask then
      --self.L1_dictionary_shrinkage = self.shrinkage_factory(false) -- don't enforce nonnegative weights when shrinking the L1 dictionary weights
      self.L1_dictionary_shrinkage = self.shrinkage_factory() -- DO enforce nonnegative weights when shrinking the L1 dictionary
      self.current_L1_dictionary_shrink_val = torch.Tensor()
      self.L1_dictionary_shrink_val_L1_code_part = torch.Tensor()
   end

   self.minimize_nonsmooth = self.minimizer_factory()



   -- this is for keeping parameters related to fista algorithm
   self.params = params or {}
   -- related to FISTA
   self.params.L = self.params.L or 0.1
   self.params.Lstep = self.params.Lstep or 1.5
   self.params.maxiter = self.params.maxiter or 50
   self.params.maxline = self.params.maxline or 20
   self.params.errthres = self.params.errthres or 1e-4
   self.params.doFistaUpdate = true

   self.wake_L = self.params.L
   self.sleep_L = self.params.L
   self.test_L = self.params.L

   self.gradInput = nil
   --self:reset() -- A reset is performed by LinearPSD:__init() via a call to PSD:__init() well after FactoredSparseCoder:__init() finishes
   --self:init_L1_dict()
   --self:normalize()
end

function FactoredSparseCoder:reset(stdv)
   self.cmul_dictionary:reset(stdv)
   --self.cmul_dictionary:reset('nonnegative')
   self.cmul_dictionary.bias:fill(0)

   if not(self.use_L1_dictionary) and self.L1_dictionary then -- this is probably not critical, since the L1_dictionary is not loaded into the processing chain
      error('L1_dictionary is not used but is defined')
      self.L1_dictionary:reset("identity")
   elseif self.use_L1_dictionary then
      --self.L1_dictionary:reset(stdv) -- IS THIS THE RIGHT SIZE?!?
      self:init_L1_dict() 
      self.L1_dictionary.bias:fill(0)
   end

   if self.use_top_level_classifier then
      self.top_level_classifier_dictionary:reset(stdv)
      self.top_level_classifier_dictionary.weight:fill(1/math.sqrt(self.top_level_classifier_dictionary.weight:size(1))) -- normalize the initial columns to have L2 magnitude 1
      self.top_level_classifier_dictionary.bias:fill(0)
   end

   self:normalize('reset')
   --print(self.L1_dictionary.weight:select(2,self.L1_dictionary.weight:size(2) - 2):unfold(1,8,8))
   --print(self.L1_dictionary.weight:select(2,self.L1_dictionary.weight:size(2) - 1):unfold(1,8,8))
   --print(self.L1_dictionary.weight:select(2,self.L1_dictionary.weight:size(2)):unfold(1,8,8))
end

-- Minimize the energy with respect to the code.  Input is actually the desired output of the main coder, target is the desired output of the top-level classifier, and icode is the initial value of hte internal code
function FactoredSparseCoder:updateOutput(input, icode, target)
   self.input = input
   --self.target = target -- 1-of-n code for MSECriterion
   if self.use_top_level_classifier then 
      local max_val, max_index = torch.max(target,1) -- index code for ClassNLLCriterion
      self.target = max_index[1]
   end

   -- init code to all zeros
   --self.concat_code:fill(0)

   -- if this is a wake stage, just use the last value of the concat_code as the seed
   if (not(self.use_top_level_classifier) and (self.wake_sleep_stage == 'wake')) or (self.wake_sleep_stage == 'sleep') or (self.wake_sleep_stage == 'test') then
      --print('initializing concat_code')
      self.concat_code:copy(icode) 
      -- IT IS ***CRITICAL*** THAT THE INITAL VALUE OF THE L2 UNITS NOT BE SET TOO LARGE, or all L2 units will remain large and basically equal
      -- However, if the L2 units are initialized too small, all units collapse to zero
      self.extract_L2_from_concat(self.concat_code):fill(0.1) -- 0.3
   end

   --print('Initial code')
   --print(self.factored_code:unfold(1,8,8))
   --io.read()

   -- do fista solution
   if self.wake_sleep_stage == 'wake' then
      self.params.L = self.wake_L
   elseif self.wake_sleep_stage == 'sleep' then
      self.params.L = self.sleep_L
   elseif self.wake_sleep_stage == 'test' then
      self.params.L = self.test_L
   end
   
   local oldL = self.params.L
   local concat_code, h = optim.FistaLS(self.smooth_cost, self.nonsmooth_cost, self.minimize_nonsmooth, self.concat_code, self.params)
   local smooth_cost = h[#h].F
   self.output = self.processing_chain.output

   --print('fista ran for ' .. #h .. ' updates')
   
   if not(self.use_top_level_classifier) or (self.wake_sleep_stage == 'wake') then
      self.smooth_cost(concat_code, 'verbose')
      self.nonsmooth_cost(concat_code, 'verbose')
   end
   

   --local error_hist = {};
   --local i
   --for i = 1,#h do
   --   error_hist[i] = h[i].F
   --end
   --print(error_hist)
   --print('FactoredSparseCoder: ' .. fval)

   -- let's just halve the params.L (eq. to double learning rate)
   if oldL == self.params.L then
      self.params.L = self.params.L / 2 
   end

   if self.wake_sleep_stage == 'wake' then
      self.wake_L = self.params.L
   elseif self.wake_sleep_stage == 'sleep' then
      self.sleep_L = self.params.L
   elseif self.wake_sleep_stage == 'test' then
      self.test_L = self.params.L
   end


   --print('Current value of L is: ' .. self.params.L) 

   return smooth_cost, h
end

-- no grad output, because we are unsup
-- d(||Ax-b||+lam||x||_1)/dx
function FactoredSparseCoder:updateGradInput(input, target)
   -- calculate grad wrt to (x) which is code.
   if self.gradInput then
      -- this should never run
   end
   return self.gradInput
end

function FactoredSparseCoder:zeroGradParameters()
   self.cmul_dictionary:zeroGradParameters()
   if self.use_L1_dictionary then 
      self.L1_dictionary:zeroGradParameters()
   end
   
   if self.use_lagrange_multiplier_L1_units then
      self.lagrange_grad_L1_units:zero()
   end
   if self.use_lagrange_multiplier_cmul_mask then
      self.lagrange_grad_cmul_mask:zero()
   end

   if self.use_top_level_classifier then
      self.top_level_classifier_dictionary:zeroGradParameters()
   end
end

-- no gradOutput is required or produced; unsupervised learning depends upon minimizing an energy function, rather than propagating gradients
-- d(||Ax-b||+lam||x||_1)/dA
function FactoredSparseCoder:accGradParameters() -- traditionally, accGradParameters takes in input, target, but the input gradients from the criteria are already calculated in the process of minimizing the energy, so they aren't necessary here
   if self.wake_sleep_stage == 'test' then
      print('ERROR!!!  accGradParameters was called in test mode!!!')
   end

   -- since at the minimum of the energy with respect to the units, the total derivative of the energy with respect to the parameters is equal to the partial derivative of the energy with respect to the parameters, and since updateGradInput was already run throughout the network in the process of finding the minimum of the energy with respect to the units, we can just accumulate the gradients of the parameters for each of the component dictionaries
   --self.input_reconstruction_cost:updateGradInput(self.cmul_dictionary.output,input) -- this should be unnecessary, since it is done by FISTA
   self.cmul_dictionary:accGradParameters(self.cmul.output, self.input_reconstruction_cost.gradInput)
   self.cmul_dictionary.gradBias:fill(0)
   local L1_code_from_concat = self.extract_L1_from_concat(self.concat_code)

   if self.use_L1_dictionary and self.use_L1_dictionary_training then
      self.L1_dictionary:accGradParameters(L1_code_from_concat, self.extract_L1_from_factored_code(self.cmul.gradInput))
      self.L1_dictionary.gradBias:fill(0)
   end

   -- only update the lagrange multipliers during the sleep stage, if we're using the top-level classifier only in the wake stage;
   -- only update the lagrange multipliers during the wake stage if we're using the top-level classifier in both the wake and sleep stages
   if not(self.use_top_level_classifier) or ((self.wake_sleep_stage == 'sleep') and not(self.sleep_stage_use_classifier)) or ((self.wake_sleep_stage == 'wake') and self.sleep_stage_use_classifier) then
      if self.use_lagrange_multiplier_L1_units then
	 self.lagrange_history_L1_units:mul(self.L1_lagrange_decay)
	 
	 self.abs_calc_L1_units:resizeAs(L1_code_from_concat)
	 self.abs_calc_L1_units:abs(L1_code_from_concat)
	 
	 self.lagrange_history_L1_units:add(self.abs_calc_L1_units) 
	 self.lagrange_grad_L1_units:add(self.lagrange_history_L1_units):add(-1 * self.lagrange_target_value_L1_units)
      end

      if self.use_lagrange_multiplier_cmul_mask then
	 self.lagrange_history_cmul_mask:mul(self.L1_lagrange_decay)
	 
	 self.abs_calc_cmul_mask:resizeAs(self.L1_processing_chain.output)
	 self.abs_calc_cmul_mask:abs(self.L1_processing_chain.output) 

	 self.lagrange_history_cmul_mask:add(self.abs_calc_cmul_mask) 
	 self.lagrange_grad_cmul_mask:add(self.lagrange_history_cmul_mask):add(-1 * self.lagrange_target_value_cmul_mask)
      end
   end

   if self.use_top_level_classifier and ((self.wake_sleep_stage == 'wake') or self.sleep_stage_use_classifier) then 
      --self.top_level_classifier_dictionary:accGradParameters(L1_code_from_concat, self.top_level_classification_cost.gradInput)
      self.top_level_classifier_dictionary:accGradParameters(L1_code_from_concat, self.top_level_classifier_log_softmax.gradInput)
      self.top_level_classifier_dictionary.gradBias:fill(0)
   end
   
end

function FactoredSparseCoder:updateParameters(learning_rate)
   local wake_only_learning_rate = learning_rate
   local default_learning_rate = ((self.use_top_level_classifier and (self.wake_sleep_stage == 'sleep') and self.sleep_stage_learning_rate_scaling_factor) or 1) * learning_rate
   if (self.wake_sleep_stage == 'sleep') and not(self.use_top_level_classifier) then print('WARNING!  Learning rate is not negative in sleep stage!') end -- sleep is only used in conjunction with a top-level classifier; the maximum likelihood configurations with the inputs only constrained in their total L2 magnitude, and in the absence of a top-level classifier, are not very interesting.  

   if self.wake_sleep_stage == 'test' then error('updateParameters was called in test mode!!!') end

   self.cmul_dictionary:updateParameters(default_learning_rate)
   self.cmul_dictionary.bias:fill(0)

   if self.use_L1_dictionary and self.use_L1_dictionary_training then
      self.L1_dictionary:updateParameters(default_learning_rate * self.L1_dictionary_learning_rate_scaling)
      self.L1_dictionary.bias:fill(0)

      -- apply an L1 regularizer to the L1 dictionary, to encourage sparse connections
      if self.shrink_L1_dictionary and (not(self.use_top_level_classifier) or (self.wake_sleep_stage == 'wake')) then -- only do L1 shrinkage once per wake/sleep alternation
	 self.L1_dictionary.weight:shrinkage(wake_only_learning_rate * self.L1_dictionary_learning_rate_scaling * (1 + self.sleep_stage_learning_rate_scaling_factor) * self.L1_dictionary_lambda)
      end
   end
   
   -- run this regardless of whether the current stage is wake or sleep, but gradParameters are only accumulated during the sleep stage
   -- lagrange multipliers follow the gradient, rather than the negative gradient; make sure that the gradient is followed in the same direction regardless of manipulations of the sign of learning_rate based upon wake/sleep alternations
   if self.use_lagrange_multiplier_L1_units then
      self.lagrange_multiplier_L1_units:add(self.lagrange_multiplier_L1_units_learning_rate_scaling * wake_only_learning_rate * (1 + ((self.use_top_level_classifier and self.sleep_stage_learning_rate_scaling_factor) or 0)), self.lagrange_grad_L1_units) 
      
      self.lagrange_multiplier_L1_units[torch.lt(self.lagrange_multiplier_L1_units, self.lagrange_multiplier_L1_units_zeros)] = 0 -- bound the lagrange multipliers below by zero - WARNING: this is unnecessarily INEFFICIENT, since memory is allocated on each call
   end
   
   if self.use_lagrange_multiplier_cmul_mask then
      self.lagrange_multiplier_cmul_mask:add(self.lagrange_multiplier_cmul_mask_learning_rate_scaling * wake_only_learning_rate * (1 + ((self.use_top_level_classifier and self.sleep_stage_learning_rate_scaling_factor) or 0)), self.lagrange_grad_cmul_mask) 
      
      self.lagrange_multiplier_cmul_mask[torch.lt(self.lagrange_multiplier_cmul_mask, self.lagrange_multiplier_cmul_mask_zeros)] = 0 -- bound the lagrange multipliers below by zero - WARNING: this is unnecessarily INEFFICIENT, since memory is allocated on each call
   end
   

   -- shrink the L1_dictionary based upon the L1 regularizer on the cmul mask
   if self.shrink_cmul_mask and (not(self.use_top_level_classifier) or (self.wake_sleep_stage == 'wake')) and self.use_L1_dictionary and self.use_L1_dictionary_training then  -- MAKE SURE THAT SHRINKAGE IS NOT DONE IN REVERSE DURING THE SLEEP STAGE!!!
      local L1_code_from_concat = self.extract_L1_from_concat(self.concat_code)
      self.L1_dictionary_shrink_val_L1_code_part:resizeAs(L1_code_from_concat)
      self.L1_dictionary_shrink_val_L1_code_part:abs(L1_code_from_concat) -- |a*b| = |a|*|b|
      if self.use_lagrange_multiplier_cmul_mask then 
	 -- we only shrink in the wake stage, so scale the shrinkage down by the difference between the wake and sleep stage learning rates
	 self.L1_dictionary_shrink_val_L1_code_part:mul(wake_only_learning_rate * self.L1_dictionary_learning_rate_scaling * (1 + self.sleep_stage_learning_rate_scaling_factor))
	 self.current_L1_dictionary_shrink_val:resizeAs(self.L1_dictionary.weight)
	 self.current_L1_dictionary_shrink_val:ger(self.lagrange_multiplier_cmul_mask, self.L1_dictionary_shrink_val_L1_code_part)
	 --print('shrinking by')
	 --print(self.current_L1_dictionary_shrink_val:select(1,1):unfold(1,10,10))
	 --print('before shrinking')
	 --print(self.L1_dictionary.weight:select(1,1):unfold(1,10,10))
	 self.L1_dictionary_shrinkage(self.L1_dictionary.weight, self.current_L1_dictionary_shrink_val)
	 --print('after shrinking')
	 --print(self.L1_dictionary.weight:select(1,1):unfold(1,10,10))
	 --io.read()
      else
	 -- we only shrink in the wake stage, so scale the shrinkage down by the difference between the wake and sleep stage learning rates
	 self.L1_dictionary_shrink_val_L1_code_part:mul(wake_only_learning_rate * self.L1_dictionary_learning_rate_scaling * self.L1_lambda_cmul * (1 + self.sleep_stage_learning_rate_scaling_factor))  
	 local L1_units_abs_duplicate_rows = torch.Tensor(self.L1_dictionary_shrink_val_L1_code_part:storage(), self.L1_dictionary_shrink_val_L1_code_part:storageOffset(), self.L1_dictionary.weight:size(), torch.LongStorage{0,1}) -- this doesn't allocate new main storage, so it should be relatively efficient, even though a new Tensor view is created on each iteration
	 --print('before shrinkage')
	 --print(self.L1_dictionary.weight:select(1,1):unfold(1,10,10))
	 self.L1_dictionary_shrinkage(self.L1_dictionary.weight, L1_units_abs_duplicate_rows) -- NOTE that self.L1_dictionary_shrink_val_L1_code_part cannot be reused after this, since it is multiplied by -1 when shrinking
	 --print('after shrinkage')
	 --print(self.L1_dictionary.weight:select(1,1):unfold(1,10,10))
	 --io.read()
      end 
   end -- shrink_cmul_mask

   if self.use_top_level_classifier then -- run this regardless of whether the current stage is wake or sleep, but gradParameters are only accumulated during the wake stage
      self.top_level_classifier_dictionary:updateParameters(default_learning_rate) -- * self.L1_dictionary_learning_rate_scaling
      self.top_level_classifier_dictionary.bias:fill(0)
   end
   
   self:normalize() -- added by Jason 6/5/12
end

function FactoredSparseCoder:normalize(mode)
   -- normalize the dictionary
   local function normalize_dictionary(w)
      for i=1,w:size(2) do
	 w:select(2,i):div(w:select(2,i):norm()+1e-12)
      end
   end

   local function L1_normalize_dictionary(w)
      for i=1,w:size(2) do
	 --[[ local abs_sum = 0
	 w:select(2,i):apply(function(x) abs_sum = abs_sum + math.abs(x) end)
	 if abs_sum ~= w:select(2,i):norm(1) then
	    print('abs abs_sum = ' .. abs_sum .. ' but L1 norm = ' .. w:select(2,i):norm(1))
	 end --]]
	 w:select(2,i):div(w:select(2,i):norm(1)+1e-12)
      end
   end

   -- normalize the columns of the reconstruction (cmul) dictionary
   if self.use_L1_dictionary and (mode == 'reset' or true or not(self.use_lagrange_multiplier_cmul_mask)) then 
      --print('normalizing the cmul_dictionary')
      normalize_dictionary(self.cmul_dictionary.weight)
   end

   -- set all but num_dimensions elements to 0
   if self.use_L1_dictionary and self.bound_L1_dictionary_dimensions then 
      local num_dimensions = 10
      local sorted, sort_order = torch.sort(torch.abs(self.L1_dictionary.weight:transpose(1,2))) -- WARNING: this is INEFFICIENT since memory is allocated on each call
      local sort_order_t = sort_order:transpose(1,2)
      for r=1,sort_order_t:size(1) - num_dimensions do
	 for c=1,sort_order_t:size(2) do
	    --print('r ' .. r .. ' c ' .. c .. ' so ')
	    self.L1_dictionary.weight[{sort_order_t[{r,c}], c}] = 0
	 end
      end
   end
   
   -- normalize the columns of the L1 dictionary.  If we use L1 lagrange multipliers, they constrain the balance between scaling the L1 dictionary and scaling the L1 units, so this normalization is not necessary or desirable.  However, if the L1_dictionary is not normalized, it's possible to achieve a desired L1 norm for the L1 units by scaling the corresponding L1_dictionary element, without altering its contribution to the reconstruction.
   if self.use_L1_dictionary and (mode == 'reset' or true or not(self.use_lagrange_multiplier_target_L1_units)) then 
      --print('normalizing the L1_dictionary')
      if self.L1_norm_L1_dictionary then 
	 L1_normalize_dictionary(self.L1_dictionary.weight)
      else 
	 normalize_dictionary(self.L1_dictionary.weight)
      end
   end

   if (mode == 'reset') and self.use_top_level_classifier then
      normalize_dictionary(self.top_level_classifier_dictionary.weight)
   end

end


function FactoredSparseCoder:init_L1_dict()
   --print('initializing L1 dictionary')
   --print(self.L1_dictionary)
   if self.use_L1_dictionary then
      --print('here we go')
      local w = self.L1_dictionary.weight
      local num_rows, num_cols = w:size(1), w:size(2)
      for i=1,num_cols do
	 local j = 0
	 w:select(2, i):apply(function()
				 j = j + 1
				 return (math.abs(math.floor(i * num_rows/num_cols) - j) <= 2 and 1) or 0
			      end)
      end
   end
   --local dc = image.toDisplayTensor{input=self.L1_dictionary.weight:transpose(1,2):unfold(2,20,20),padding=1,nrow=10,symmetric=true}
   --image.savePNG('inital_L1_dictionary.png',dc)
end



function FactoredSparseCoder:parameters()
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   local w = {}
   local gw = {}

   local module_array = {self.cmul_dictionary}
   if self.use_L1_dictionary then
      table.insert(module_array, self.L1_dictionary)
   end
   if self.use_top_level_classifier then
      table.insert(module_array, self.top_level_classifier_dictionary)
   end
   
   for i=1,#module_array do
      --print('extracting parameters from module', self.modules[i])
      local mw,mgw = module_array[i]:parameters() -- this ends up being called recursively, and expects the eventual return of an array of parameters and an array of parameter gradients
      if mw then
         tinsert(w,mw)
         tinsert(gw,mgw)
      end
   end
   
   if self.use_lagrange_multiplier_L1_units then -- enable this to save lagrange multipliers
      tinsert(w, self.lagrange_multiplier_L1_units)
      tinsert(gw, torch.Tensor())
   end

   if self.use_lagrange_multiplier_cmul_mask then -- enable this to save lagrange multipliers
      tinsert(w, self.lagrange_multiplier_cmul_mask)
      tinsert(gw, torch.Tensor())
   end


   return w,gw   
end


--[[ OLD VERSION!!!
function FactoredSparseCoder:parameters()
   local seq = nn.Sequential()
   seq:add(self.cmul_dictionary)
   if self.use_L1_dictionary then
      seq:add(self.L1_dictionary)
   end
   if self.use_top_level_classifier then
      seq:add(self.top_level_classifier_dictionary)
   end
   return seq:parameters()
end
--]]
   