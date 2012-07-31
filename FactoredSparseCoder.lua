local FactoredSparseCoder, parent = torch.class('unsup.FactoredSparseCoder','unsup.UnsupModule')


-- Consider using some sort of weight regularization to enforce the desiderata that each L1 unit corresponds to a low-dimensional subspace, and thus its dictionary must be sparse
-- L1 normalization with the same lambda causes activity to crash even when the L1 dictionary is not trainable, since it makes the dictionary elements too small; we must reduce the L1 penalty.  However, consider the possibility that an L1 subtractive penalty would be better, since it will actually sparsify.
-- Learning the L1 dictionary with the number of non-zero elements capped seems to be stable, but once a given set of non-zero elements is chosen, it is very difficult to change, since one of these elements would need to go almost to zero before another could be activated.  Moreover, the dictionary elements that devleop the most non-zero elements are the most strongly activated.  Bounding the number of non-zero elements is like a sharp regularizer based on the L0 norm.  
-- Low-dimensional subspaces should generally not be activated, but when they are, they can be very active.  The problem with the trained L1 dictionaries is that, as the L1 dictionary elements become non-sparse, the same units tend to be very active in response to all inputs.  In the limit of a uniform dictionary element, the corresponding unit could represent any input, and would generally be extremely active.  The L1 norm captures sparsity between units, which such a configurations satisfies, but not within a single unit over time.
-- Use lagrange multipliers to control L1 norm of each unit over time (separately, with past times low-pass filtered; i.e. weighted using exponential decay).  Since the desired L1 norm is fixed, this sets the scaling factor for each unit, so the magnitude of the L1 dictionary can be left unconstrained.  This ensures that each unit is equally utilized, where utilization is defined by the L1 norm and thus promotes sparsity.  Previous researchers have constrained the target variance, which is like the single-unit L2 norm over time rather than the single-unit L2 norm over time.  The lagrange multipliers act like an additional set of weights, and are fixed within a fista run; otherwise, the L1 constraint will be exactly satisfied in each fista run, completely determining the activity of the L1 units
-- Is there a problem balancing the L1 norm with the L2 norm if we use a lagrange multiplier to control the mangitude of the L1-regularized units?  Once again, this may be resolved by allowing the magnitude of the L1 dictionary elements to vary.
-- Even with L1 lagrange multipliers ensuring that the L1 norm of each unit over time is constant, what will keep each unit from learning an approximately uniform projection over the L2 units, and then arbitrarily choosing a small set of L1 units to activate all L2 units, allowing perfect reconstruction?  Such an arrangement would not minimize the L1 norms of the L1 units.  Even if the L1 dictionary matrix was scaled so that the lagrange constraints were satisfied, the loss function could be reduced by focusing the L1 dictionary elements, since not all dictionary elements need to be significantly used to reconstruct any given input.  However, the L1 regularizer only weakly constrains the magnitude of the activated units.  To ensure that each L1 unit corresponds to a low-dimensional subspace, it makes sense to apply an L1 regularizer to the L1_dictionary elements directly.  Otherwise, we decompose each input into a set of subspaces that need not be low-dimensional.  An L1 regularizer on the L1 units ensures that inputs are built out of a small number of active subspaces; an L1 regularizer on the L1_dictionary elements ensures that those subspaces are themselves low-dimensional.  

-- When using L1 lagrange multipliers, the sparsity is primarily set by self.L1_lagrange_target_value
-- Keep in mind that the encoder learns the L1 and L2 concat units, not the the cmul factored units, and so is not directly comparable to the cmul_dictionary.  Rather, the last set of encoder dictionary elements corresponds to the filters_L1_proj_dec figures, which project the L1_dictionary matrix through the cmul_dictionary matrix

-- It seems important to make all units (or at the very least either the L1 or the L2 units) non-negative if we're using a top-level classifier.  Otherwise, negative L1 unit values are symmetric to positive L1 unit values with regards to the reconstruction, since the sign of the L2 units can be flipped, but have opposite impact on the classifier output.  Just making the L2 units nonnegative would probably have a similar effect, but would allow low-dimensional subspaces to be of either sign.  Just making the L1 units nonnegative would make the presence and anti-presence of low-dimensional subspaces indistinguishable to the top-level classifier.



-- Yann strongly advocates in favor of pulling up on a particular incorrect answer in the sleep phase, rather than selecting the answer chosen by the system (DO THIS TOMORROW!!!)
-- Yann wants to see an L1 penalty and matrix on the where (L2) side, and a small code without regularization on the what (L1) side.  This constrains information on both sides: either via a small code, or via sparsification.  However, this is similar to the current setup, with the "what" and "where" designations flipped, and the L2 norm on a huge set of units replaced by a small code.  
-- When using an L2 penalty on the where side, Yann is extremely concerned that the pressure to keep the subspaces low-dimensional is too weak.  
-- It might make sense to impose an L1 penalty on the where units.  We want the "what" units to select low-dimensional subspaces, but presently the only pressure on the dimensionality of these subspaces seems to be a direct L1 norm on the L1 dictionary.  By putting an L1 norm on the "where" pathway, we ensure that the network will use a sparse reconstruction, regardless of how promiscuous the L1 dictionary elements are.  However, there is still only a little pressure to make the L1 dictionary elements sparse, aside from a direct regularizer.  CONTINUE HERE!  What keeps each "what" unit confined to a low-dimensional subspace?!?  TEST PERFORMANCE WITH JUST L1 UNITS TOMORROW!!!

-- Consider making part of the energy the magnitude of cmul activation by the L1 units.  Given the choice between a diffuse L1 unit and a focused L1 unit with the same reconstruction error, we should strongly favor the focused unit.  In particular, the partial derivative of the energy with respect to the parameters should *not* cause an L1 unit to connect to all active cmul units; only the active L1 unit(s) with the strongest existing connection should strengthen this connection.  Really, the quantity we want to sparsify is the mask induced by the L1 units.  Any given pixel input should be explained by a low-dimensional subspace of the L2 units, where the active subspace is defined by the projection of the L1 units onto the cmul units.  It should then not be necessary to regulate the L1 units with lagrange multipliers, since there is no pressure to make the L1 dictionary large and the L1 units small; only the product matters.  The dictionary elements should shrink to ensure that the induced mask is optimally efficient.  It is difficult to put an L1 norm on the cmul mask induced by the L1 units, since the regularizer no longer governs each unit separately.  As a simplification, put an L1 norm on the mask projection of each L1 unit separately.  This is equivalent to an L1 norm on the cmul mask if all L1 units and dictionary elements are nonnegative.  To update the L1 units, shrink with magnitude equal to the sum of the absolute values of the corresponding dictionary column; to update the weights, shrink with magnitude equal to the absolute value of the corresponding L1 unit.  
-- This approach does not directly realize the goal of causing only the active L1 unit with the strongest existing connection to strengthen this connection, if the total cmul mask activation is too weak.  However, this the overall dynamics are similar to what happens in standard sparse coding.  If an input pixel is insufficiently explained, all active sparse coding units strengthen their connectivity.  Small discrete dictionary elements nevertheless develop, since extraneous correlations average out, with only the central part is reinforced by continued training, and the L1 regularizer disproportionately reduces weak connections.  Keep in mind that weak, extraneous connections will primarily be eliminated when a unit is strongly activated by an input for which the connection is unnecessary; the L1 regularizer on the weights is weak when the unit itself is only weakly activated.  
-- It might also be possible to put an L1 lagrange multiplier on the cmul units themselves, to ensure that all are used equally.

--Since an L1 regularizer on the cmul mask encourages the sparsest possible mask, early in training when most cmul_dictionary elements are untrained, the L1 units will naturally come to be strongly and sparsely connected to the best-trained cmul units.  Once all mask connections to the untrained units are eliminated, they will never be activated and thus will remain untrained.  Really, we want the activity of each cmul mask element to be equal, according to an L1 norm.  

-- Putting an L1 regularizer on the cmul units directly tends to make the L1 dictionary elements extremely sparse, with only one non-zero component.  This might be due in part to the fact that the sparsity of the L1 code itself is not constrained, so the network is free to use as many L1 units as it likes.  By putting an additional L1 regularizer on the L1 units, in addition to that on the cmul mask, we get a sparse mask that is sparsely generated.  This should be combinable with a lagrange multiplier on the cmul mask activity (described below): the cmul lagrange multipliers will adjust so that cmul mask activity is correct given the constant L1 norm on the L1 units, but the gradient on the weights will only be due to the cmul lagrange multipliers, even though the unit activities are further modified by the L1 unit regurlarizer.  The L1 unit regularizer affects the configuration and thus the weight gradient, even though the L1 norm on the cmul mask is fixed by the lagrange multipliers.
-- How about a lagrange multiplier controlling the overall L1 norm of the cmul mask.  As it is, I'm having difficulty calibrating the sparsity coefficient.  However, unlike putting L1 norms on the L1 units, no single cmul unit can reconstruct every input, and so there's little risk that activity will be sparse in space but not in time (with the same single unit always active).  
-- The lagrange multipliers must be on each cmul mask element separately, or many elements of the cmul mask may be completely turned off at the desired overall level of cmul mask sparsity.



-- Switch to KL-divergence rather than mean-squared error for classification.  Try turning up the scaling on the classification contribution.  Is it safe to turn up the learning rate?  What happens if we increase the size of the L1 layer?  The amount of information stored in this layer is already limited by the direct L1 norm.  Given that we assume that multiple cmul units will be connected to each L1 unit, it's a little odd to have fewer L1 units that cmul units, since the difference between units can already be largely accomodated by disjoint sets of units.  



-- precipitous declines in lagrange multipliers using cmul mask targets seem to be due to the elimination of all connections to the cmul unit through the L1_dictionary.  To ensure that this doesn't happen, enforce more sparsity on the L1 units, and less on the cmul units.  Note that we're already controlling the activity of the cmul units with L1 lagrange multipliers.  There doesn't seem to be any harm in also controlling the activity of the L1 units.  



-- 7/6/12 - either increased top-level-classifier error or increased sparsity seems to be making the cmul dictionary unstable and substantially reducin performance
-- consider the possibility that there just isn't enough representational power with out 200 cmul units and 100 L1 units.  Confirm that when digits are incorrectly classified, it is not because they can be accurately reconstructed using multiple digit IDs, but because they cannot be accurately reconstructed with any digit ID.  Each cmul unit has less representational power because it is not independent.  BUT, remember that just because a cmul mask is on does not imply that the corresponding cmul unit must be active; the L2 modulator is still free.  This would seem to ensure that L1/L2 networks have more representational power than a standard sparse coder.  At the same time, the mask is sparse and structured, whereas a direct, traditional sparsity requirement enforces a similar level of overall sparsity without the corresponding structure of the L1_dictionary.  That is, the L1/L2 network is more restricted than a traditional sparse coder at any given level of sparsity.  



-- with L1_lagrange_target = 1.6/stuff, about five L1 units seem to be significantly active in response to an input.  Surprisingly, their projections onto the input do not consistently correspond to the same digit type.  This suggests that the L1 activations are insufficiently sparse, which allows weakly-matched L1 units to be activated to reconstruct unimportant details of the input.  Keep in mind that very similar L1 units are unlikely to be coactivated, since it is more efficient to activate one of the set strongly.


-- If lambda is too large, only one (or a very small number of) L1 element is strongly activated, and the L1 dictionary learns to be indistinct, since each input must be reconstructed using only a single L1 unit

-- For testing purposes, setting the L2 component to all-ones and refraining from updating turns the network into an unfactored sparse coder; setting the L1 component to all-ones and refraining from updating turns the network into an unfactored PCA

-- inputSize   : size of input
-- cmul_code_size  : size of each half of the internal combined_code
-- L1_code_size : size of the code underlying the L1 half of combined_code
-- L2_code_size : size of the code underlying the L2 half of combined_code; currently disabled and assumed equal to cmul_code_size
-- lambda      : sparsity coefficient
-- params      : optim.FistaLS parameters

-- it's probably easiest to fully split the code after factorization, and then just copy into a concatenated tensor for sharing with fista.  Each module stores output and gradInput internally, so these cannot easily be defined as views of a larger shared Storage
function FactoredSparseCoder:__init(input_size, cmul_code_size, L1_code_size, target_size, lambda, params)

   parent.__init(self)

   self.use_L1_dictionary = true
   self.use_L1_dictionary_training = true
   self.use_lagrange_multiplier_cmul_mask = true
   self.use_lagrange_multiplier_L1_units = true
   -- L1 norming the L1 dictionary, in conjunction with shrink_L1_dictionary_outputs, makes the fista dynamics identical to an L1 norm on the units themselves, since the scaling of each unit provided by the dictionary is fixed equal to one.
   self.L1_norm_L1_dictionary = false
   self.bound_L1_dictionary_dimensions = false -- WARNING: this contains unnecessarily INEFFICIENT operations
   self.shrink_L1_dictionary = false -- in some sense, this enforces the desiderata of low-dimensional subspaces, but only indirectly.  Rather than controlling the sparsity of the true cmul mask, which is the dot product of the L1 units with the L1_dictionary, it only addresses the L1 dictionary.  This allows very non-sparse masks to minimize the energy, so the gradient of the reconstruction error with respect to the weights at the local minima of the energy will tend to make the L1_dictionary dense.  Given the strong pressure from the reconstruction error to use dense L1_dictionary, it seems plausible that the strength of a direct L1 sparsity on the L1_dictionary will be difficult to calibrate.  
   self.shrink_L1_dictionary_outputs = true
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
   local L2_code_size = cmul_code_size
   if not(self.use_L1_dictionary) then
      L1_code_size = cmul_code_size
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
   
   -- This was effectively scaled down by a factor of 100 or 200 when we removed kex.nnhacks, which scaled up the learning rate of nn.Linear units by their in-degree, necessitating a complementary 1/200 scaling of the general learning rate.  However, keep in mind that the newly sclaed down learning rate only affects the training of the L1_dictionary weights, but not the sparsification of the L1 units during normal network dynamics.  It is thus probably not safe to scale L1_lambda_cmul up by a factor of 200 to fully counteract the effective change in learning rate
   self.L1_lambda_cmul = chosen_lambda / 30 -- /10
   --self.L1_L2_lambda = chosen_lambda
   self.L1_dictionary_lambda = 1e-6 --5e-8 --1e-8 --1e-10 --1e-7
   
   if self.use_lagrange_multiplier_L1_units or self.use_lagrange_multiplier_cmul_mask then 
      self.L1_lagrange_decay = 0.985 --0.98 -- 0.99
      -- in principle, the L1_dictionary_learning_rate_scaling should be either 0 or 1; otherwise, the dynamics and the training minimize different energy functions, so the partial derivative of the training-energy with respect to the unit activities is not zero, and the total derivative of the unit activities with respect to the parameters contributes to the total derivative of the training-energy with respect to the parameters
      self.L1_dictionary_learning_rate_scaling = 0 --2e-1 --1e-1 --0.01 --0.1 
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


   print('L2_code_size = ' .. L2_code_size .. ', L1_code_size = ' .. L1_code_size .. ', cmul_code_size = ' .. cmul_code_size .. ', L1_lambda = ' .. self.L1_lambda .. ', L1_lambda_cmul = ' .. self.L1_lambda_cmul .. ', L2_lambda = ' .. self.L2_lambda)

   -- dictionaries are trainable linear layers; cmul combines factored representations
   if self.use_L1_dictionary then
      self.L1_dictionary = nn.Linear(L1_code_size, cmul_code_size)
   end
   self.cmul = nn.InputCMul()
   self.cmul_dictionary = nn.Linear(cmul_code_size, input_size)
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
      self.concat_constant_zeros_L1_L2 = torch.zeros(L1_code_size)
   end
   -- L1 sparsity cost
   if self.shrink_L1_dictionary_outputs then 
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
   self.L1_pc_narrow = nn.Narrow(1,L2_code_size+1,L1_code_size)
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
      self.L1_tlc_narrow = nn.Narrow(1,L2_code_size+1,L1_code_size)
      self.top_level_classifier_dictionary = nn.Linear(L1_code_size, target_size)
      self.top_level_classifier_log_softmax = nn.LogSoftMax()
      self.top_level_classifier:add(self.L1_tlc_narrow)
      self.top_level_classifier:add(self.top_level_classifier_dictionary)
      self.top_level_classifier:add(self.top_level_classifier_log_softmax)
   end

   -- this is going to be set at each forward call.
   self.input = nil
   self.target = nil

   --self.factored_code = torch.Tensor(2*cmul_code_size):fill(0)
   self.concat_code = torch.Tensor(L1_code_size + L2_code_size):fill(0)
   self.code = self.concat_code -- we create this variable solely because unsup.PSD expects it
   
   -- this is going to be passed to unsup.FistaLS
   --self.grad_concat_code_smooth = torch.Tensor(L1_code_size + L2_code_size):fill(0) -- REMOVE THIS - SHOULD BE UNNECESSARY
   --self.grad_concat_code_nonsmooth = torch.Tensor(L1_code_size + L2_code_size):fill(0) -- REMOVE THIS - SHOULD BE UNNECESSARY

   self.extract_L2_from_concat = function(this_concat_code) return this_concat_code:narrow(1,1,L2_code_size) end
   self.extract_L1_from_concat = function(this_concat_code) return this_concat_code:narrow(1,L2_code_size+1,L1_code_size) end

   self.extract_L2_from_factored_code = function(this_factored_code) return this_factored_code:narrow(1,1,cmul_code_size) end
   self.extract_L1_from_factored_code = function(this_factored_code) return this_factored_code:narrow(1,cmul_code_size+1,cmul_code_size) end

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
      local lagrange_size_cmul_mask = torch.LongStorage{cmul_code_size}
      self.lagrange_multiplier_cmul_mask = torch.Tensor(lagrange_size_cmul_mask):fill(3*self.L1_lambda_cmul) 
      self.lagrange_history_cmul_mask = torch.Tensor(lagrange_size_cmul_mask):fill(self.lagrange_target_value_cmul_mask) -- keeps a running average of the L1 activity for comparison with the target
      self.lagrange_grad_cmul_mask = torch.Tensor(lagrange_size_cmul_mask):zero()
      self.abs_calc_cmul_mask = torch.Tensor(lagrange_size_cmul_mask)
      self.lagrange_multiplier_cmul_mask_zeros = torch.Tensor(zeros_storage:storage(), zeros_storage:storageOffset(), lagrange_size_cmul_mask, torch.LongStorage{0}) -- a tensor of all zeros, equal in size to the L1_dictionary, but with only one actual element
      
   end


   self.cost_output_counter = 0
   
   -- Now I need a function to pass along as the smooth cost (f)
   -- input is code, do reconstruction, calculate cost
   -- and possibly derivatives too
   self.smooth_cost = function(concat_code, mode)
      local grad_concat_code_smooth = nil
      local L2_code_from_concat = self.extract_L2_from_concat(concat_code)
      local L1_code_from_concat = self.extract_L1_from_concat(concat_code)

      if not(self.use_L2_code) then
	 L2_code_from_concat:fill(1)
      end

      -- forward function evaluation
      --print('     about to call self.processing_chain:updateOutput(concat_code)')
      local reconstruction = self.processing_chain:updateOutput(concat_code)
      --print('     finished calling self.processing_chain:updateOutput(concat_code)')
      local reconstruction_cost = self.input_reconstruction_cost:updateOutput(reconstruction, self.input) 
      local L2_cost = self.L2_lambda * 0.5 * self.concat_code_L2_cost:updateOutput(L2_code_from_concat, self.concat_constant_zeros_L2)
      if self.L1_L2_lambda and (self.L1_L2_lambda ~= 0) then 
	 cost = cost + self.L1_L2_lambda * 0.5 * self.concat_code_L1_L2_cost:updateOutput(L1_code_from_concat, self.concat_constant_zeros_L1_L2)
      end
      
      local classification = nil
      local classification_cost = 0
      if self.use_top_level_classifier and (not(self.wake_sleep_stage == 'sleep') or self.sleep_stage_use_classifier) then -- WARNING: this is UNNECESSARILY INEFFICIENT!!!  Only need to update classification to check if output is correct
	 classification = self.top_level_classifier:updateOutput(concat_code)
	 -- don't update the classification cost in the test phase, since it's gradient is not used, and the two must be matched for the line search in the fista algorithm
	 if not(self.wake_sleep_stage == 'test') then 
	    -- minimize rather than maximize the likelihood of the correct target during sleep
	    classification_cost = self.top_level_classifier_scaling_factor * self.top_level_classification_cost:updateOutput(classification, self.target)
	    if self.wake_sleep_stage == 'sleep' then classification_cost = -1*classification_cost end
	 end
      end
      local cost = reconstruction_cost + L2_cost + classification_cost
      
      local reconstruction_grad_mag, L2_grad_mag, classification_grad_mag = 0,0,0
      if mode and mode:match('verbose') then
	 self.cost_output_counter = self.cost_output_counter + 1
      end
      	    
      
      -- derivative wrt code
      if mode and (mode:match('dx') or (mode:match('verbose') and self.cost_output_counter >= 250)) then
	 local grad_reconstruction = self.input_reconstruction_cost:updateGradInput(reconstruction, self.input)
	 grad_concat_code_smooth = self.processing_chain:updateGradInput(concat_code, grad_reconstruction)
	 --self.grad_concat_code_smooth:copy(grad_concat_code_smooth)

	 if mode and mode:match('verbose') and self.cost_output_counter >= 250 then
	    reconstruction_grad_mag = grad_concat_code_smooth:norm()
	 end

	 if not(self.use_L2_code) then
	    print('L2 code disabled!')
	    self.extract_L2_from_concat(grad_concat_code_smooth):fill(0)
	 else
	    -- THIS WAS INCORRECTLY SET TO A CONSTANT BEFORE 6/21, DUE TO THE ACCIDENTAL SUBSITUTION OF A COMMA FOR A MULTIPLICATION!!!
	    local L2_grad = self.concat_code_L2_cost:updateGradInput(L2_code_from_concat, self.concat_constant_zeros_L2):mul(self.L2_lambda * 0.5)
	    self.extract_L2_from_concat(grad_concat_code_smooth):add(L2_grad)

	    if mode and mode:match('verbose') and self.cost_output_counter >= 250 then
	       L2_grad_mag = L2_grad:norm()
	    end
	    --self.extract_L2_from_concat(grad_concat_code_smooth):add(torch.mul(self.concat_code_L2_cost:updateGradInput(L2_code_from_concat, self.concat_constant_zeros_L2), self.L2_lambda * 0.5)) -- WARNING: this is INEFFICIENT since torch.mul allocates memory on each iterations
	 end

	 if self.L1_L2_lambda and (self.L1_L2_lambda ~= 0) then 
	    -- NOTE: it might be more parsimonious to do this with a nn.Narrow layer
	    print('Make sure this returns an error, since it multilies a scalar by a tensor')
	    self.extract_L1_from_concat(grad_concat_code_smooth):add(self.L1_L2_lambda * 0.5 * self.concat_code_L1_L2_cost:updateGradInput(L1_code_from_concat, self.concat_constant_zeros_L1_L2))
	    --self.extract_L1_from_concat(grad_concat_code_smooth):add(torch.mul(self.concat_code_L1_L2_cost:updateGradInput(L1_code_from_concat, self.concat_constant_zeros_L1_L2), self.L1_L2_lambda * 0.5))
	 end

	 if self.use_top_level_classifier and ((self.wake_sleep_stage == 'wake') or ((self.wake_sleep_stage == 'sleep') and self.sleep_stage_use_classifier)) then  
	    local grad_classification_cost = self.top_level_classification_cost:updateGradInput(classification, self.target):mul(self.top_level_classifier_scaling_factor)
	    if self.wake_sleep_stage == 'sleep' then grad_classification_cost:mul(-1) end -- minimize rather than maximize the likelihood of the correct target during sleep
	    local classification_grad = self.top_level_classifier:updateGradInput(concat_code, grad_classification_cost)
	    grad_concat_code_smooth:add(classification_grad)
	    if mode and mode:match('verbose') and self.cost_output_counter >= 250 then
	       classification_grad_mag = classification_grad:norm()
	    end
	 end
	 
      end -- dx mode

      if mode and mode:match('verbose') and (self.cost_output_counter >= 250) then
	 print('rec: ' .. reconstruction_cost .. ' ; clas: ' .. classification_cost .. ' ; L2 ' .. L2_cost .. ' ; rec grad: ' .. reconstruction_grad_mag .. ' ; clas grad ' .. classification_grad_mag .. ' ; L2 grad ' .. L2_grad_mag)
      end
      
      return cost, grad_concat_code_smooth
   end
   



   -- the nonsmooth gradient (i.e. the shrink magnitude) is needed in both nonsmooth_cost and minimize_nonsmooth, so just define it once
   -- output is returned in current_shrink_val
   function self.nonsmooth_shrinkage_cmul_factory()
      local L1_dictionary_abs_copy = torch.Tensor()
      
      local function calculate_nonsmooth_shrinkage_cmul(current_shrink_val) 
	 -- Make sure not to alter the gradients in the processing chain, since this will affect the parameter updates; the weight update due to an L1 regularizer needs to be done via a shrink, rather than a standard gradient step
	 -- we need the sum of the absolute values of each column of the weight matrix, with each row scaled by the appropriate lagrange multiplier
	 L1_dictionary_abs_copy:resizeAs(self.L1_dictionary.weight)
	 L1_dictionary_abs_copy:abs(self.L1_dictionary.weight) 
	 current_shrink_val:resize(self.L1_dictionary.weight:size(2)) -- this should be the same size as L1_code_from_concat, but there's no need to recreate it here
	 
	 if self.use_lagrange_multiplier_cmul_mask then
	    current_shrink_val:mv(L1_dictionary_abs_copy:t(), self.lagrange_multiplier_cmul_mask)
	 else	    
	    current_shrink_val:sum(L1_dictionary_abs_copy, 1) -- WARNING: this is slightly INEFFICIENT; it would probably be best to write a new c function that performs the abs-sum directly, rather than making a copy of the L1_dictionary in order to perform the absolute value computation
	    current_shrink_val:mul(self.L1_lambda_cmul) 
	 end
      end -- calculate_nonsmooth_shrinkage_cmul
      
      return calculate_nonsmooth_shrinkage_cmul
   end

   self.calculate_nonsmooth_shrinkage_cmul = self.nonsmooth_shrinkage_cmul_factory()
   


   -- Next, we need function (g) that will be the non-smooth function
   function self.nonsmooth_cost_factory() -- create a continuation so its easy to bind a persistent copy of current_shrink_val to the function
      local current_shrink_val = torch.Tensor()
      local L1_code_from_concat_sign = torch.Tensor()
      local code_abs_L1_units = torch.Tensor() -- used to accumulate the L1 norm
      local code_abs_cmul_mask = torch.Tensor() -- used to accumulate the L1 norm

      local function nonsmooth_cost(concat_code, mode)
	 --local L2_code_from_concat = self.extract_L2_from_concat(concat_code)
	 local L1_code_from_concat = self.extract_L1_from_concat(concat_code)
	 
	 local grad_concat_code_nonsmooth = nil
	 local L1_cost = 0
	 local L1_grad_mag = 0
	 if self.shrink_L1_units then 
	    if self.use_lagrange_multiplier_L1_units then
	       code_abs_L1_units:resizeAs(L1_code_from_concat)
	       code_abs_L1_units:abs(L1_code_from_concat)
	       L1_cost = L1_cost + self.lagrange_multiplier_L1_units:dot(code_abs_L1_units)
	    else
	       L1_cost = L1_cost + self.L1_lambda * self.concat_code_L1_cost:updateOutput(L1_code_from_concat) 
	    end
	 end
	 
	 if self.shrink_L1_dictionary_outputs then
	    if self.use_lagrange_multiplier_cmul_mask then
	       -- This is *NOT* correct if any of the L1 units or L1 dictionary entries are negative
	       code_abs_cmul_mask:resizeAs(self.L1_processing_chain.output) -- WARNING: this will be INEFFICIENT if both L1 units and cmul mask are subject to lagrange multipliers
	       code_abs_cmul_mask:abs(self.L1_processing_chain.output)
	       L1_cost = L1_cost + self.lagrange_multiplier_cmul_mask:dot(code_abs_cmul_mask)
	    else
	       L1_cost = L1_cost + self.L1_lambda_cmul * self.L1_processing_chain.output:norm(1) -- THIS IS NOT CORRECT; we should actually take the absolute value of the dictionary matrix and the L1_code_from_concat before multiplying.  However, this calculation does not affect either fista or the weight update, and is correct if everything is nonnegative, so we'll leave this for now
	    end
	 end
	 
	 
	 if mode and (mode:match('dx') or (mode:match('verbose') and self.cost_output_counter >= 250)) then
	    --print('Gradient of nonsmooth cost should never be evaluated')
	    if self.shrink_L1_units then
	       if self.use_lagrange_multiplier_L1_units then
		  grad_concat_code_nonsmooth = self.concat_code_L1_cost:updateGradInput(L1_code_from_concat):cmul(self.lagrange_multiplier_L1_units) 
	       else
		  grad_concat_code_nonsmooth = self.concat_code_L1_cost:updateGradInput(L1_code_from_concat):mul(self.L1_lambda) 
	       end
	    end
	    
	    if self.shrink_L1_dictionary_outputs then
	       self.calculate_nonsmooth_shrinkage_cmul(current_shrink_val) 
	       
	       L1_code_from_concat_sign:resizeAs(L1_code_from_concat)
	       L1_code_from_concat_sign:sign(L1_code_from_concat)
	       current_shrink_val:cmul(L1_code_from_concat_sign) 
	       
	       if not(grad_concat_code_nonsmooth) then 
		  grad_concat_code_nonsmooth = current_shrink_val -- this is *only* safe so long as current_shrink_val is not used elsewhere for other computations!!!
	       else
		  grad_concat_code_nonsmooth:add(current_shrink_val)
	       end
	    end
	    
	    if mode and mode:match('verbose') and self.cost_output_counter >= 250 then
	       L1_grad_mag = grad_concat_code_nonsmooth:norm()
	       print('L1: ' .. L1_cost .. ' ; L1 grad: ' .. L1_grad_mag)
	       self.cost_output_counter = 0
	    end
	 end
	 
	 return L1_cost, grad_concat_code_nonsmooth
      end
      
      return nonsmooth_cost -- the local function, bound to the persistent local variable, is the output of the factory
   end

   self.nonsmooth_cost = self.nonsmooth_cost_factory()
   
   


   
   function self.shrinkage_factory(nonnegative_L1_units)
      local shrunk_indices = torch.ByteTensor() -- allocate this down here for clarity, so it's close to where it's used
      local shrink_sign = torch.Tensor() -- sign() returns a DoubleTensor
      local shrink_val_bounded = torch.Tensor()
      local unrepeated_shrink_val = torch.Tensor()
      if type(nonnegative_L1_units) == 'nil' then nonnegative_L1_units = true end -- use nonnegative units by default


      local function this_shrinkage(vec, shrink_val)
	 --self.shrink_le:resize(vec_size)
	 --self.shrink_ge:resize(vec_size)

	 if nonnegative_L1_units then
	    -- if any elements of shrink_val are < 0, it is essential that the corresponding elements of vec be set equal to zero
	    shrink_val_bounded:resizeAs(shrink_val)
	    shrink_val_bounded:copy(shrink_val)
	    shrink_val_bounded[torch.le(shrink_val, 0)] = 0
	    shrunk_indices = torch.le(vec, shrink_val_bounded)
	    vec:add(-1, shrink_val) -- don't worry about adding to negative values, since they will be set equal to zero by shrunk_indices
	 else
	    local vec_size = vec:size()
	    shrunk_indices:resize(vec_size)
	    shrink_sign:resize(vec_size)
	    unrepeated_shrink_val:set(shrink_val:storage())
	    
	    local shrink_le = torch.le(vec, shrink_val) -- WARNING: this is INEFFICIENT, since torch.le and torch.ge unnecessarily allocate new memory on every iteration
	    
	    unrepeated_shrink_val:mul(-1) -- if shrink_val has a stride of length 0 to repeat a constant row or column, make sure that the multiplied constant is only applied once per underlying entry

	    local shrink_ge = torch.ge(vec, shrink_val) -- WARNING: this is INEFFICIENT, since torch.le and torch.ge unnecessarily allocate new memory on every iteration
	    shrunk_indices:cmul(shrink_le, shrink_ge)
	    shrink_sign:sign(vec)

	    vec:addcmul(shrink_val, shrink_sign) -- shrink_val has already been multiplied by -1
	 end
	 
	 --shrunk_indices:cmul(torch.le(vec, shrink_val), torch.ge(vec, torch.mul(shrink_val, -1))) -- WARNING: this is INEFFICIENT, since torch.le and torch.ge unnecessarily allocate new memory on every iteration
	 --vec:addcmul(-1, shrink_val, torch.sign(vec)) -- WARNING: this is INEFFICIENT, since torch.sign unnecessarily allocates memory on every iteration
	 
	 vec[shrunk_indices] = 0
      end

      return this_shrinkage
   end
   
   if (self.shrink_L1_units and self.use_lagrange_multiplier_L1_units) or self.shrink_L1_dictionary_outputs then 
      self.L1_shrinkage = self.shrinkage_factory()
   end



   if self.shrink_L1_dictionary_outputs then
      --self.L1_dictionary_shrinkage = self.shrinkage_factory(false) -- don't enforce nonnegative weights when shrinking the L1 dictionary weights
      self.L1_dictionary_shrinkage = self.shrinkage_factory() -- DO enforce nonnegative weights when shrinking the L1 dictionary
      self.current_dictionary_shrink_val = torch.Tensor()
      self.dictionary_shrink_val_L1_code_part = torch.Tensor()
   end
   
   -- Finally we need argmin_x Q(x,y)
   function self.minimizer_factory() -- create a continuation so it's easy to bind a persistent copy of current_shrink_val to the function
      local current_shrink_val_L1 = torch.Tensor()
      local current_shrink_val_dict = torch.Tensor()

      local function minimize_nonsmooth(concat_code, L)
	 local L1_code_from_concat = self.extract_L1_from_concat(concat_code)
	 
	 if self.shrink_L1_units then
	    if self.use_lagrange_multiplier_L1_units then 
	       current_shrink_val_L1:resize(self.lagrange_multiplier_L1_units:size())
	       current_shrink_val_L1:div(self.lagrange_multiplier_L1_units, L) -- this will be multiplied by negative one in self.shrinkage, so it must be recomputed each time
	       self.L1_shrinkage(L1_code_from_concat, current_shrink_val_L1) 
	    else
	       L1_code_from_concat:shrinkage(self.L1_lambda/L) -- this uses Koray's optimized shrink which only accepts a single shrink value
	    end
	 end
	 
	 if self.shrink_L1_dictionary_outputs then
	    self.calculate_nonsmooth_shrinkage_cmul(current_shrink_val_dict) 

	    current_shrink_val_dict:div(L) 
	    self.L1_shrinkage(L1_code_from_concat, current_shrink_val_dict) 
	 end
	 
	 local nonnegative_L2_units = true
	 if nonnegative_L2_units then
	    local L2_code_from_concat = self.extract_L2_from_concat(concat_code)
	    --L2_code_from_concat:shrinkage(self.L2_lambda/(2*L))
	    L2_code_from_concat[torch.lt(L2_code_from_concat,0)] = 0 -- WARNING: this is extremely INEFFICIENT, since torch.lt() allocates new memory on each call
	 end
      end
      
      return minimize_nonsmooth -- the local function, bound to the persistent local variable, is the output of the factory
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

   if not(self.use_L1_dictionary) then -- this is probably not critical, since the L1_dictionary is not loaded into the processing chain
      self.L1_dictionary:reset("identity")
   else
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

-- we do inference in forward
function FactoredSparseCoder:updateOutput(input, icode, target)
   self.input = input
   --self.target = target
   local max_val, max_index = torch.max(target,1)
   self.target = max_index[1]

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
   self.L1_dictionary:zeroGradParameters()
   
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

-- no grad output, because we are unsup
-- d(||Ax-b||+lam||x||_1)/dA
function FactoredSparseCoder:accGradParameters(input, target)
   if self.wake_sleep_stage == 'test' then
      print('ERROR!!!  accGradParameters was called in test mode!!!')
   end


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

function FactoredSparseCoder:updateParameters(learningRate)
   local L1_lagrange_multiplier_learning_rate = learningRate
   local wake_only_learning_rate = learningRate
   local default_learning_rate = ((self.use_top_level_classifier and (self.wake_sleep_stage == 'sleep') and self.sleep_stage_learning_rate_scaling_factor) or 1) * learningRate

   if self.wake_sleep_stage == 'test' then
      print('ERROR!!!  updateParameters was called in test mode!!!')
   end

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
   -- lagrange multipliers follow the gradient, rather than the negative gradient; make sure that the gradient is followed in the same direction regardless of manipulations of the sign of learningRate based upon wake/sleep alternations
   if self.use_lagrange_multiplier_L1_units then
      self.lagrange_multiplier_L1_units:add(self.lagrange_multiplier_L1_units_learning_rate_scaling * L1_lagrange_multiplier_learning_rate * (1 + ((self.use_top_level_classifier and self.sleep_stage_learning_rate_scaling_factor) or 0)), self.lagrange_grad_L1_units) 
      
      self.lagrange_multiplier_L1_units[torch.lt(self.lagrange_multiplier_L1_units, self.lagrange_multiplier_L1_units_zeros)] = 0 -- bound the lagrange multipliers below by zero - WARNING: this is unnecessarily INEFFICIENT, since memory is allocated on each call
   end
   
   if self.use_lagrange_multiplier_cmul_mask then
      self.lagrange_multiplier_cmul_mask:add(self.lagrange_multiplier_cmul_mask_learning_rate_scaling * L1_lagrange_multiplier_learning_rate * (1 + ((self.use_top_level_classifier and self.sleep_stage_learning_rate_scaling_factor) or 0)), self.lagrange_grad_cmul_mask) 
      
      self.lagrange_multiplier_cmul_mask[torch.lt(self.lagrange_multiplier_cmul_mask, self.lagrange_multiplier_cmul_mask_zeros)] = 0 -- bound the lagrange multipliers below by zero - WARNING: this is unnecessarily INEFFICIENT, since memory is allocated on each call
   end
   

   if self.shrink_L1_dictionary_outputs and (not(self.use_top_level_classifier) or (self.wake_sleep_stage == 'wake')) then  -- MAKE SURE THAT SHRINKAGE IS NOT DONE IN REVERSE DURING THE SLEEP STAGE!!!
      local L1_code_from_concat = self.extract_L1_from_concat(self.concat_code)
      self.dictionary_shrink_val_L1_code_part:resizeAs(L1_code_from_concat)
      self.dictionary_shrink_val_L1_code_part:abs(L1_code_from_concat)
      if self.use_lagrange_multiplier_cmul_mask then 
	 -- we only shrink in the wake stage, so scale the shrinkage down by the difference between the wake and sleep stage learning rates
	 self.dictionary_shrink_val_L1_code_part:mul(wake_only_learning_rate * self.L1_dictionary_learning_rate_scaling * (1 + self.sleep_stage_learning_rate_scaling_factor))
	 self.current_dictionary_shrink_val:resizeAs(self.L1_dictionary.weight)
	 self.current_dictionary_shrink_val:ger(self.lagrange_multiplier_cmul_mask, self.dictionary_shrink_val_L1_code_part)
	 --print('shrinking by')
	 --print(self.current_dictionary_shrink_val:select(1,1):unfold(1,10,10))
	 --print('before shrinking')
	 --print(self.L1_dictionary.weight:select(1,1):unfold(1,10,10))
	 self.L1_dictionary_shrinkage(self.L1_dictionary.weight, self.current_dictionary_shrink_val)
	 --print('after shrinking')
	 --print(self.L1_dictionary.weight:select(1,1):unfold(1,10,10))
	 --io.read()
      else
	 -- we only shrink in the wake stage, so scale the shrinkage down by the difference between the wake and sleep stage learning rates
	 self.dictionary_shrink_val_L1_code_part:mul(wake_only_learning_rate * self.L1_dictionary_learning_rate_scaling * self.L1_lambda_cmul * (1 + self.sleep_stage_learning_rate_scaling_factor))  
	 local L1_units_abs_duplicate_rows = torch.Tensor(self.dictionary_shrink_val_L1_code_part:storage(), self.dictionary_shrink_val_L1_code_part:storageOffset(), self.L1_dictionary.weight:size(), torch.LongStorage{0,1}) -- this doesn't allocate new main storage, so it should be relatively efficient, even though a new Tensor view is created on each iteration
	 --print('before shrinkage')
	 --print(self.L1_dictionary.weight:select(1,1):unfold(1,10,10))
	 self.L1_dictionary_shrinkage(self.L1_dictionary.weight, L1_units_abs_duplicate_rows) -- NOTE that self.dictionary_shrink_val_L1_code_part cannot be reused after this, since it is multiplied by -1 when shrinking
	 --print('after shrinkage')
	 --print(self.L1_dictionary.weight:select(1,1):unfold(1,10,10))
	 --io.read()
      end 
   end -- shrink_L1_dictionary_outputs

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
   