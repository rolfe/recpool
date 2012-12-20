require 'image'

local function plot_training_error(t)
   gnuplot.pngfigure(params.rundir .. '/error.png')
   gnuplot.plot(avTrainingError:narrow(1,1,math.max(t/params.textstatinterval,2)))
   gnuplot.title('Training Error')
   gnuplot.xlabel('# iterations / ' .. params.textstatinterval)
   gnuplot.ylabel('Cost')
   
   -- clean up plots
   gnuplot.plotflush()
   gnuplot.closeall()
end

local function save_filter(current_filter, filter_name, log_directory, num_display_columns)
   num_display_columns = num_display_columns or 10
   local current_filter_side_length 
   if current_filter:size(1) % 3 == 0 then -- make sure that CIFAR input filters align the R, G, and B channels coherently
      current_filter_side_length = math.sqrt(current_filter:size(1)/3) 
      --current_filter = current_filter:reshape(current_filter:size(2),3,32,32) -- reshape makes a copy of the entire filter, which seems unnecessarily inefficient
      -- after unfolding, the original dimension iterates across groups; the last dimension iterates within groups
      current_filter = current_filter:unfold(1,current_filter_side_length,current_filter_side_length):unfold(1,current_filter_side_length,current_filter_side_length):transpose(1,2) -- may still need to transpose the last two dimensions!!!
      --current_filter_side_length = math.sqrt(current_filter:size(1))
   else
      current_filter_side_length = math.sqrt(current_filter:size(1))
      current_filter = current_filter:unfold(1,current_filter_side_length, current_filter_side_length):transpose(1,2)
   end
   local current_image = image.toDisplayTensor{input=current_filter,padding=1,nrow=num_display_columns,symmetric=true}
   
   -- ideally, the pdf viewer should refresh automatically.  This 
   image.savePNG(paths.concat(log_directory, filter_name .. '.png'), current_image)
end

-- dataset is nExamples x input_dim
-- hidden_activation is nExamples x hidden_dim
-- construct a dictionary matrix that optimally reconstructs the data_set from the hidden_activation
function construct_optimal_dictionary(data_set, hidden_activation, optimal_dictionary_matrix, odm_offset, odm_stride, filter_name, log_directory)
   -- gels only works properly if hidden_activation is full rank, and is unstable if hidden_activation is ill-conditioned.  Remove any hidden units that do not have sufficient activation.
   local num_active_units = 0
   local activation_norms = torch.Tensor(hidden_activation:size(2)):zero()
   for i=1,hidden_activation:size(2) do
      activation_norms[i] = hidden_activation:select(2,i):norm()
      if hidden_activation:select(2,i):norm() > 0.05 then
	 num_active_units = num_active_units + 1
      end
   end
   print('found ' .. num_active_units .. ' active units')

   -- construct a reduced version of hidden activation, which only contains the active hidden units.  Use this to reconstruct the optimal dictionary
   local conservative_hidden_activation = torch.Tensor(hidden_activation:size(1), num_active_units)
   num_active_units = 0
   for i=1,hidden_activation:size(2) do
      if hidden_activation:select(2,i):norm() > 0.05 then
	 num_active_units = num_active_units + 1
	 conservative_hidden_activation:select(2,num_active_units):copy(hidden_activation:select(2,i))
      end
   end

   local conservative_optimal_dictionary_matrix = torch.gels(data_set, conservative_hidden_activation)
   
   -- save each optimal dictionary separately - for debug only, at this point
   local optimal_dictionary_matrix_slice = torch.Tensor(hidden_activation:size(2), data_set:size(2)):zero()
   
   num_active_units = 0
   for i=1,hidden_activation:size(2) do
      if hidden_activation:select(2,i):norm() > 0.05 then
	 num_active_units = num_active_units + 1
	 optimal_dictionary_matrix:select(1,(i-1)*odm_stride + odm_offset):copy(conservative_optimal_dictionary_matrix:select(1,num_active_units)):div(conservative_optimal_dictionary_matrix:select(1,num_active_units):norm()) -- this ignores the extra rows --:narrow(1,1,hidden_activation:size(2)))
	 optimal_dictionary_matrix_slice:select(1,i):copy(conservative_optimal_dictionary_matrix:select(1,num_active_units))
      end
   end

   if filter_name and log_directory then
      save_filter(optimal_dictionary_matrix:t(), filter_name, log_directory)
   end

   print('actual error is ' .. data_set:dist(hidden_activation*optimal_dictionary_matrix_slice))
   print('predicted error is ' .. math.sqrt(conservative_optimal_dictionary_matrix:narrow(1,conservative_hidden_activation:size(2)+1,
											  conservative_hidden_activation:size(1) - conservative_hidden_activation:size(2)):pow(2):sum()))
end


function receptive_field_builder_factory(nExamples, input_size, hidden_layer_size, total_num_shrink_copies, model)
   local accumulated_inputs = {} -- array holding the (unscaled) receptive fields; initialized by the first call to accumulate_weighted_inputs
   local receptive_field_builder = {}
   local shrink_val_tensor = torch.Tensor(total_num_shrink_copies, nExamples, hidden_layer_size) -- output of the shrink nonlinearities for each element of the dataset
   local data_set_tensor = torch.Tensor(nExamples, input_size) -- accumulate the entire dataset used in the diagnostic run; this way, the analysis is correct even if we only present part of the dataset to the model
   local first_activation, num_activations = torch.Tensor(hidden_layer_size), torch.Tensor(hidden_layer_size)
   local data_set_index = 1 -- present position in the dataset

   -- helper function to build receptive fields
   function receptive_field_builder:accumulate_weighted_inputs(input_tensor, weight_tensor, accumulated_inputs_index)
      if input_tensor:nDimension() == 1 then -- inputs and weights are vectors; we aren't using minibatches
	 if not(accumulated_inputs[accumulated_inputs_index]) then
	    accumulated_inputs[accumulated_inputs_index] = torch.ger(input_tensor, weight_tensor)
	 else
	    accumulated_inputs[accumulated_inputs_index]:addr(input_tensor, weight_tensor)
	 end
      else
	 if not(accumulated_inputs[accumulated_inputs_index]) then
	    accumulated_inputs[accumulated_inputs_index] = torch.mm(input_tensor:t(), weight_tensor)
	 else
	    accumulated_inputs[accumulated_inputs_index]:addmm(input_tensor:t(), weight_tensor)
	 end
      end
   end
   
   -- this is the interface ot the outside world
   function receptive_field_builder:accumulate_shrink_weighted_inputs(new_input, base_shrink, shrink_copies)
      local batch_size = new_input:size(1)
      if data_set_index >= nExamples then
	 error('accumulated ' .. data_set_index .. ' elements in the receptive field builder, but only expected ' .. nExamples)
      end

      data_set_tensor:narrow(1,data_set_index,batch_size):copy(new_input) -- copy the input values from the dataset

      self:accumulate_weighted_inputs(new_input, base_shrink.output, 1) -- accumulate the linear receptive fields
      shrink_val_tensor:select(1,1):narrow(1,data_set_index,batch_size):copy(base_shrink.output) -- copy the hidden unit values
      for i = 1,#shrink_copies do
	 self:accumulate_weighted_inputs(new_input, shrink_copies[i].output, i+1)
	 shrink_val_tensor:select(1,i+1):narrow(1,data_set_index,batch_size):copy(shrink_copies[i].output)
      end

      data_set_index = data_set_index + batch_size
   end
   
   function receptive_field_builder:extract_receptive_fields(index)
      local receptive_field_output = accumulated_inputs[index]:clone()
      for i = 1,receptive_field_output:size(2) do
	 local selected_col = receptive_field_output:select(2,i)
	 selected_col:div(selected_col:norm())
      end
      return receptive_field_output
   end
   
   function receptive_field_builder:plot_receptive_fields(opt)
      --shrink_val_tensor:select(2,nExamples+1):zero()
      --data_set_tensor:select(1,nExamples+1):fill(1)

      -- show evolution of optimal dictionaries in a single figure
      local optimal_dictionary_matrix = torch.Tensor(shrink_val_tensor:size(3) * shrink_val_tensor:size(1), data_set_tensor:size(2)):zero()
      for i = 1,#accumulated_inputs do
	 local receptive_field_output = self:extract_receptive_fields(i)
	 save_filter(receptive_field_output, 'shrink receptive field ' .. i, opt.log_directory)
	 construct_optimal_dictionary(data_set_tensor, shrink_val_tensor:select(1,i), optimal_dictionary_matrix, i, shrink_val_tensor:size(1), 'shrink dictionary ' .. i, opt.log_directory)
      end
      save_filter(optimal_dictionary_matrix:t(), 'shrink dictionary', opt.log_directory, shrink_val_tensor:size(1))

      local activated_at_zero = torch.gt(shrink_val_tensor:select(1,1), 0):double():sum(1):select(1,1)
      local activated_at_one = torch.add(torch.gt(shrink_val_tensor:select(1,2), 0):double(), -1, torch.gt(shrink_val_tensor:select(1,1), 0):double()):maxZero():sum(1):select(1,1)
      local activated_at_end = torch.gt(shrink_val_tensor:select(1,shrink_val_tensor:size(1)), 0):double():sum(1):select(1,1)
      --local activated_after_zero = torch.gt(shrink_val_tensor:narrow(1,2,total_num_shrink_copies-1):sum(1):select(1,1), 0):double():sum(1):select(1,1) -- works since activities are non-negative
      local activated_ever = torch.gt(shrink_val_tensor:sum(1):select(1,1), 0):double():sum(1):select(1,1) -- works since activities are non-negative
      -- activated after zero but not at zero = activated_ever - activated_at_zero
      activated_ever[torch.le(activated_ever, 1)] = 1
      local safe_activated_at_end = activated_at_end:clone()
      safe_activated_at_end[torch.le(activated_at_end, 1)] = 1
      local average_value_when_activated = torch.sum(shrink_val_tensor:select(1,shrink_val_tensor:size(1)), 1):select(1,1):cdiv(safe_activated_at_end)

      local percentage_late_activation = torch.cdiv(torch.add(activated_ever, -1, activated_at_zero), activated_ever)
      local percentage_first_iter_activation = torch.cdiv(activated_at_zero, activated_ever)
      local percentage_second_iter_activation = torch.cdiv(activated_at_one, activated_ever)
      local percentage_activated_at_end = torch.div(activated_at_end, shrink_val_tensor:size(2))
      --print('percentage late activation', percentage_late_activation:unfold(1,10,10))

      require 'gnuplot'
      local norm_vec = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      local enc_norm_vec = torch.Tensor(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight:size(1))
      local dec_norm_vec = torch.Tensor(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:size(2))
      local classification_norm_vec = torch.Tensor(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:size(2))
      local prod_norm_vec = torch.Tensor(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:size(2))
      local average_recurrent_pos_connection_angle = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      local average_recurrent_neg_connection_angle = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      local average_recurrent_pos_connection_categoricalness = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      local average_recurrent_neg_connection_categoricalness = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      local average_recurrent_total_connection_categoricalness = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      local average_recurrent_part_connection_angle = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      local average_recurrent_categorical_connection_angle = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      local average_recurrent_categorical_connection_angle_mod = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))

      local deviation_of_recurrent_weight_from_ISTA = torch.Tensor(model.layers[1].module_list.explaining_away.weight:nElement()):fill(-100)
      local deviation_of_recurrent_weight_from_ISTA_just_parts_inputs = torch.Tensor(model.layers[1].module_list.explaining_away.weight:nElement()):fill(-100)
      local categoricalness_of_recurrent_weight_recipient = torch.Tensor(model.layers[1].module_list.explaining_away.weight:nElement()):fill(-100)

      local dot_product_between_decoders_per_connection_from_part_to_part = torch.Tensor(model.layers[1].module_list.explaining_away.weight:nElement()):fill(-100)
      local dot_product_between_decoders_per_connection_from_categorical_to_part = torch.Tensor(model.layers[1].module_list.explaining_away.weight:nElement()):fill(-100)
      local dot_product_between_decoders_per_connection_from_part_to_categorical = torch.Tensor(model.layers[1].module_list.explaining_away.weight:nElement()):fill(-100)
      local dot_product_between_decoders_per_connection_from_categorical_to_categorical = torch.Tensor(model.layers[1].module_list.explaining_away.weight:nElement()):fill(-100)
      local angle_between_classifiers_per_connection_from_categorical_to_categorical = torch.Tensor(model.layers[1].module_list.explaining_away.weight:nElement()):fill(-100)
      local weight_of_connections_from_part_to_part = torch.Tensor(model.layers[1].module_list.explaining_away.weight:nElement()):fill(-100)
      local weight_of_connections_from_categorical_to_part = torch.Tensor(model.layers[1].module_list.explaining_away.weight:nElement()):fill(-100)
      local weight_of_connections_from_part_to_categorical = torch.Tensor(model.layers[1].module_list.explaining_away.weight:nElement()):fill(-100)
      local weight_of_connections_from_categorical_to_categorical = torch.Tensor(model.layers[1].module_list.explaining_away.weight:nElement()):fill(-100)

      local cwm_pc_num_bins = 100
      local connection_weight_means_part_to_categorical = torch.Tensor(cwm_pc_num_bins):zero()
      local connection_weight_counts_part_to_categorical = torch.Tensor(cwm_pc_num_bins):zero()
      local connection_weight_dot_products_part_to_categorical = torch.linspace(-1,1,cwm_pc_num_bins)


      --torch.diag(torch.mm(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight, model.layers[1].module_list.decoding_feature_extraction_dictionary.weight)), 
	 
      for i = 1,model.layers[1].module_list.explaining_away.weight:size(1) do
	 norm_vec[i] = model.layers[1].module_list.explaining_away.weight:select(1,i):norm()
	 enc_norm_vec[i] = model.layers[1].module_list.encoding_feature_extraction_dictionary.weight:select(1,i):norm()
	 dec_norm_vec[i] = model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:select(2,i):norm()
	 classification_norm_vec[i] = model.module_list.classification_dictionary.weight:select(2,i):norm()
	 prod_norm_vec[i] = torch.dot(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight:select(1,i), 
				      model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:select(2,i))
      end
      --print(norm_vec:unfold(1,10,10))
      local angle_between_encoder_and_decoder = torch.cdiv(prod_norm_vec, torch.cmul(enc_norm_vec, dec_norm_vec)):acos()

      for i = 1,model.layers[1].module_list.explaining_away.weight:size(1) do
	 local pos_norm, neg_norm, pos_weighted_sum_angle, neg_weighted_sum_angle, pos_weighted_sum_categoricalness, neg_weighted_sum_categoricalness = 0, 0, 0, 0, 0, 0
	 local part_norm, categorical_norm, part_weighted_sum_angle, categorical_weighted_sum_angle, categorical_weighted_sum_angle_mod = 0, 0, 0, 0, 0
	 local sorted_recurrent_weights = torch.abs(model.layers[1].module_list.explaining_away.weight:select(1,i)):sort()
	 local median_abs_weight = sorted_recurrent_weights[math.ceil(sorted_recurrent_weights:size(1) * (97.5/100))]
	 --print(median_abs_weight)
      	 for j = 1,model.layers[1].module_list.explaining_away.weight:size(2) do
	    local dot_product_between_decoders = torch.dot(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:select(2,i), 
							   model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:select(2,j))
	    local angle_between_classifiers = torch.dot(model.module_list.classification_dictionary.weight:select(2,i), 
							model.module_list.classification_dictionary.weight:select(2,j)) / (classification_norm_vec[i] * classification_norm_vec[j])

	    local exp_away_linearized_index = j + (i-1)*model.layers[1].module_list.explaining_away.weight:size(2)
	    --deviation_of_recurrent_weight_from_ISTA[exp_away_linearized_index] = math.max(-3, math.min(3, -1 * model.layers[1].module_list.explaining_away.weight[{i,j}] + (1.25/11)*dot_product_between_decoders)) -- - (((i == j) and 1) or 0)))
	    deviation_of_recurrent_weight_from_ISTA[exp_away_linearized_index] = math.max(-0.5, math.min(2, -1 * (((math.abs(model.layers[1].module_list.explaining_away.weight[{i,j}]) > median_abs_weight) and 1) or 0) * model.layers[1].module_list.explaining_away.weight[{i,j}] / dot_product_between_decoders))
	    deviation_of_recurrent_weight_from_ISTA_just_parts_inputs[exp_away_linearized_index] = math.max(-0.5, math.min(2, -1 * (((math.abs(model.layers[1].module_list.explaining_away.weight[{i,j}]) > median_abs_weight) and 1) or 0) * (((angle_between_encoder_and_decoder[j] < 0.55) and 1) or 0) * model.layers[1].module_list.explaining_away.weight[{i,j}] / dot_product_between_decoders))
	    categoricalness_of_recurrent_weight_recipient[exp_away_linearized_index] = angle_between_encoder_and_decoder[i]

	    local cwm_bin = math.floor(cwm_pc_num_bins * (dot_product_between_decoders + 1) / 2)
	    if (angle_between_encoder_and_decoder[i] > 0.7) and (angle_between_encoder_and_decoder[j] < 0.5) then
	       connection_weight_means_part_to_categorical[cwm_bin] = connection_weight_means_part_to_categorical[cwm_bin] + model.layers[1].module_list.explaining_away.weight[{i,j}]
	       connection_weight_counts_part_to_categorical[cwm_bin] = connection_weight_counts_part_to_categorical[cwm_bin] + 1
	    end

	    dot_product_between_decoders_per_connection_from_part_to_part[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] < 0.5) and 1) or 0) * (((angle_between_encoder_and_decoder[j] < 0.5) and 1) or 0) * dot_product_between_decoders
	    weight_of_connections_from_part_to_part[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] < 0.5) and 1) or 0) * (((angle_between_encoder_and_decoder[j] < 0.5) and 1) or 0) * model.layers[1].module_list.explaining_away.weight[{i,j}]

	    dot_product_between_decoders_per_connection_from_categorical_to_part[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] < 0.5) and 1) or 0) * (((angle_between_encoder_and_decoder[j] > 0.7) and 1) or 0) * dot_product_between_decoders
	    weight_of_connections_from_categorical_to_part[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] < 0.5) and 1) or 0) * (((angle_between_encoder_and_decoder[j] > 0.7) and 1) or 0) * model.layers[1].module_list.explaining_away.weight[{i,j}]

	    dot_product_between_decoders_per_connection_from_part_to_categorical[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] > 0.7) and 1) or 0) * (((angle_between_encoder_and_decoder[j] < 0.5) and 1) or 0) * dot_product_between_decoders
	    weight_of_connections_from_part_to_categorical[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] > 0.7) and 1) or 0) * (((angle_between_encoder_and_decoder[j] < 0.5) and 1) or 0) * model.layers[1].module_list.explaining_away.weight[{i,j}]

	    dot_product_between_decoders_per_connection_from_categorical_to_categorical[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] > 0.7) and 1) or 0) * (((angle_between_encoder_and_decoder[j] > 0.7) and 1) or 0) * dot_product_between_decoders
	    weight_of_connections_from_categorical_to_categorical[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] > 0.7) and 1) or 0) * (((angle_between_encoder_and_decoder[j] > 0.7) and 1) or 0) * model.layers[1].module_list.explaining_away.weight[{i,j}]
	    angle_between_classifiers_per_connection_from_categorical_to_categorical[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] > 0.7) and 1) or 0) * (((angle_between_encoder_and_decoder[j] > 0.7) and 1) or 0) * angle_between_classifiers

	    if i ~= j then -- ignore the diagonal
	       local val_angle = math.abs(model.layers[1].module_list.explaining_away.weight[{i,j}]) *
		  math.acos(dot_product_between_decoders / (dec_norm_vec[i] * dec_norm_vec[j]))
	       local val_categoricalness = math.abs(model.layers[1].module_list.explaining_away.weight[{i,j}]) * angle_between_encoder_and_decoder[j]
	       
	       if model.layers[1].module_list.explaining_away.weight[{i,j}] >= 0 then
		  pos_weighted_sum_angle = pos_weighted_sum_angle + val_angle
		  pos_weighted_sum_categoricalness = pos_weighted_sum_categoricalness + val_categoricalness
		  pos_norm = pos_norm + math.abs(model.layers[1].module_list.explaining_away.weight[{i,j}])
	       else
		  neg_weighted_sum_angle = neg_weighted_sum_angle + val_angle
		  neg_weighted_sum_categoricalness = neg_weighted_sum_categoricalness + val_categoricalness
		  neg_norm = neg_norm + math.abs(model.layers[1].module_list.explaining_away.weight[{i,j}])
	       end

	       if angle_between_encoder_and_decoder[j] < 0.5 then
		  part_weighted_sum_angle = part_weighted_sum_angle + 
		     model.layers[1].module_list.explaining_away.weight[{i,j}] * (math.pi/2 - math.acos(dot_product_between_decoders / (dec_norm_vec[i] * dec_norm_vec[j])))
		  part_norm = part_norm + math.abs(model.layers[1].module_list.explaining_away.weight[{i,j}])
	       elseif angle_between_encoder_and_decoder[j] > 0.7 then
		  categorical_weighted_sum_angle = categorical_weighted_sum_angle + 
		     model.layers[1].module_list.explaining_away.weight[{i,j}] * (math.pi/4 - math.acos(dot_product_between_decoders / (dec_norm_vec[i] * dec_norm_vec[j])))
		  categorical_weighted_sum_angle_mod = categorical_weighted_sum_angle_mod + 
		     model.layers[1].module_list.explaining_away.weight[{i,j}] * (math.pi/2 - math.acos(dot_product_between_decoders / (dec_norm_vec[i] * dec_norm_vec[j])))
		  categorical_norm = categorical_norm + math.abs(model.layers[1].module_list.explaining_away.weight[{i,j}])
	       end
	    end
	 end
	 pos_norm = (((pos_norm == 0) and 1) or pos_norm)
	 neg_norm = (((neg_norm == 0) and 1) or neg_norm)
	 average_recurrent_pos_connection_angle[i] = pos_weighted_sum_angle / pos_norm
	 average_recurrent_neg_connection_angle[i] = neg_weighted_sum_angle / neg_norm
	 average_recurrent_pos_connection_categoricalness[i] = pos_weighted_sum_categoricalness / pos_norm
	 average_recurrent_neg_connection_categoricalness[i] = neg_weighted_sum_categoricalness / neg_norm
	 average_recurrent_total_connection_categoricalness[i] = (pos_weighted_sum_categoricalness + neg_weighted_sum_categoricalness) / (pos_norm + neg_norm)
	 part_norm = (((part_norm == 0) and 1) or part_norm)
	 categorical_norm = (((categorical_norm == 0) and 1) or categorical_norm)
	 average_recurrent_part_connection_angle[i] = part_weighted_sum_angle / part_norm
	 average_recurrent_categorical_connection_angle[i] = categorical_weighted_sum_angle / categorical_norm
	 average_recurrent_categorical_connection_angle_mod[i] = categorical_weighted_sum_angle_mod / categorical_norm
      end		  	  

      connection_weight_counts_part_to_categorical[torch.lt(connection_weight_counts_part_to_categorical, 1)] = 1

      local norm_classification_connection = torch.Tensor(model.module_list.classification_dictionary.weight:size(2))
      for i = 1,model.module_list.classification_dictionary.weight:size(2) do
	 norm_classification_connection[i] = model.module_list.classification_dictionary.weight:select(2,i):norm()
      end

      gnuplot.figure() -- percentage of inputs for which the unit is activated at some point, but the first activation occurs after the first iteration; versus the magnitude of the recurrent connections; categorical units turn on later, since they have poorly structured encoder inputs but strong connections to part-units.
      gnuplot.plot(angle_between_encoder_and_decoder, percentage_late_activation)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('prob of late activation')

      gnuplot.figure() -- percentage of inputs for which the unit is activated at some point, but the first activation occurs at the second iteration; versus the magnitude of the recurrent connections; categorical units turn on later, since they have poorly structured encoder inputs but strong connections to part-units.
      gnuplot.plot(angle_between_encoder_and_decoder, percentage_second_iter_activation)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('prob of second iter activation')

      gnuplot.figure() -- percentage of inputs for which the unit is activated at the end
      gnuplot.plot(angle_between_encoder_and_decoder, percentage_activated_at_end)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('prob activated at end')

      gnuplot.figure() -- histogram of recurrent connections; categorical units have larger recurrent connections
      gnuplot.hist(norm_vec, 50)
      
      gnuplot.figure() -- mean recurrent connections versus magnitude of recurrent connections; categorical units have more negative and larger recurrent connections (this is actually a little counterintuitive, since categorical units derive most of their excitation from recurrent connections; presumably, they perform an and-not computation, and there are many units that can veto the activity of a given categorical unit; the nature of this computation will be explicated by plotting the dictionaries of the largest recurrent connections to each unit
      gnuplot.plot(angle_between_encoder_and_decoder,
		   torch.add(model.layers[1].module_list.explaining_away.weight, torch.diag(torch.ones(model.layers[1].module_list.explaining_away.weight:size(1)))):mean(1):select(1,1))
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('recurrent connection mean')
      
      gnuplot.figure() -- recurrent connection diagonal versus categoricalness
      gnuplot.plot(angle_between_encoder_and_decoder, torch.diag(model.layers[1].module_list.explaining_away.weight))
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('explaining away matrix diagonal')

      
      --[[
      gnuplot.figure() -- mean recurrent connections excluding diagonal versus magnitude of recurrent connections; categorical units have more negative and larger recurrent connections (this is actually a little counterintuitive, since categorical units derive most of their excitation from recurrent connections; presumably, they perform an and-not computation, and there are many units that can veto the activity of a given categorical unit; the nature of this computation will be explicated by plotting the dictionaries of the largest recurrent connections to each unit
      gnuplot.plot(angle_between_encoder_and_decoder,
		   torch.add(model.layers[1].module_list.explaining_away.weight, -1, torch.diag(torch.diag(model.layers[1].module_list.explaining_away.weight))):mean(1):select(1,1))
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('recurrent connection mean without diagonal')
      --]]

      gnuplot.figure() -- mean decoder column versus categoricalness
      gnuplot.plot(angle_between_encoder_and_decoder,
		   model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:mean(1):select(1,1)) -- argument to mean is the dimension collapsed
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('decoder mean')

      gnuplot.figure() -- mean decoder column versus categoricalness
      gnuplot.plot(angle_between_encoder_and_decoder,
		   model.layers[1].module_list.encoding_feature_extraction_dictionary.weight:mean(2):select(2,1)) -- argument to mean is the dimension collapsed
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('encoder mean')

      
      gnuplot.figure() -- cos(angle) between encoder and decoder versus magnitude of recurrent input; categorical units have unaligned encoder/decoder pairs and larger recurrent connections
      gnuplot.plot(angle_between_encoder_and_decoder, norm_vec)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('recurrent connection magnitude')

      gnuplot.figure() -- cos(angle) between encoder and decoder versus magnitude of recurrent input; categorical units have unaligned encoder/decoder pairs and larger recurrent connections
      gnuplot.plot(angle_between_encoder_and_decoder, average_recurrent_pos_connection_angle)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('weighted average angle between decoder and positively recurrently connected decoders')

      gnuplot.figure() -- cos(angle) between encoder and decoder versus magnitude of recurrent input; categorical units have unaligned encoder/decoder pairs and larger recurrent connections
      gnuplot.plot(angle_between_encoder_and_decoder, average_recurrent_neg_connection_angle)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('weighted average angle between decoder and negatively recurrently connected decoders')

      gnuplot.figure() 
      gnuplot.plot(angle_between_encoder_and_decoder, average_recurrent_part_connection_angle)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('weighted average angle between decoder and part-restricted decoders')

      gnuplot.figure() 
      gnuplot.plot(angle_between_encoder_and_decoder, average_recurrent_categorical_connection_angle)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('weighted average angle between decoder and categorical-restricted decoders')

      --[[ this doesn't work as well as the pi/4 version above
      gnuplot.figure() 
      gnuplot.plot(angle_between_encoder_and_decoder, average_recurrent_categorical_connection_angle_mod)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('weighted average angle between decoder and categorical-restricted decoders - pi/2')
      --]]

      gnuplot.figure() 
      gnuplot.plot(angle_between_encoder_and_decoder, norm_classification_connection)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('classification dictionary connection magnitude')

      --[[
      gnuplot.figure() -- cos(angle) between encoder and decoder versus magnitude of recurrent input; categorical units have unaligned encoder/decoder pairs and larger recurrent connections
      gnuplot.plot(angle_between_encoder_and_decoder, average_recurrent_pos_connection_categoricalness)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('weighted average categoricalness between decoder and positively recurrently connected decoders')

      gnuplot.figure() -- cos(angle) between encoder and decoder versus magnitude of recurrent input; categorical units have unaligned encoder/decoder pairs and larger recurrent connections
      gnuplot.plot(angle_between_encoder_and_decoder, average_recurrent_neg_connection_categoricalness)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('weighted average categoricalness between decoder and negatively recurrently connected decoders')
      --]]

      gnuplot.figure() -- cos(angle) between encoder and decoder versus magnitude of recurrent input; categorical units have unaligned encoder/decoder pairs and larger recurrent connections
      gnuplot.plot(angle_between_encoder_and_decoder, average_recurrent_total_connection_categoricalness)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('weighted average categoricalness between decoder and total recurrently connected decoders')

      print(angle_between_encoder_and_decoder:unfold(1,10,10))
      print(average_value_when_activated:unfold(1,10,10))
      print(average_value_when_activated:size())

      gnuplot.figure() 
      gnuplot.plot(angle_between_encoder_and_decoder, average_value_when_activated)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('average final value of unit when activated')

      gnuplot.figure() 
      gnuplot.plot(norm_classification_connection, average_value_when_activated)
      gnuplot.xlabel('classification_dictionary_connection_magnitude')
      gnuplot.ylabel('average final value of unit when activated')

      gnuplot.figure() 
      gnuplot.plot(categoricalness_of_recurrent_weight_recipient, deviation_of_recurrent_weight_from_ISTA, '.')
      gnuplot.xlabel('categoricalness of recurrent weight recipient')
      gnuplot.ylabel('ratio between recurrent weight and ISTA ideal')

      --[[
      gnuplot.figure() 
      gnuplot.plot(categoricalness_of_recurrent_weight_recipient, deviation_of_recurrent_weight_from_ISTA_just_parts_inputs)
      gnuplot.xlabel('categoricalness of recurrent weight recipient')
      gnuplot.ylabel('ratio between recurrent weight and ISTA ideal restricted to parts inputs')
      --]]

      gnuplot.figure() 
      gnuplot.plot(dot_product_between_decoders_per_connection_from_part_to_part, weight_of_connections_from_part_to_part, '.')
      gnuplot.xlabel('dot product between decoders from part to part')
      gnuplot.ylabel('connection weight')

      gnuplot.figure() 
      gnuplot.plot(dot_product_between_decoders_per_connection_from_categorical_to_part, weight_of_connections_from_categorical_to_part, '.')
      gnuplot.xlabel('dot product between decoders from categorical to part')
      gnuplot.ylabel('connection weight')

      gnuplot.figure() 
      gnuplot.plot(dot_product_between_decoders_per_connection_from_part_to_categorical, weight_of_connections_from_part_to_categorical, '.')
      gnuplot.xlabel('dot product between decoders from part to categorical')
      gnuplot.ylabel('connection weight')

      gnuplot.figure() 
      gnuplot.plot(dot_product_between_decoders_per_connection_from_categorical_to_categorical, weight_of_connections_from_categorical_to_categorical, '.')
      gnuplot.xlabel('dot product between decoders from categorical to categorical')
      gnuplot.ylabel('connection weight')

      gnuplot.figure() 
      gnuplot.plot(connection_weight_dot_products_part_to_categorical, connection_weight_means_part_to_categorical:cdiv(connection_weight_counts_part_to_categorical))
      gnuplot.xlabel('dot product between decoders from part to categorical')
      gnuplot.ylabel('average connection weight')

      gnuplot.figure() 
      gnuplot.plot(angle_between_classifiers_per_connection_from_categorical_to_categorical, weight_of_connections_from_categorical_to_categorical, '.')
      gnuplot.xlabel('cos(angle) between classifiers from categorical to categorical')
      gnuplot.ylabel('connection weight')





      plot_reconstruction_connections(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight, shrink_val_tensor:select(1,shrink_val_tensor:size(1)):narrow(1,1,200), data_set_tensor:narrow(1,1,200), opt, 20)

      --[[
      first_activation:zero()
      num_activations:zero()
      --total_num_shrink_copies, nExamples, hidden_layer_size
      for i = 1,hidden_layer_size do
	 for j = 1,nExamples do
	    for k = 1,total_num_shrink_copies do
	       if shrink_val_tensor[{k,j,i}] > 0 then
		  --first_activation[i] = first_activation[i] + (((k == 1) and 0) or 1) --k-1
		  if k > 1 then
		     first_activation[i] = first_activation[i] + 1
		  end
		  num_activations[i] = num_activations[i] + 1
		  break
	       end
	    end
	 end
      end
      num_activations[torch.le(num_activations, 1)] = 1
      --first_activation:cdiv(num_activations)
      print(torch.cdiv(first_activation, num_activations):unfold(1,10,10))

      print(torch.add(activated_ever, -1, activated_at_zero):unfold(1,10,10))
      print(first_activation:unfold(1,10,10))

      print(activated_ever:unfold(1,10,10))
      print(num_activations:unfold(1,10,10))
      --]]
   end
   
   function receptive_field_builder:reset()
      data_set_index = 0
      for i = 1,#accumulated_inputs do
	 accumulated_inputs[i]:zero()
      end
   end
   
   return receptive_field_builder
end

-- plot the decoding dictionaries of the top n largest magnitude connections to each unit, scaled by the connection weight.  This gives a sense of how each unit's activation is computed based from the other units
function plot_explaining_away_connections(decoding_filter, explaining_away_filter, opt, restrict_to_positive)
   local num_sorted_inputs = 20
   local explaining_away_mag_filter = explaining_away_filter:clone():add(torch.diag(torch.ones(explaining_away_filter:size(2)))) -- correct for the fact that the identity matrix needs to be added in manually
   local file_name
   if restrict_to_positive then
      explaining_away_mag_filter:maxZero()
      file_name = 'positive sorted recurrent connections'
   else
      explaining_away_mag_filter:abs()
      file_name = 'sorted recurrent connections'
   end
   local explaining_away_mag_filter_sorted, explaining_away_mag_filter_sort_indices = explaining_away_mag_filter:sort(2, true)
   local desired_indices = explaining_away_mag_filter_sort_indices:narrow(2,1,num_sorted_inputs)
   --local flattened_desired_indices = desired_indices:reshape(desired_indices:nElement()) -- takes element row-wise, and it is the rows that have been sorted and narrowed
   local sorted_recurrent_connection_filter = torch.Tensor(decoding_filter:size(1), (num_sorted_inputs + 1) * explaining_away_filter:size(1))
   local output_filter_index = 1
   for i = 1,desired_indices:size(1) do
      -- the first column is the decoder of the selected unit
      local current_column = sorted_recurrent_connection_filter:select(2,output_filter_index)
      current_column:copy(decoding_filter:select(2,i)):mul(0.1)
      output_filter_index = output_filter_index + 1

      -- the remaining columns are the decoders of the units connected to it, sorted and scaled by connection strength
      for j = 1,desired_indices:size(2) do
	 current_column = sorted_recurrent_connection_filter:select(2,output_filter_index)
	 current_column:copy(decoding_filter:select(2,desired_indices[{i,j}]))
	 current_column:mul(explaining_away_filter[{i,desired_indices[{i,j}]}]) -- + (((i == desired_indices[{i,j}]) and 1) or 0))
	 output_filter_index = output_filter_index + 1
      end
   end
   save_filter(sorted_recurrent_connection_filter, file_name, opt.log_directory, num_sorted_inputs + 1) -- this depends upon save_filter using a row length of num_sorted_inputs!!!
end

-- plot the decoding dictionaries of the n units with the largest activations in response to each network input, scaled by the activation strength.  This gives a sense of how each input is reconstructed by the hidden activity
function plot_reconstruction_connections(decoding_filter, activation_tensor, input_tensor, opt, num_display_columns)
   -- different network inputs (elements of the dataset) go along the first dimension of activation_tensor
   -- different hidden units go along the second dimension of activation_tensor
   num_display_columns = num_display_columns or 10

   if activation_tensor:size(1) ~= input_tensor:size(1) then
      error('number of data set elements in activation tensor ' .. activation_tensor:size(1) .. ' does not match the number in input tensor ' .. input_tensor:size(1))
   end
   local num_sorted_inputs = num_display_columns - 2
   local activation_mag_tensor = activation_tensor:clone():abs() -- this probably isn't necessary, since units are non-negative by default
   local activation_mag_tensor_sorted, activation_mag_tensor_sort_indices = activation_mag_tensor:sort(2, true)
   local desired_indices = activation_mag_tensor_sort_indices:narrow(2,1,num_sorted_inputs)
   --local flattened_desired_indices = desired_indices:reshape(desired_indices:nElement()) -- takes element row-wise, and it is the rows that have been sorted and narrowed
   local sorted_reconstruction_filter = torch.Tensor(decoding_filter:size(1), num_display_columns * activation_tensor:size(1))
   local output_filter_index = 1
   for i = 1,desired_indices:size(1) do
      local current_column = sorted_reconstruction_filter:select(2,output_filter_index)
      current_column:copy(input_tensor:select(1,i))
      output_filter_index = output_filter_index + 1

      current_column = sorted_reconstruction_filter:select(2,output_filter_index)
      torch.mv(current_column, decoding_filter, activation_tensor:select(1,i))
      output_filter_index = output_filter_index + 1

      
      for j = 1,desired_indices:size(2) do
	 current_column = sorted_reconstruction_filter:select(2,output_filter_index)
	 current_column:copy(decoding_filter:select(2,desired_indices[{i,j}]))
	 current_column:mul(activation_tensor[{i,desired_indices[{i,j}]}]) 
	 output_filter_index = output_filter_index + 1
      end
   end
   save_filter(sorted_reconstruction_filter, 'sorted reconstruction dictionary columns', opt.log_directory, num_display_columns) -- this depends upon save_filter using a row length of num_sorted_inputs!!!
end

function plot_filters(opt, time_index, filter_list, filter_enc_dec_list, filter_name_list)
   for i = 1,#filter_list do
      local current_filter
      if filter_enc_dec_list[i] == 'encoder' then
	 current_filter = filter_list[i]:transpose(1,2)
	 --print('processing ' .. filter_name_list[i] .. ' as a encoder', current_filter:size())
	 if filter_name_list[i] == 'encoding pooling dictionary_1' then
	    --print('combining ' .. filter_name_list[i-2] .. ' with ' .. filter_name_list[i])
	    --print(filter_list[i-2]:size(), filter_list[i]:transpose(1,2):size())
	    save_filter(torch.mm(filter_list[i-2], filter_list[i]:transpose(1,2)), 'encoder reconstruction', opt.log_directory)
	 elseif filter_name_list[i] == 'explaining away_1' then
	    save_filter(torch.mm(filter_list[i-2]:transpose(1,2), filter_list[i]:transpose(1,2)), 'explaining away reconstruction', opt.log_directory)
	    --save_filter(torch.mm(filter_list[i-2]:transpose(1,2), torch.add(filter_list[i]:transpose(1,2), torch.diag(torch.ones(filter_list[i]:size(1))))), 'explaining away reconstruction', opt.log_directory)
	    save_filter(torch.mm(filter_list[i-1], filter_list[i]:transpose(1,2)), 'explaining away dec reconstruction', opt.log_directory)
	    --save_filter(torch.mm(filter_list[i-1], torch.add(filter_list[i]:transpose(1,2), torch.diag(torch.ones(filter_list[i]:size(1))))), 'explaining away dec reconstruction', opt.log_directory)
	    --save_filter(torch.mm(filter_list[i-1], torch.add(filter_list[i]:transpose(1,2), -1, torch.diag(torch.diag(filter_list[i]:transpose(1,2))))), 'explaining away dec reconstruction', opt.log_directory)
	 end
      elseif filter_enc_dec_list[i] == 'decoder' then
	 current_filter = filter_list[i]
	 --print('processing ' .. filter_name_list[i] .. ' as a decoder', current_filter:size())
	 if filter_name_list[i] == 'decoding pooling dictionary_1' then
	    print('combining ' .. filter_name_list[i-3] .. ' with ' .. filter_name_list[i])
	    --print(filter_list[i-2]:size(), filter_list[i]:transpose(1,2):size())
	    save_filter(torch.mm(filter_list[i-3], filter_list[i]), 'decoder reconstruction', opt.log_directory)
	 end
      else
	 error('filter_enc_dec_list[' .. i .. '] was incorrectly set to ' .. filter_enc_dec_list[i])
      end
      save_filter(current_filter, filter_name_list[i], opt.log_directory)

      --gnuplot.figure(i)
      --gnuplot.imagesc(current_image)
      --gnuplot.title(filter_name_list[i])
      
      if time_index % 1 == 0 then
	 --image.savePNG(paths.concat(opt.log_directory, filter_name_list[i] .. '_' .. time_index .. '.png'), current_image)
      end
   end
   --gnuplot.plotflush()
end

function plot_reconstructions(opt, input, output)
   local image_list = {input, output}
   local current_image

   for i = 1,#image_list do
      local current_image_side_length = math.sqrt(image_list[i]:size(1))
      current_image = image_list[i]:unfold(1,current_image_side_length, current_image_side_length)
      gnuplot.figure(i)
      gnuplot.imagesc(current_image)
   end
   
   gnuplot.plotflush()
end




function save_parameters(flattened_parameters, directory_name, iteration)
   -- The parameters have already been flattened by the trainer.  Flattening them again would move the parameters to a new flattened tensor.  This would be a Bad Thing.

   -- store model
   print('starting to store model')
   local mf = torch.DiskFile(directory_name .. '/model_' .. iteration .. '.bin','w'):binary()
   print('about to writeObject')
   mf:writeObject(flattened_parameters)
   print('about to close')
   mf:close()
   print('finished storing model')

   --flattened_parameters = nil
   --collectgarbage()
end

function save_performance_history(performance, directory_name, iteration)
   print('starting to store performance')
   local mf = torch.DiskFile(directory_name .. '/performance_history.txt','rw'):ascii() -- ascii is redundant, since it is the default
   mf:seekEnd()
   print('about to write performance')
   mf:writeString(iteration .. ', ' .. performance .. '\n')
   print('about to close')
   mf:close()
   print('finished storing performance')
end

function load_parameters(flattened_parameters, file_name)
   -- flatten the parameters for loading from storage.  While this has already been done in the trainer, the trainer probably shouldn't be responsible for saving and loading the parameters
   --local flattened_parameters = model:getParameters() 

   print('loading flattened parameters from ' .. file_name)
   local mf = torch.DiskFile(file_name,'r'):binary()
   local saved_parameters = mf:readObject()
   print('current parameters have size ' .. flattened_parameters:nElement() .. '; loaded parameters have size ' .. saved_parameters:nElement())
   flattened_parameters:copy(saved_parameters)
   mf:close()

   --flattened_parameters = nil
   saved_parameters = nil
   collectgarbage()
end
