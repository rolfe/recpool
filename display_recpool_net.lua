require 'image'
require 'gnuplot'

--local part_thresh, cat_thresh = 0.5, 0.7 -- FOR PAPER
--local part_thresh, cat_thresh = 0.45, 0.5 -- ENTROPY EXPERIMENTS
local part_thresh, cat_thresh = 0.25, 0.3 -- CIFAR ENTROPY EXPERIMENTS
--local part_thresh, cat_thresh = 0.2, 0.275 -- CIFAR ENTROPY EXPERIMENTS 8x8


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

-- symmetric == false -> the filters have already been scaled, and should be plotted as-is (or something like that; check!)
-- symmetric == true or nil -> scale colormap between -max and max, with gray = 0
function save_filter(current_filter, filter_name, log_directory, num_display_columns, symmetric)
   if symmetric == nil then symmetric = true end -- default value of symmetric is true
   num_display_columns = num_display_columns or 10
   local current_filter_side_length 
   if true and (current_filter:size(1) % 3 == 0) then -- make sure that CIFAR input filters align the R, G, and B channels coherently
      current_filter_side_length = math.sqrt(current_filter:size(1)/3) 
      --current_filter = current_filter:reshape(current_filter:size(2),3,32,32) -- reshape makes a copy of the entire filter, which seems unnecessarily inefficient
      -- after unfolding, the original dimension iterates across groups; the last dimension iterates within groups
      current_filter = current_filter:unfold(1,current_filter_side_length,current_filter_side_length):unfold(1,current_filter_side_length,current_filter_side_length):transpose(1,2) -- may still need to transpose the last two dimensions!!!
      --current_filter_side_length = math.sqrt(current_filter:size(1))
   else
      current_filter_side_length = math.sqrt(current_filter:size(1))
      current_filter = current_filter:unfold(1,current_filter_side_length, current_filter_side_length):transpose(1,2)
   end
   if symmetric then current_filter = current_filter:clone():mul(-1) end -- flip the current filter from white to black; make a copy so we don't risk corrupting the original data
   local current_image = image.toDisplayTensor{input=current_filter,padding=1,nrow=num_display_columns,symmetric=symmetric}
   
   -- ideally, the pdf viewer should refresh automatically.  This 
   image.savePNG(paths.concat(log_directory, filter_name .. '.png'), current_image)
end

-- dataset is nExamples x input_dim
-- hidden_activation is nExamples x hidden_dim
-- construct a dictionary matrix that optimally reconstructs the data_set from the hidden_activation
-- odm stands for optimal dictionary matrix
-- output_matrix is of size hidden_dim x input_dim ; it is already restricted to the correct hidden unit
function construct_optimal_dictionary(data_set, hidden_activation, output_matrix)
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
	 local selected_filter = conservative_optimal_dictionary_matrix:select(1,num_active_units)
	 --output_matrix:select(1,(i-1)*odm_stride + odm_offset):copy(selected_filter:div(selected_filter:norm())) -- this ignores the extra rows --:narrow(1,1,hidden_activation:size(2)))
	 -- select the desired row of the output matrix
	 output_matrix:select(1,i):copy(selected_filter):div(selected_filter:norm()) -- this ignores the extra rows --:narrow(1,1,hidden_activation:size(2)))
	 optimal_dictionary_matrix_slice:select(1,i):copy(conservative_optimal_dictionary_matrix:select(1,num_active_units))
      end
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
   local class_tensor = torch.Tensor(nExamples) -- the class should always be a positive integer
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
   
   -- this is the interface to the outside world
   function receptive_field_builder:accumulate_shrink_weighted_inputs(new_input, base_shrink, shrink_copies, new_target)
      local batch_size = new_input:size(1)
      if data_set_index >= nExamples then
	 error('accumulated ' .. data_set_index .. ' elements in the receptive field builder, but only expected ' .. nExamples)
      end

      data_set_tensor:narrow(1,data_set_index,batch_size):copy(new_input) -- copy the input values from the dataset
      class_tensor:narrow(1,data_set_index,batch_size):copy(new_target)

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
   
   function receptive_field_builder:plot_receptive_fields(opt, encoding_filter, decoding_filter)
      --shrink_val_tensor:select(2,nExamples+1):zero()
      --data_set_tensor:select(1,nExamples+1):fill(1)

      -- show evolution of optimal dictionaries in a single figure -- hidden_layer_size, total_num_shrink_copies, input_size
      --local optimal_dictionary_matrix = torch.Tensor(shrink_val_tensor:size(3) * shrink_val_tensor:size(1), data_set_tensor:size(2)):zero()
      local optimal_dictionary_matrix = torch.Tensor(hidden_layer_size, total_num_shrink_copies, input_size):zero()
      for i = 1,#accumulated_inputs do -- iterate over shrink copies/hidden layers
	 local receptive_field_output = self:extract_receptive_fields(i)
	 save_filter(receptive_field_output, 'shrink receptive field ' .. i, opt.log_directory)
	 --construct_optimal_dictionary(data_set_tensor, shrink_val_tensor:select(1,i), optimal_dictionary_matrix, i, shrink_val_tensor:size(1), 'shrink dictionary ' .. i, opt.log_directory)
	 construct_optimal_dictionary(data_set_tensor, shrink_val_tensor:select(1,i), optimal_dictionary_matrix:select(2,i))
      end

      local max_val = math.max(math.abs(optimal_dictionary_matrix:min()), math.abs(optimal_dictionary_matrix:max()))
      optimal_dictionary_matrix:mul(-1)
      optimal_dictionary_matrix:add(max_val):div(2*max_val)
      
      print('total min and max are ' .. optimal_dictionary_matrix:min() .. ', ' .. optimal_dictionary_matrix:max())


      local function categoricalness_enc_dec_alignment(i)
	 local enc = encoding_filter:select(1,i)
	 local dec = decoding_filter:select(2,i)
	 local angle = math.acos(torch.dot(enc, dec)/(enc:norm() * dec:norm()))
	 if angle > cat_thresh then return 'categorical' 
	 elseif angle < part_thresh then return 'part' 
	 else return 'intermediate' end 
      end

      local part_indices, categorical_indices = {}, {}
      local first_part = 6
      local first_categorical = 16

      -- Use indices 6,7,8, restricted to part units, for the part-unit-only figure
      -- Use indices 16,17,18, restricted to categorical units, for the categorical-unit-only figure
      if encoding_filter and decoding_filter then
	 for current_index = 1,encoding_filter:size(1) do
	    if categoricalness_enc_dec_alignment(current_index) == 'part' then
	       if first_part > 1 then first_part = first_part - 1 
	       elseif #part_indices < 3 then part_indices[#part_indices + 1] = current_index end
	    elseif categoricalness_enc_dec_alignment(current_index) == 'categorical' then
	       if first_categorical > 1 then first_categorical = first_categorical - 1
	       elseif #categorical_indices < 3 then categorical_indices[#categorical_indices + 1] = current_index end
	    end
	    if (#part_indices >= 20) and (#categorical_indices >= 20) then 
	       break 
	    end
	 end
      end

      local max_encoder = math.max(math.abs(encoding_filter:min()), math.abs(encoding_filter:max()))
      local max_decoder = math.max(math.abs(decoding_filter:min()), math.abs(decoding_filter:max()))

      -- construct the full image from the composite pieces
      local function make_figure(num_rows, row_mapper, file_name)
	 local filter_side_length = math.sqrt(input_size)
	 local padding = 1
	 local extra_padding = 8
	 local total_extra_padding = 2*extra_padding
	 local xmaps = total_num_shrink_copies + 2
	 local ymaps = num_rows
	 local height = filter_side_length + padding
	 local width = filter_side_length + padding
	 local white_value = 1 --(args.symmetric and math.max(math.abs(args.input:min()),math.abs(args.input:max()))) or args.input:max()
	 local image_out = torch.Tensor(height*ymaps, width*xmaps + total_extra_padding):fill(white_value)

	 for y = 1,ymaps do
	    for x = 1,xmaps do
	       local current_extra_padding = (math.min(x - 1, 1) + math.max(x - (total_num_shrink_copies + 1), 0)) * extra_padding
	       local selected_image_region = image_out:narrow(1,(y-1)*height+1+padding/2,filter_side_length):narrow(2,(x-1)*width+1+padding/2 + current_extra_padding,filter_side_length)
	       local selected_transfer_image
	       if x == 1 then -- flip the color maps and normalize the encoder and decoder filters; we can multiply by -1 before normalizing because the normalization is symmetric around 0
		  selected_transfer_image = encoding_filter:select(1,row_mapper(y)):clone():mul(-1):add(max_encoder):div(2*max_encoder)
	       elseif x == total_num_shrink_copies + 2 then
		  selected_transfer_image = decoding_filter:select(2,row_mapper(y)):clone():mul(-1):add(max_decoder):div(2*max_decoder)
	       else 
		  selected_transfer_image = optimal_dictionary_matrix[{row_mapper(y), x-1, {}}]
	       end
		  
	       selected_image_region:copy(selected_transfer_image:unfold(1, filter_side_length, filter_side_length))
	    end
	 end
	 image.savePNG(paths.concat(opt.log_directory, file_name), image_out)
      end

      make_figure(#part_indices, function(x) return part_indices[x] end, 'shrink_dictionary_part.png')
      make_figure(#categorical_indices, function(x) return categorical_indices[x] end, 'shrink_dictionary_categorical.png')
      make_figure(hidden_layer_size, function(x) return x end, 'shrink_dictionary.png')
   end

   function receptive_field_builder:plot_reconstruction_connections(opt)
      local input_dim = model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:size(1)
      if (input_dim == 2) or (input_dim == 3)  then -- plot reconstructions only as 2d points
	 plot_reconstruction_connections_2d(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight, 
					    ((opt.plot_temporal_reconstructions and shrink_val_tensor) or shrink_val_tensor:select(1,shrink_val_tensor:size(1))), 
					    data_set_tensor, class_tensor, opt, 20)
      else -- plot filters, as well as reconstructions, as square bitmaps
	 plot_reconstruction_connections(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight, shrink_val_tensor:select(1,shrink_val_tensor:size(1)), data_set_tensor, opt, 20)
      end
   end

   function receptive_field_builder:plot_part_unit_sharing(opt)
      plot_part_sharing_histogram(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight, 
				  model.layers[1].module_list.decoding_feature_extraction_dictionary.weight, 
				  shrink_val_tensor:select(1,shrink_val_tensor:size(1)), class_tensor, opt)
   end

   function receptive_field_builder:plot_other_figures(opt)
      plot_part_sharing_histogram(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight, 
				  model.layers[1].module_list.decoding_feature_extraction_dictionary.weight, 
				  shrink_val_tensor:select(1,shrink_val_tensor:size(1)), class_tensor, opt)

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

      local norm_vec = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      local enc_norm_vec = torch.Tensor(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight:size(1))
      local dec_norm_vec = torch.Tensor(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:size(2))
      local classification_norm_vec = torch.Tensor(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:size(2))
      local prod_norm_vec = torch.Tensor(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:size(2))
      local ista_ideal_prod = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      local ista_ideal_norm_vec = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      
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

      local ista_ideal_matrix = torch.mm(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:t(), model.layers[1].module_list.decoding_feature_extraction_dictionary.weight):mul(-1) --:add(-1, torch.diag(torch.ones(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:size(2)))) -- NOT NECESSARY since the identity matrix is already added in explicitly

      --torch.diag(torch.mm(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight, model.layers[1].module_list.decoding_feature_extraction_dictionary.weight)), 
	 
      for i = 1,model.layers[1].module_list.explaining_away.weight:size(1) do
	 norm_vec[i] = model.layers[1].module_list.explaining_away.weight:select(1,i):norm()
	 enc_norm_vec[i] = model.layers[1].module_list.encoding_feature_extraction_dictionary.weight:select(1,i):norm()
	 dec_norm_vec[i] = model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:select(2,i):norm()
	 classification_norm_vec[i] = model.module_list.classification_dictionary.weight:select(2,i):norm()
	 prod_norm_vec[i] = torch.dot(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight:select(1,i), 
				      model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:select(2,i))

	 ista_ideal_prod[i] = torch.dot(model.layers[1].module_list.explaining_away.weight:select(1,i),
				  ista_ideal_matrix:select(1,i))
	 ista_ideal_norm_vec[i] = ista_ideal_matrix:select(1,i):norm()
      end
      --print(norm_vec:unfold(1,10,10))
      local angle_between_encoder_and_decoder = torch.cdiv(prod_norm_vec, torch.cmul(enc_norm_vec, dec_norm_vec)):acos()
      local angle_between_recurrent_input_and_ISTA_ideal = torch.cdiv(ista_ideal_prod, torch.cmul(norm_vec, ista_ideal_norm_vec)):acos()

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
	    -- plot the ratio between the actual weight and the ISTA-ideal weight, but only for the weights larger than the median, since the ratio is unstable for small weights.  Bound the ratio between -0.5 and 2, so outliers don't disrupt the scale of the plot.  Multiply by -1 since the ista ideal is -1 * dot_preocut_between_decoders
	    deviation_of_recurrent_weight_from_ISTA[exp_away_linearized_index] = math.max(-0.5, math.min(2, -1 * (((math.abs(model.layers[1].module_list.explaining_away.weight[{i,j}]) > median_abs_weight) and 1) or 0) * model.layers[1].module_list.explaining_away.weight[{i,j}] / dot_product_between_decoders))
	    deviation_of_recurrent_weight_from_ISTA_just_parts_inputs[exp_away_linearized_index] = math.max(-0.5, math.min(2, -1 * (((math.abs(model.layers[1].module_list.explaining_away.weight[{i,j}]) > median_abs_weight) and 1) or 0) * (((angle_between_encoder_and_decoder[j] < 0.55) and 1) or 0) * model.layers[1].module_list.explaining_away.weight[{i,j}] / dot_product_between_decoders))
	    categoricalness_of_recurrent_weight_recipient[exp_away_linearized_index] = angle_between_encoder_and_decoder[i]

	    local cwm_bin = math.max(1, math.floor(cwm_pc_num_bins * (dot_product_between_decoders + 1) / 2))
	    if (angle_between_encoder_and_decoder[i] > cat_thresh) and (angle_between_encoder_and_decoder[j] < part_thresh) then
	       connection_weight_means_part_to_categorical[cwm_bin] = connection_weight_means_part_to_categorical[cwm_bin] + model.layers[1].module_list.explaining_away.weight[{i,j}]
	       connection_weight_counts_part_to_categorical[cwm_bin] = connection_weight_counts_part_to_categorical[cwm_bin] + 1
	    end

	    dot_product_between_decoders_per_connection_from_part_to_part[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] < part_thresh) and 1) or 0) * (((angle_between_encoder_and_decoder[j] < part_thresh) and 1) or 0) * dot_product_between_decoders
	    weight_of_connections_from_part_to_part[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] < part_thresh) and 1) or 0) * (((angle_between_encoder_and_decoder[j] < part_thresh) and 1) or 0) * model.layers[1].module_list.explaining_away.weight[{i,j}]

	    dot_product_between_decoders_per_connection_from_categorical_to_part[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] < part_thresh) and 1) or 0) * (((angle_between_encoder_and_decoder[j] > cat_thresh) and 1) or 0) * dot_product_between_decoders
	    weight_of_connections_from_categorical_to_part[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] < part_thresh) and 1) or 0) * (((angle_between_encoder_and_decoder[j] > cat_thresh) and 1) or 0) * model.layers[1].module_list.explaining_away.weight[{i,j}]

	    dot_product_between_decoders_per_connection_from_part_to_categorical[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] > cat_thresh) and 1) or 0) * (((angle_between_encoder_and_decoder[j] < part_thresh) and 1) or 0) * dot_product_between_decoders
	    weight_of_connections_from_part_to_categorical[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] > cat_thresh) and 1) or 0) * (((angle_between_encoder_and_decoder[j] < part_thresh) and 1) or 0) * model.layers[1].module_list.explaining_away.weight[{i,j}]

	    dot_product_between_decoders_per_connection_from_categorical_to_categorical[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] > cat_thresh) and 1) or 0) * (((angle_between_encoder_and_decoder[j] > cat_thresh) and 1) or 0) * dot_product_between_decoders
	    weight_of_connections_from_categorical_to_categorical[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] > cat_thresh) and 1) or 0) * (((angle_between_encoder_and_decoder[j] > cat_thresh) and 1) or 0) * model.layers[1].module_list.explaining_away.weight[{i,j}]
	    angle_between_classifiers_per_connection_from_categorical_to_categorical[exp_away_linearized_index] = 
	       (((angle_between_encoder_and_decoder[i] > cat_thresh) and 1) or 0) * (((angle_between_encoder_and_decoder[j] > cat_thresh) and 1) or 0) * angle_between_classifiers

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

	       if angle_between_encoder_and_decoder[j] < part_thresh then
		  part_weighted_sum_angle = part_weighted_sum_angle + 
		     model.layers[1].module_list.explaining_away.weight[{i,j}] * (math.pi/2 - math.acos(dot_product_between_decoders / (dec_norm_vec[i] * dec_norm_vec[j])))
		  part_norm = part_norm + math.abs(model.layers[1].module_list.explaining_away.weight[{i,j}])
	       elseif angle_between_encoder_and_decoder[j] > cat_thresh then
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


      gnuplot.pngfigure(opt.log_directory .. '/scat_recurrent_weight_match_to_ista_ideal.png') 
      gnuplot.plot(angle_between_encoder_and_decoder, angle_between_recurrent_input_and_ISTA_ideal)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('angle between recurrent input and ista ideal')
      gnuplot.plotflush()

      gnuplot.figure() -- percentage of inputs for which the unit is activated at some point, but the first activation occurs after the first iteration; versus the magnitude of the recurrent connections; categorical units turn on later, since they have poorly structured encoder inputs but strong connections to part-units.
      gnuplot.plot(angle_between_encoder_and_decoder, percentage_late_activation)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('prob of late activation')

      gnuplot.pngfigure(opt.log_directory .. '/scat_prob_of_second_iter_activation.png') -- percentage of inputs for which the unit is activated at some point, but the first activation occurs at the second iteration; versus the magnitude of the recurrent connections; categorical units turn on later, since they have poorly structured encoder inputs but strong connections to part-units.
      gnuplot.plot(angle_between_encoder_and_decoder, percentage_second_iter_activation)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('prob of second iter activation')
      gnuplot.plotflush()

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

      gnuplot.pngfigure(opt.log_directory .. '/scat_decoder_mean.png') -- mean decoder column versus categoricalness
      gnuplot.plot(angle_between_encoder_and_decoder,
		   model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:mean(1):select(1,1)) -- argument to mean is the dimension collapsed
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('decoder mean')
      gnuplot.plotflush()

      gnuplot.figure() -- mean decoder column versus categoricalness
      gnuplot.plot(angle_between_encoder_and_decoder,
		   model.layers[1].module_list.encoding_feature_extraction_dictionary.weight:mean(2):select(2,1)) -- argument to mean is the dimension collapsed
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('encoder mean')

      
      gnuplot.pngfigure(opt.log_directory .. '/scat_recurrent_connection_magnitude.png') -- cos(angle) between encoder and decoder versus magnitude of recurrent input; categorical units have unaligned encoder/decoder pairs and larger recurrent connections
      gnuplot.plot(angle_between_encoder_and_decoder, norm_vec)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('recurrent connection magnitude')
      gnuplot.plotflush()

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

      gnuplot.pngfigure(opt.log_directory .. '/scat_classification_dictionary_connection_magnitude.png') 
      gnuplot.plot(angle_between_encoder_and_decoder, norm_classification_connection)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('classification dictionary connection magnitude')
      gnuplot.plotflush()
      

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

      gnuplot.pngfigure(opt.log_directory .. '/scat_weighted_average_categoricalness.png') -- cos(angle) between encoder and decoder versus magnitude of recurrent input; categorical units have unaligned encoder/decoder pairs and larger recurrent connections
      gnuplot.plot(angle_between_encoder_and_decoder, average_recurrent_total_connection_categoricalness)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('weighted average afferent enc-dec angle')
      gnuplot.plotflush()

      print(angle_between_encoder_and_decoder:unfold(1,10,10))
      print(average_value_when_activated:unfold(1,10,10))
      print(average_value_when_activated:size())

      gnuplot.pngfigure(opt.log_directory .. '/scat_average_final_value_when_activation.png') 
      gnuplot.plot(angle_between_encoder_and_decoder, average_value_when_activated)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('average final value of unit when activated')
      gnuplot.plotflush()

      gnuplot.pngfigure(opt.log_directory .. '/scat_class_dict_mag_versus_final_activation.png')
      gnuplot.plot(norm_classification_connection, average_value_when_activated)
      gnuplot.xlabel('classification dictionary column magnitude')
      gnuplot.ylabel('average final value of unit when activated')
      gnuplot.plotflush()

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

      gnuplot.pngfigure(opt.log_directory .. '/scat_ista_weights_part_to_part.png')
      --gnuplot.figure() 
      gnuplot.plot(dot_product_between_decoders_per_connection_from_part_to_part, weight_of_connections_from_part_to_part, '.')
      gnuplot.xlabel('dot product between decoders from part to part')
      gnuplot.ylabel('connection weight')
      gnuplot.plotflush()

      gnuplot.pngfigure(opt.log_directory .. '/scat_ista_weights_categorical_to_part.png')
      --gnuplot.figure() 
      gnuplot.plot(dot_product_between_decoders_per_connection_from_categorical_to_part, weight_of_connections_from_categorical_to_part, '.')
      gnuplot.xlabel('dot product between decoders from categorical to part')
      gnuplot.ylabel('connection weight')
      gnuplot.plotflush()

      gnuplot.figure() 
      gnuplot.plot(dot_product_between_decoders_per_connection_from_part_to_categorical, weight_of_connections_from_part_to_categorical, '.')
      gnuplot.xlabel('dot product between decoders from part to categorical')
      gnuplot.ylabel('connection weight')

      gnuplot.figure() 
      gnuplot.plot(dot_product_between_decoders_per_connection_from_categorical_to_categorical, weight_of_connections_from_categorical_to_categorical, '.')
      gnuplot.xlabel('dot product between decoders from categorical to categorical')
      gnuplot.ylabel('connection weight')

      gnuplot.pngfigure(opt.log_directory .. '/scat_v_diagram.png') 
      gnuplot.plot(connection_weight_dot_products_part_to_categorical, connection_weight_means_part_to_categorical:cdiv(connection_weight_counts_part_to_categorical))
      gnuplot.xlabel('dot product between decoders from part to categorical')
      gnuplot.ylabel('average connection weight')
      gnuplot.plotflush()

      gnuplot.figure() 
      gnuplot.plot(angle_between_classifiers_per_connection_from_categorical_to_categorical, weight_of_connections_from_categorical_to_categorical, '.')
      gnuplot.xlabel('cos(angle) between classifiers from categorical to categorical')
      gnuplot.ylabel('connection weight')


      --plot_reconstruction_connections(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight, shrink_val_tensor:select(1,shrink_val_tensor:size(1)), data_set_tensor, opt, 20)
      plot_hidden_unit_trajectories(shrink_val_tensor:select(2,1), opt, 400)
      plot_hidden_unit_trajectories(shrink_val_tensor:select(2,1), opt, 400, 1, model.layers[1].module_list.encoding_feature_extraction_dictionary.weight, 
				    model.layers[1].module_list.decoding_feature_extraction_dictionary.weight) -- shrink_val_tensor = torch.Tensor(total_num_shrink_copies, nExamples, hidden_layer_size)
      plot_hidden_unit_trajectories(shrink_val_tensor:select(2,1), opt, 400, -1, model.layers[1].module_list.encoding_feature_extraction_dictionary.weight, 
				    model.layers[1].module_list.decoding_feature_extraction_dictionary.weight) -- shrink_val_tensor = torch.Tensor(total_num_shrink_copies, nExamples, hidden_layer_size)


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



local function plot_bar(args) -- {bar_length, max_bar_length, image_edge_length, max_decoding, current_column}
   local bar_sign = args.bar_length/math.abs(args.bar_length)
   if args.bar_length > args.max_bar_length then 
      print('bar length > max bar length') 
      args.bar_length = args.max_bar_length
   end
   for i=1,math.ceil((args.image_edge_length - 2) * math.abs(args.bar_length)/args.max_bar_length) do
      args.current_column[args.image_edge_length + 1 + i] = args.max_decoding * bar_sign
   end
end


-- plot the decoding dictionaries of the top n largest magnitude connections to each unit, scaled by the connection weight.  This gives a sense of how each unit's activation is computed based from the other units.  If restrictions is a table, it is organized like {(rows of fig contain connections from common: source, destination), (restrict source to: any, part, categorical), (restrict dest to: any, part, categorical) (separate by class)}
function plot_explaining_away_connections(encoding_filter, decoding_filter, explaining_away_filter_orig, opt, restrictions, classification_filter, start_display_row, num_display_rows)
   local num_sorted_connections = 20 -- number of connections to show for each unit
   local explaining_away_mag_filter = explaining_away_filter_orig:clone() -- this is used to select which connections to display, and is altered below depending upon the type of connections desired
   local explaining_away_filter = explaining_away_filter_orig:clone() -- make a copy so as to avoid corrupting the original filter
   local file_name, col_type, row_type
   local separate_by_class = false -- reorder each row of the display so that connections of a given class are grouped together.  This makes it apparent if they tend to have the same sign
   local restrict_source_and_dest = false -- don't plot all units and connections; rather based upon the value of restrictions, plot only sources and destinations of particular types
   local dont_restrict_max_on_col = false -- when computing the maximal connection value for scaling the bars on top of the decoders, should the col-restriction be enforced?
   local dont_restrict_max_on_row = false
      
   if restrictions == 'restrict to positive' then
      explaining_away_mag_filter:maxZero()
      file_name = 'positive sorted recurrent connections'
   elseif type(restrictions) == 'table' then
      local connection_direction_name
      if false and (restrictions[3] == 'categorical') then -- the projections to the categorical units are not ISTA-like, and so the diagonal doesn't have any special meaning
	 print('adding in diagonal')
	 explaining_away_mag_filter:add(torch.diag(torch.ones(explaining_away_filter:size(2))))
	 explaining_away_filter:add(torch.diag(torch.ones(explaining_away_filter:size(2))))
      end
      
      -- 'source' plots connections from a common source (cols of the explaining_away matrix); 'destination' plots connections to a common destination (rows of the explaining_away matrix)
      -- restrictions[2] controls the source of the connections
      -- restrictions[3] controls the dest of the connections: read as 'from [2] to [3]'
      if restrictions[1] == 'source' then -- to enable the same code to organize the plot by source or by destination, transpose everything if we want to plot by source; if not transposed, the rows of the explaining_away matrix correspond to a single destination
	 connection_direction_name = 'outgoing'
	 col_type = restrictions[3] -- restrictions can be 'part', 'categorical', or 'all', and are implemented with the categoricalness filters and test_cat_restriction() below
	 row_type = restrictions[2] 
	 explaining_away_mag_filter = explaining_away_mag_filter:t()
	 explaining_away_filter = explaining_away_filter:t() -- transpose this to match the sorting filter; all rows now contain the outgoing rather than incoming connections
	 dont_restrict_max_on_row = true -- normalize magnitude bars based upon all incoming connections; i.e., only restrict based on cols
      elseif restrictions[1] == 'destination' then
	 connection_direction_name = 'incoming'
	 col_type = restrictions[2] 
	 row_type = restrictions[3] 
	 dont_restrict_max_on_col = true -- normalize magnitude bars based upon all incoming connections; i.e., only restrict based on rows
      else 
	 error('cannot sort connections by ' .. restrictions[1] '; choose source or destination')
      end
      explaining_away_mag_filter = explaining_away_mag_filter:abs()
      restrict_source_and_dest = true
      separate_by_class = restrictions[4]
      file_name = 'sorted_' .. connection_direction_name .. '_connections_from_' .. (restrictions[2] or 'any') .. '_to_' .. (restrictions[3] or 'any')
   else
      explaining_away_mag_filter:abs()
      file_name = 'sorted recurrent connections'
   end


   -- two options for defining categoricalness
   local function categoricalness_classification_filter(i)
      if classification_filter:select(2,i):norm() > 0.5 then return 'categorical' 
      elseif classification_filter:select(2,i):norm() < 0.15 then return 'part' 
      else return 'intermediate' end 
   end
   
   local function categoricalness_enc_dec_alignment(i)
      local enc = encoding_filter:select(1,i)
      local dec = decoding_filter:select(2,i)
      local angle = math.acos(torch.dot(enc, dec)/(enc:norm() * dec:norm()))
      if angle > cat_thresh then return 'categorical'  -- cat_thresh and part_thres are defined at the top of display_recpool_net.lua
      elseif angle < part_thresh then return 'part' 
      else return 'intermediate' end 
   end

   -- choose between different definitions of categoricalness
   --local categoricalness = categoricalness_classification_filter 
   local categoricalness = categoricalness_enc_dec_alignment
      
   local function test_cat_restriction(restriction, index)
      return (restriction == 'any') or (restriction == categoricalness(index))
   end


   local max_exp_away = math.max(math.abs(explaining_away_filter:max()), math.abs(explaining_away_filter:min()))
   local max_decoding = math.max(math.abs(decoding_filter:max()), math.abs(decoding_filter:min()))
   local image_edge_length, image_edge_center = math.floor(math.sqrt(decoding_filter:size(1))), math.floor(math.sqrt(decoding_filter:size(1))/2)

   -- sort each row (i.e., along the columns) of the explaining_away_mag_filter, and extract the permutation induced
   local explaining_away_mag_filter_sorted, desired_indices = explaining_away_mag_filter:sort(2, true) 
   
   -- if the considered connections are restricted, set max_exp_away to the maximum amongst connections from part/cat/all to part/cat/all units
   if restrict_source_and_dest then 
      max_exp_away = nil
      for i = 1,explaining_away_filter:size(1) do
	 for j = 1,explaining_away_filter:size(2) do
	    if (dont_restrict_max_on_row or test_cat_restriction(row_type, i)) and 
	       (dont_restrict_max_on_col or test_cat_restriction(col_type, j)) then
	       max_exp_away = max_exp_away or math.abs(explaining_away_filter[{i,j}])
	       max_exp_away = math.max(max_exp_away, math.abs(explaining_away_filter[{i,j}]))
	    end
	 end
      end
   end

   -- for each possible destination, if it is of the right type, consider all possible sources and add those of the right type to the figure
   -- to organize based upon sources rather than destinations, transpose the explaining-away matrix above
   -- to sort the connections first based on digit ID and then based on connection strength, create an array for each digit ID and use this intermediate storage to organize the data for the figure

   -- determine which rows fit our criterion, so we don't consider irrelevant ones
   -- it is necessary to do this first, since we may or may not specify a subset of these rows that we want.  If we don't specify the desired subset, we need to determine the number of rows to allocate the appropriate amount of memory for sorted_recurrent_connection_filter
   local desired_display_rows = torch.Tensor(explaining_away_filter:size(1)):zero()
   local current_test_row = 0
   for i = 1,explaining_away_filter:size(1) do
      if not(restrict_source_and_dest) or test_cat_restriction(row_type, i) then
	 current_test_row = current_test_row + 1
	 desired_display_rows[current_test_row] = i
      end
   end
   desired_display_rows = desired_display_rows:narrow(1,1,current_test_row)
   -- restrict attention to only the desired subset of acceptable rows
   if start_display_row then
      print('desired display rows restricted from ', desired_display_rows:narrow(1,1,math.min(20, desired_display_rows:size(1))))
      if start_display_row > desired_display_rows:size(1) then
	 print('start_display_row = ' .. start_display_row .. ' < desired_display_rows:size(1) = ' .. desired_display_rows:size(1) .. ' ; resetting to 1')
	 start_display_row = 1
      end
      local available_display_rows = desired_display_rows:size(1) - start_display_row + 1
      desired_display_rows = desired_display_rows:narrow(1,start_display_row, math.min(num_display_rows, available_display_rows))
      if num_display_rows > available_display_rows then
	 print('Only have ' .. available_display_rows .. ' available display rows, rather than the desired ' .. num_display_rows .. ' ; resetting')
	 num_display_rows = available_display_rows
      end
      print('to ', desired_display_rows)
   else
      num_display_rows = current_test_row
   end

   -- this tensor holds the data for the output image.  The final image is made by unfolding this tensor.  The data for each mini-image is located in a single column.
   local sorted_recurrent_connection_filter = torch.Tensor(num_display_rows, num_sorted_connections + 1, decoding_filter:size(1)):zero() 
   --local output_filter_index = 1 -- position in the tensor that will be used to generate the output image
   local i = nil -- make sure we don't use this
   for out_row_index = 1,num_display_rows do
      in_row_index = desired_display_rows[out_row_index]
      if restrict_source_and_dest and not(test_cat_restriction(row_type, in_row_index)) then error('prescreened row was not acceptable!') end

      -- the first column is the decoder of the selected unit
      local current_column = sorted_recurrent_connection_filter[{out_row_index, 1, {}}] -- :select(2,output_filter_index)
      current_column:copy(decoding_filter:select(2,in_row_index)) --:mul(0.1)
      --output_filter_index = output_filter_index + 1

      -- find the num_sorted_connections largest connections of the desired type, separate them by the digit ID of the source, and then refill desired_indices so it will be grouped by digit ID
      if separate_by_class then 
	 local sorted_index_copy = desired_indices:select(1,in_row_index) --:clone()
	 local indices_by_class = {}
	 for k = 1,classification_filter:size(1) do
	    indices_by_class[k] = {}
	 end
	 local col_index = 0 -- search position within the row of sorted_index_copy, which is taken from desired_indices
	 for j = 1,num_sorted_connections do
	    col_index = col_index + 1
	    while not(test_cat_restriction(col_type, sorted_index_copy[col_index])) do -- consider only columns of the right type, in sorted order 
	       col_index = col_index + 1 
	       if col_index > desired_indices:size(2) then break end
	    end 
	    if col_index > desired_indices:size(2) then break end
	    
	    -- determine the digit ID associated with this unit, and place it accordingly into indices_by_class
	    local _, index_class = torch.max(classification_filter:select(2,sorted_index_copy[col_index]), 1) 
	    index_class = index_class[1]
	    indices_by_class[index_class][#(indices_by_class[index_class]) + 1] = sorted_index_copy[col_index]
	 end
	 
	 -- refill desired_indices based upon indices_by_class, which is sorted by magnitude within each digit-ID bin
	 col_index = 0
	 for j = 1,#indices_by_class do
	    for k = 1,#(indices_by_class[j]) do
	       col_index = col_index + 1
	       desired_indices[{in_row_index,col_index}] = indices_by_class[j][k]
	    end
	 end
	 if col_index < num_sorted_connections then
	    print('WARNING: col_index was less than num_sorted_connections')
	    --for k = col_index+1,num_sorted_connections do
	    --   desired_indices[{in_row_index,k}] = 1
	    --end
	 elseif col_index > num_sorted_connections then
	    error('col_index was greater than num_sorted_connections')
	 end
      end
      
      -- the remaining columns are the decoders of the units connected to it, sorted and scaled by connection strength
      local col_index = 0
      for j = 1,num_sorted_connections do
	 if restrict_source_and_dest then -- if we're plotting connections from part units to categorical units, skip all connections to part units
	    col_index = col_index + 1
	    if col_index > desired_indices:size(2) then break end
	    while not(test_cat_restriction(col_type, desired_indices[{in_row_index,col_index}])) do 
	       col_index = col_index + 1 
	       if col_index > desired_indices:size(2) then break end
	    end 
	 else col_index = j end
	 
	 if separate_by_class and col_index > num_sorted_connections then error('requested unprepared index') end -- only num_sorted_connections columns were properly filled above
	 
	 current_column = sorted_recurrent_connection_filter[{out_row_index, j+1, {}}] --:select(2,output_filter_index)
	 if col_index <= desired_indices:size(2) then -- don't copy anything if we're run out acceptable units
	    current_column:copy(decoding_filter:select(2,desired_indices[{in_row_index,col_index}]))
	    --current_column:mul(explaining_away_filter[{i,desired_indices[{i,col_index}]}]) -- + (((i == desired_indices[{i,j}]) and 1) or 0))
	 
	    -- draw a bar indicating the size (even if negative) of the recurrent connection; the step size is the sign of the connection
	    plot_bar{bar_length = explaining_away_filter[{in_row_index,desired_indices[{in_row_index,col_index}]}],
		     max_bar_length = max_exp_away, image_edge_length = image_edge_length, max_decoding = max_decoding, current_column = current_column}
	 end
	 --[[
	 local rec_connection_size = explaining_away_filter[{in_row_index,desired_indices[{in_row_index,col_index}]}]
	 local rec_connection_sign = rec_connection_size/math.abs(rec_connection_size)
	 for i=1,math.ceil((image_edge_length - 2) * math.abs(rec_connection_size)/max_exp_away) do
	    current_column[image_edge_length + 1 + i] = max_decoding * rec_connection_sign
	 end
	 --]]
	 --output_filter_index = output_filter_index + 1
      end
   end -- for all units
   print(sorted_recurrent_connection_filter:size())
   --[[
   print('narrowing to ' .. (output_filter_index - 1))
   local narrowed_sorted_recurrent_connection_filter = sorted_recurrent_connection_filter:narrow(2,1,output_filter_index - 1)
   if start_display_row and num_display_rows then
      print('narrowing further: ' .. start_display_row .. ', ' .. num_display_rows)
      narrowed_sorted_recurrent_connection_filter = narrowed_sorted_recurrent_connection_filter:narrow(2, 1 + (start_display_row - 1)*(num_sorted_connections + 1), 
												       num_display_rows*(num_sorted_connections + 1))
   end
   save_filter(narrowed_sorted_recurrent_connection_filter, file_name, opt.log_directory, num_sorted_connections + 1) -- this depends upon save_filter using a row length of num_sorted_connections!!!
   --]]

   -- the image must be bounded between 0 and 1 to be passed into image.savePNG()
   sorted_recurrent_connection_filter:mul(-1)
   sorted_recurrent_connection_filter:add(max_decoding):div(2*max_decoding)
   
   print('total min and max are ' .. sorted_recurrent_connection_filter:min() .. ', ' .. sorted_recurrent_connection_filter:max())
   
   -- construct the full image from the composite pieces
   local filter_side_length = math.sqrt(decoding_filter:size(1))
   --local current_filter = current_filter:unfold(1,current_filter_side_length, current_filter_side_length)
   
   local padding = 1
   local extra_padding = 8
   local total_extra_padding = extra_padding
   local xmaps = num_sorted_connections + 1
   local ymaps = num_display_rows
   local height = filter_side_length + padding
   local width = filter_side_length + padding
   local white_value = 1 --(args.symmetric and math.max(math.abs(args.input:min()),math.abs(args.input:max()))) or args.input:max()
   local image_out = torch.Tensor(height*ymaps, width*xmaps + total_extra_padding):fill(white_value)
   for y = 1,ymaps do
      for x = 1,xmaps do
	 local current_extra_padding = math.min(x - 1, 1) * extra_padding
	 local selected_image_region = image_out:narrow(1,(y-1)*height+1+padding/2,filter_side_length):narrow(2,(x-1)*width+1+padding/2 + current_extra_padding,filter_side_length)
	 selected_image_region:copy(sorted_recurrent_connection_filter[{y, x, {}}]:unfold(1, filter_side_length, filter_side_length))
      end
   end
   image.savePNG(paths.concat(opt.log_directory, file_name .. '.png'), image_out)
end



-- plot the decoding dictionaries of the top n largest magnitude connections to each unit, scaled by the connection weight.  This gives a sense of how each unit's activation is computed based from the other units.  If restrictions is a table, it is organized like {(rows of fig contain connections from common: source, destination), (restrict source to: any, part, categorical), (restrict dest to: any, part, categorical)}
function plot_most_categorical_filters(encoding_filter, decoding_filter, classification_filter, data, opt)
   local average_digits = torch.Tensor(data:nClass(), data:dataSize()):zero()
   local num_examples_per_class = torch.Tensor(data:nClass()):zero()

   for i = 1,data:nExample() do
      local current_class = data.labels[i]
      num_examples_per_class:narrow(1,current_class,1):add(1)
      average_digits:select(1,current_class):add(data.data[i]:double())
   end
   
   for i = 1,data:nClass() do
      average_digits:select(1,i):div(num_examples_per_class[i])
   end

   local num_sorted_connections = 4 -- number of connections to show for each class label
   local file_name, col_type, row_type
      

   -- two options for defining categoricalness
   local function categoricalness_classification_filter(i)
      return classification_filter:select(2,i):norm()
   end
   
   local function categoricalness_enc_dec_alignment(i)
      local enc = encoding_filter:select(1,i)
      local dec = decoding_filter:select(2,i)
      local angle = math.acos(torch.dot(enc, dec)/(enc:norm() * dec:norm()))
      return angle
   end

   -- choose between different definitions of categoricalness
   local categoricalness = categoricalness_classification_filter 
   --local categoricalness = categoricalness_enc_dec_alignment


   local max_decoding = math.max(math.abs(decoding_filter:max()), math.abs(decoding_filter:min()))
   local max_average_digits = math.max(math.abs(average_digits:max()), math.abs(average_digits:min()))
   local total_max_output = math.max(max_decoding, max_average_digits)

   local image_edge_length, image_edge_center = math.floor(math.sqrt(decoding_filter:size(1))), math.floor(math.sqrt(decoding_filter:size(1))/2)

   -- sort each row (i.e., along the columns) of the explaining_away_mag_filter, and extract the permutation induced
   local categoricalness_of_all_units = torch.Tensor(decoding_filter:size(2))
   local most_categorical_filters = torch.Tensor(data:nClass(), num_sorted_connections, decoding_filter:size(1))
   local max_enc_dec_alignment = math.abs(categoricalness_enc_dec_alignment(1))
   for i = 1,decoding_filter:size(2) do
      categoricalness_of_all_units[i] = categoricalness(i)
      max_enc_dec_alignment = math.max(max_enc_dec_alignment, math.abs(categoricalness_enc_dec_alignment(i)))
   end
   local categoricalness_sorted, sorted_indices = categoricalness_of_all_units:sort(true)
   for i = 1,data:nClass() do
      current_index = 1
      for j = 1,num_sorted_connections do
	 local _,max_index = torch.max(classification_filter:select(2,sorted_indices[current_index]), 1)
	 max_index = max_index[1]
	 while (max_index ~= i) and (current_index < classification_filter:size(2)) do 
	    current_index = current_index + 1
	    --print(classification_filter:select(2,sorted_indices[current_index]))
	    _,max_index = torch.max(classification_filter:select(2,sorted_indices[current_index]), 1)
	    max_index = max_index[1]
	 end
	 --print('for class ' .. i  .. ' selecting sorted digit ' .. current_index .. ' with categoricalness ' .. categoricalness_sorted[current_index] .. ' / ' .. categoricalness_enc_dec_alignment(sorted_indices[current_index]))
	 local current_column = most_categorical_filters[{i, j, {}}]
	 current_column:copy(decoding_filter:select(2,sorted_indices[current_index]))
	 plot_bar{bar_length = categoricalness_enc_dec_alignment(sorted_indices[current_index]), max_bar_length = max_enc_dec_alignment, image_edge_length = image_edge_length,
		  max_decoding = total_max_output, current_column = current_column}
	 current_index = current_index + 1
      end
   end
	 
   -- the image must be bounded between 0 and 1 to be passed into image.savePNG()
   average_digits:mul(-1)
   average_digits:add(total_max_output):div(2*total_max_output)

   most_categorical_filters:mul(-1)
   most_categorical_filters:add(total_max_output):div(2*total_max_output)
   
   
   -- construct the full image from the composite pieces
   local filter_side_length = math.sqrt(decoding_filter:size(1))
   local classes_per_row = 5 --2
   
   local padding = 1
   local extra_padding_class = 5 --filter_side_length --5 -- padding between different digit classes
   local extra_padding_avg = 2 --4 -- 2 -- padding after the average filter, which appears before the most categorical decoders of each digit class
   local total_extra_padding = extra_padding_class * (classes_per_row - 1) + extra_padding_avg * classes_per_row
   local xmaps = (num_sorted_connections + 1) * classes_per_row
   local ymaps = math.ceil(data:nClass() / classes_per_row)
   local height = filter_side_length + padding
   local width = filter_side_length + padding
   local white_value = 1 --(args.symmetric and math.max(math.abs(args.input:min()),math.abs(args.input:max()))) or args.input:max()
   local image_out = torch.Tensor(height*ymaps, width*xmaps + total_extra_padding):fill(white_value)
   for y = 1,ymaps do
      for x = 1,xmaps do
	 local current_extra_padding = math.floor((x-1) / (num_sorted_connections + 1)) * extra_padding_class + 
	    math.floor((x + num_sorted_connections - 1) / (num_sorted_connections + 1)) * extra_padding_avg
	 local selected_image_region = image_out:narrow(1,(y-1)*height+1+padding/2,filter_side_length):narrow(2,(x-1)*width+1+padding/2 + current_extra_padding,filter_side_length)
	 if x % (num_sorted_connections + 1) == 1 then
	    selected_image_region:copy(average_digits[{(y-1)*classes_per_row + math.ceil(x/(num_sorted_connections + 1)), {}}]:unfold(1, filter_side_length, filter_side_length))
	 else
	    selected_image_region:copy(most_categorical_filters[{(y-1)*classes_per_row + math.ceil(x/(num_sorted_connections + 1)), (x-1) % (num_sorted_connections + 1), {}}]:unfold(1, filter_side_length, filter_side_length))
	 end
      end
   end
   image.savePNG(paths.concat(opt.log_directory, 'most_categorical_filters.png'), image_out)
end


function plot_hidden_unit_trajectories(activation_tensor, opt, num_trajectories, only_plot_parts, encoding_filter, decoding_filter)
   -- shrink_val_tensor = torch.Tensor(total_num_shrink_copies, nExamples, hidden_layer_size)

   local function categoricalness_enc_dec_alignment(i)
      local enc = encoding_filter:select(1,i)
      local dec = decoding_filter:select(2,i)
      local angle = math.acos(torch.dot(enc, dec)/(enc:norm() * dec:norm()))
      if angle > cat_thresh then return 1
      elseif angle < part_thresh then return -1
      else return 0 end
   end


   num_trajectories = math.min(num_trajectories, activation_tensor:size(2))
   local plot_args = {}
   local x = torch.linspace(1,activation_tensor:size(1),activation_tensor:size(1))
   local i = 1
   while (#plot_args < num_trajectories) and (i <= activation_tensor:size(2)) do
      if not(only_plot_parts) or (categoricalness_enc_dec_alignment(i) == only_plot_parts) then
	 plot_args[#plot_args + 1] = {x, activation_tensor:select(2,i), '-'}
      end
      i = i+1
   end
   gnuplot.figure()
   gnuplot.plot(unpack(plot_args))
end

-- plot the decoding dictionaries of the n units with the largest activations in response to each network input, scaled by the activation strength.  This gives a sense of how each input is reconstructed by the hidden activity
function plot_reconstruction_connections(decoding_filter, activation_tensor, input_tensor, opt, num_display_columns)
   -- different network inputs (elements of the dataset) go along the first dimension of activation_tensor (and input_tensor)
   -- different hidden units go along the second dimension of activation_tensor
   local SHOW_PROGRESSIVE_SUBTRACTION = false
   num_display_columns = num_display_columns or 10
   local num_reconstructions_to_plot = 500 --3 --activation_tensor:size(1) -- reduce this to restrict to fewer examples; activation_tensor and input_tensor contain WAY TOO MANY examples to plot them all
   local num_sorted_inputs = num_display_columns - 2 -- the first two columns are the original input and the final reconstruction
   local num_display_rows_per_input = ((SHOW_PROGRESSIVE_SUBTRACTION and 3) or 2)
   local image_edge_length, image_edge_center = math.floor(math.sqrt(decoding_filter:size(1))), math.floor(math.sqrt(decoding_filter:size(1))/2)
   local max_decoding = math.max(math.abs(decoding_filter:min()), math.abs(decoding_filter:max()))
   --local min_input, max_input = input_tensor:min(), input_tensor:max()

   if activation_tensor:size(1) ~= input_tensor:size(1) then
      error('number of data set elements in activation tensor ' .. activation_tensor:size(1) .. ' does not match the number in input tensor ' .. input_tensor:size(1))
   elseif (activation_tensor:size(1) < num_reconstructions_to_plot) or (input_tensor:size(1) < num_reconstructions_to_plot) then
      error('number of data set elements in activation tensor ' .. activation_tensor:size(1) .. ' or the number in input tensor ' .. input_tensor:size(1) .. ' is smaller than the number of requested trajectories ' .. num_reconstructions_to_plot)
   end

   -- sorted_reconstruction_filter holds all of the small images that will be used to build the larger image.  Its dimensions are: different images to reconstruct, each of which corresponds to two rows; the rows for each reconstruction, the first contains parts, the second contains the accretive reconstruction; columns, corresponding to steps in the reconstruction; the filters themselves
   local sorted_reconstruction_filter = torch.Tensor(num_reconstructions_to_plot, num_display_rows_per_input, num_display_columns, decoding_filter:size(1)):zero() -- tensor in which the figure will be constructed
   local progressive_accretion_filter = torch.Tensor(decoding_filter:size(1)) -- temporary storage for constructing the built-up output
   local progressive_subtraction_filter = torch.Tensor(decoding_filter:size(1)) -- temporary storage for constructing the built-down residual
   --local desired_run_iterations_for_plot = {5, 10, 47} -- different digits: 6,4,7
   local desired_run_iterations_for_plot = {88, 471, 379} -- all 3's -- 482
   for counter = 1,num_reconstructions_to_plot do
      i = (desired_run_iterations_for_plot and desired_run_iterations_for_plot[counter]) or counter

      progressive_subtraction_filter:copy(input_tensor:select(1,i))
      progressive_accretion_filter:zero()
      local initial_input_mag = input_tensor:select(1,i):norm()

      local activation_mag_vector = activation_tensor:select(1,i):clone():abs() -- this probably isn't necessary, since units are non-negative by default, but we do it to be safe and forward-compatible
      local activation_mag_vector_sorted, activation_mag_vector_sort_indices = activation_mag_vector:sort(true)
      local desired_indices = activation_mag_vector_sort_indices:narrow(1,1,num_sorted_inputs)
      local current_filter
      
      -- for each element of the reconstruction, in order of magnitude
      for j = 1,desired_indices:size(1) do
	 current_filter = sorted_reconstruction_filter[{counter,1,j,{}}] 
	 current_filter:copy(decoding_filter:select(2,desired_indices[j]))
	 --current_filter:mul(activation_tensor[{i,desired_indices[j]}]) 
	 plot_bar{bar_length = activation_tensor[{i,desired_indices[j]}], max_bar_length = 1.5*initial_input_mag, image_edge_length = image_edge_length, max_decoding = max_decoding, current_column = current_filter}

	 progressive_subtraction_filter:add(-activation_tensor[{i,desired_indices[j]}], decoding_filter:select(2,desired_indices[j]))
	 progressive_accretion_filter:add(activation_tensor[{i,desired_indices[j]}], decoding_filter:select(2,desired_indices[j]))
	 sorted_reconstruction_filter[{counter,2,j,{}}]:copy(progressive_accretion_filter)
	 
	 if SHOW_PROGRESSIVE_SUBTRACTION then
	    local sub_column = sorted_reconstruction_filter[{counter,3,j,{}}] 
	    sub_column:copy(progressive_subtraction_filter)
	    print('bar length ' .. progressive_subtraction_filter:norm(), ' max bar length ', 1.5*initial_input_mag)
	    plot_bar{bar_length = progressive_subtraction_filter:norm(), max_bar_length = initial_input_mag, image_edge_length = image_edge_length, max_decoding = max_decoding, current_column = sub_column}
	 end

      end

      -- plot the final reconstruction
      current_filter = sorted_reconstruction_filter[{counter,2,desired_indices:size(1)+1,{}}]
      torch.mv(current_filter, decoding_filter, activation_tensor:select(1,i))

      -- plot the initial input
      current_filter = sorted_reconstruction_filter[{counter,2,desired_indices:size(1)+2,{}}]
      current_filter:copy(input_tensor:select(1,i))
      progressive_subtraction_filter:copy(input_tensor:select(1,i))
      progressive_accretion_filter:zero()
      local initial_input_mag = current_filter:norm()

      -- white out the filter space above the final reconstruction and initial input
      --sorted_reconstruction_filter[{counter,1,desired_indices:size(1)+1,{}}]:fill(1)
      --sorted_reconstruction_filter[{counter,1,desired_indices:size(1)+2,{}}]:fill(1)
   end


   -- the image must be bounded between 0 and 1 to be passed into image.savePNG()
   local part_filters = sorted_reconstruction_filter:narrow(3,1,num_sorted_inputs):select(2,1) -- AFTER SELECTING, THE ORDER OF THE DIMENSIONS CHANGES!!!  SELECT FIRST!!!
   print('max decoding is ' .. max_decoding .. ' but actual max and min are ' .. part_filters:max() .. ', ' .. part_filters:min())
   part_filters:mul(-1)
   part_filters:add(max_decoding):div(2*max_decoding)

   -- white out the part filters above the final reconstruction and input display
   local excess_part_filters = sorted_reconstruction_filter:narrow(3,num_sorted_inputs+1,2):select(2,1)
   excess_part_filters:fill(1)
   --sorted_reconstruction_filter[{counter,1,{num_sorted_inputs+1, num_sorted_inputs+2},{}}]:fill(1)

   local accretion_filters = sorted_reconstruction_filter:select(2,2)
   local accretion_min, accretion_max = accretion_filters:min(), accretion_filters:max()
   accretion_filters:add(-1*accretion_min):div(accretion_max - accretion_min):mul(-1):add(1) -- map the dynamic range min,max -> 1,0 ; note that the dynamic range is reversed, scaled, and shifted

   print('total min and max are ' .. sorted_reconstruction_filter:min() .. ', ' .. sorted_reconstruction_filter:max())
   
   -- construct the full image from the composite pieces
   local image_size = decoding_filter:size(1)
   local filter_side_length_y = math.ceil(math.sqrt(image_size))
   local filter_side_length_x = image_size / filter_side_length_y
   --local current_filter = current_filter:unfold(1,current_filter_side_length, current_filter_side_length)
   
   local padding = 1
   local extra_padding = 8
   local total_extra_padding = (num_display_columns - num_sorted_inputs) * extra_padding
   local xmaps = num_display_columns
   local ymaps = num_display_rows_per_input * num_reconstructions_to_plot
   local height = filter_side_length_y + padding
   local width = filter_side_length_x + padding
   local white_value = 1 --(args.symmetric and math.max(math.abs(args.input:min()),math.abs(args.input:max()))) or args.input:max()
   local image_out = torch.Tensor(height*ymaps, width*xmaps + total_extra_padding):fill(white_value)
   for y = 1,ymaps do
      for x = 1,xmaps do
	 local current_extra_padding = math.max(x - num_sorted_inputs, 0) * extra_padding
	 local selected_image_region = image_out:narrow(1,(y-1)*height+1+padding/2,filter_side_length_y):narrow(2,(x-1)*width+1+padding/2 + current_extra_padding,filter_side_length_x)
	 selected_image_region:copy(sorted_reconstruction_filter[{math.ceil(y / num_display_rows_per_input), ((y-1) % num_display_rows_per_input) + 1, x, {}}]:unfold(1, filter_side_length_x, filter_side_length_x))
      end
   end
   image.savePNG(paths.concat(opt.log_directory, 'sorted_reconstruction_dictionary_columns.png'), image_out)

   --save_filter(sorted_reconstruction_filter, 'sorted reconstruction dictionary columns', opt.log_directory, num_display_columns) -- this depends upon save_filter using a row length of num_sorted_inputs!!!
end



function plot_part_sharing_histogram(encoding_filter, decoding_filter, activation_tensor, class_tensor, opt)
   -- dimensions of activation_tensor are: nExamples, hidden_layer_size
   -- class_tensor is a 1d tensor of class values
   -- go through all units.  check if the unit is a part-unit based upon the angle between the encoding and decoding filter.  If it is a part-unit, add the number of times it is non-zero for inputs of each class (this can be done slowly, since this is just diagnostic code).  Output these counts as a 2d image.
   
   local function categoricalness_enc_dec_alignment(i)
      local enc = encoding_filter:select(1,i)
      local dec = decoding_filter:select(2,i)
      local angle = math.acos(torch.dot(enc, dec)/(enc:norm() * dec:norm()))
      if angle > cat_thresh then return 'categorical' 
      elseif angle < part_thresh then return 'part' 
      else return 'intermediate' end 
   end


   local num_units = decoding_filter:size(2)
   local num_examples = class_tensor:size(1)
   local num_classes = class_tensor:max()
   local part_sharing_histogram = torch.Tensor(num_units, num_classes):zero()
   local part_unit_counter = 0

   for unit_counter = 1,num_units do
      -- check if unit is part_unit
      if(categoricalness_enc_dec_alignment(unit_counter) == 'part') then
	 part_unit_counter = part_unit_counter + 1
	 for example_counter = 1,num_examples do
	    if activation_tensor[{example_counter, unit_counter}] > 0 then
	       part_sharing_histogram[{part_unit_counter, class_tensor[example_counter]}] = part_sharing_histogram[{part_unit_counter, class_tensor[example_counter]}] + 1
	    end
	 end
      end
   end

   part_sharing_histogram = part_sharing_histogram:narrow(1,1,part_unit_counter)

   local _,preferred_class_by_unit = torch.max(part_sharing_histogram,2)
   preferred_class_by_unit = preferred_class_by_unit:select(2,1) -- strip out vestigial dimension
   local _,sort_order = torch.sort(preferred_class_by_unit)

   local sorted_part_sharing_histogram = torch.Tensor():resizeAs(part_sharing_histogram)
   for i = 1,part_unit_counter do
      sorted_part_sharing_histogram:select(1,i):copy(part_sharing_histogram:select(1,sort_order[i]))
   end

   gnuplot.pngfigure(opt.log_directory .. '/part_sharing_histogram.png') 
   gnuplot.imagesc(sorted_part_sharing_histogram)
   gnuplot.plotflush()
   gnuplot.closeall()
end

function project_from_sphere(old_coords, new_coords)
   local radius = old_coords:norm()
   radius = math.max(radius, 1e-3)
   new_coords[1] = math.acos(old_coords[3] / radius)
   new_coords[2] = math.atan(math.min(1e5, math.max(-1e-5, old_coords[2] / old_coords[1])))   
end

-- create a transformation function, which is project-from-sphere for 3d, and the identity function for 2d

-- plot the decoding trajectories formed by progressively accumulating the reconstruction contributions of the n units with the largest activations in response to each network input.  These are then connected to the final reconstruction, and then the original input, making clear the error of the reconstruction.  
function plot_reconstruction_connections_2d(decoding_filter, activation_tensor, input_tensor, class_tensor, opt, trajectory_plot_length)
   -- dimensions of activation_tensor are: (total_num_shrink_copies *only if* plot_temporal_reconstructions == 2), nExamples, hidden_layer_size
   -- different network inputs (elements of the dataset) go along the first dimension of activation_tensor (and input_tensor)
   -- different hidden units go along the second dimension of activation_tensor
   local i = -1 -- THIS SHOULD NOT BE USED; I CHANGED IT TO example_index, corresponding to num_examples
   local plot_temporal_reconstructions = opt.plot_temporal_reconstructions or false
   local base_trajectory_length, num_examples, hidden_layer_size
   if plot_temporal_reconstructions then
      if activation_tensor:dim() ~= 3 then
	 error('expected a 3d tensor for activation_tensor when plotting temporal reconstructions')
      end
      base_trajectory_length, num_examples, hidden_layer_size = activation_tensor:size(1), activation_tensor:size(2), activation_tensor:size(3)
      trajectory_plot_length = base_trajectory_length + 1 -- + 2 -- (DON'T) start at 0,0 ; (DON'T) end with true input
   else 
      if activation_tensor:dim() ~= 2 then
	 error('expected a 2d tensor for activation_tensor when *not* plotting temporal reconstructions')
      end
      num_examples, hidden_layer_size = activation_tensor:size(1), activation_tensor:size(2)
      trajectory_plot_length = trajectory_plot_length or 10
      trajectory_plot_length = math.min(hidden_layer_size + 2, trajectory_plot_length)
      base_trajectory_length = trajectory_plot_length - 2 --3 -- the first column is the origin; the last two columns are the final reconstruction (and the original input)
   end
   local num_reconstructions_to_plot = 500 --3 --num_examples -- reduce this to restrict to fewer examples; activation_tensor and input_tensor contain WAY TOO MANY examples to plot them all
   local preprojection_dim = decoding_filter:size(1)
   local output_dim = 2

   local project_to_2d
   if preprojection_dim == 2 then
      project_to_2d = function(old_coords, new_coords) new_coords:copy(old_coords) end
   elseif preprojection_dim == 3 then
      project_to_2d = project_from_sphere
   else
      error('expected 2- or 3-dimensional trajectories, rather than ' .. preprojection_dim)
   end
   
   if num_examples ~= input_tensor:size(1) then
      error('number of data set elements in activation tensor ' .. num_examples .. ' does not match the number in input tensor ' .. input_tensor:size(1))
   elseif (num_examples < num_reconstructions_to_plot) or (input_tensor:size(1) < num_reconstructions_to_plot) then
      error('number of data set elements in activation tensor ' .. num_examples .. ' or the number in input tensor ' .. input_tensor:size(1) .. ' is smaller than the number of requested trajectories ' .. num_reconstructions_to_plot)
   end

   -- reconstruction_trajectories holds all of the trajectories that will be used to build the total plot.  Its dimensions are: different images to reconstruct; columns, corresponding to steps in the reconstruction; the dimensions of the images/trajectories themselves (should be equal to 2)
   local reconstruction_trajectories = torch.Tensor(num_reconstructions_to_plot, trajectory_plot_length, output_dim):zero() -- tensor in which the figure will be constructed
   --local desired_examples_for_plot = {5, 10, 47} -- different digits: 6,4,7
   local desired_examples_for_plot = {88, 471, 379} -- all 3's -- 482
   local plot_args = {}
   local activation_mag_vector, activation_mag_vector_sorted, activation_mag_vector_sort_indices, desired_indices, num_points_per_traj, current_filter, last_filter
   local preprojection_filter = torch.Tensor(preprojection_dim)
   local last_preprojection_filter = torch.Tensor(preprojection_dim)
   for traj_num = 1,num_reconstructions_to_plot do
      local example_index = (desired_examples_for_plot and desired_examples_for_plot[traj_num]) or traj_num
      preprojection_filter:zero()
      last_preprojection_filter:zero() -- = reconstruction_trajectories[{traj_num,point_num,{}}]
      
      if not(plot_temporal_reconstructions) then
	 activation_mag_vector = activation_tensor:select(1,example_index):clone():abs() -- abs() probably isn't necessary, since units are non-negative by default, but we do it to be safe and forward-compatible
	 activation_mag_vector_sorted, activation_mag_vector_sort_indices = activation_mag_vector:sort(true)
	 desired_indices = activation_mag_vector_sort_indices:narrow(1,1,base_trajectory_length)
      end
      
      -- for each element of the reconstruction, in order of magnitude
      for point_num = 1,base_trajectory_length do
	 local offset = 1
	 --local offset = ((plot_temporal_reconstructions and 0) or 1)
	 current_filter = reconstruction_trajectories[{traj_num,point_num + offset,{}}] -- the starting point of all trajectories is 0,0, as set by the initial call to zero()
	 last_filter = reconstruction_trajectories[{traj_num,point_num + offset - 1,{}}]
	 -- the next point in the trajectory is the last point plus the scaled decoding column with hidden unit of the next largest magnitude
	 if plot_temporal_reconstructions then
	    torch.mv(preprojection_filter, decoding_filter, activation_tensor[{point_num, example_index, {}}])
	 else
	    preprojection_filter:add(last_preprojection_filter, activation_tensor[{example_index,desired_indices[point_num]}], decoding_filter:select(2,desired_indices[point_num]))
	    last_preprojection_filter:copy(preprojection_filter)
	    --current_filter:copy(last_filter):add(activation_tensor[{example_index,desired_indices[point_num]}], decoding_filter:select(2,desired_indices[point_num]))
	 end
	 project_to_2d(preprojection_filter, current_filter)
      end

      if not(plot_temporal_reconstructions) then
	 current_filter = reconstruction_trajectories[{traj_num,trajectory_plot_length-0,{}}] -- plot the final reconstruction, if we haven't done so already
	 torch.mv(preprojection_filter, decoding_filter, activation_tensor:select(1,example_index)) 
	 project_to_2d(preprojection_filter, current_filter)
      end

      -- plot the initial input
      --current_filter = reconstruction_trajectories[{traj_num,trajectory_plot_length,{}}]
      --current_filter:copy(input_tensor:select(1,example_index))

      if class_tensor[example_index] == 1 then
	 table.insert(plot_args, {reconstruction_trajectories[{traj_num,{},1}], reconstruction_trajectories[{traj_num,{},2}], '-r'})
      else
	 table.insert(plot_args, {reconstruction_trajectories[{traj_num,{},1}], reconstruction_trajectories[{traj_num,{},2}], '-b'})
      end
   end

   gnuplot.pngfigure(opt.log_directory .. '/reconstruction_connections.png') 
   gnuplot.plot(unpack(plot_args))
   gnuplot.plotflush()
   gnuplot.closeall()
end

function plot_energy_landscape_2d(model, data_set, pos_max, opt)
   local input = torch.Tensor()
   local target = torch.Tensor()
   local input_position = {0,0}
   local energy_landscape = torch.Tensor(pos_max,pos_max):zero()
   local err
   local class_landscape = torch.Tensor(pos_max,pos_max):zero()
   local _, class

   local plot_args = {}
   local traj = torch.Tensor(2,2) -- dims are: pos (1,2), dim (x,y)

   pos_max = pos_max or 100
   local num_traj_per_dim = 50
   local traj_interval = math.ceil(pos_max / num_traj_per_dim)
   
   local preprojection_dim = model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:size(1)
   local output_dim = 2

   local project_to_2d
   if preprojection_dim == 2 then
      project_to_2d = function(old_coords, new_coords) new_coords:copy(old_coords) end
   elseif preprojection_dim == 3 then
      project_to_2d = project_from_sphere
   else
      error('expected 2- or 3-dimensional trajectories, rather than ' .. preprojection_dim)
   end


   -- we can't use minibatches, since we need the exact loss function value for each input; using minibatches, a single aggregate loss is calculated for the entire minibatch
   for x = 1,pos_max do
      if x % 10 == 0 then
	 print('x = ' .. x)
      end
      for y = 1,pos_max do
	 input_position[1] = (x-1)/(pos_max-1)
	 input_position[2] = (y-1)/(pos_max-1)
	 input = data_set.data[input_position]:double() -- This doesn't copy memory if the type is already correct
	 target = data_set.labels[input_position]

	 model:set_target(target)
	 err = model:updateOutput(input)
	 energy_landscape[{pos_max + 1 - y, x}] = model.layers[1].module_list.L2_reconstruction_criterion.output --model.layers[1].module_list.feature_extraction_sparsifying_module.output --err[1]
	 --_, class = model.module_list.classification_dictionary.output:max(1)
	 --class_landscape[{pos_max + 1 - y, x}] = class[1]
	 class_landscape[{pos_max + 1 - y, x}] = model.module_list.classification_dictionary.output[1]

	 if (x % traj_interval == 0) and (y % traj_interval == 0) then
	    --traj:select(1,1):copy(input)
	    --traj:select(1,2):copy(model.layers[1].module_list.decoding_feature_extraction_dictionary.output)
	    project_to_2d(input, traj:select(1,1))
	    project_to_2d(model.layers[1].module_list.decoding_feature_extraction_dictionary.output, traj:select(1,2))
	    table.insert(plot_args, {traj:select(2,1):clone(), traj:select(2,2):clone(), '-'})
	 end
      end
   end

   --energy_landscape:minN(1.5)
   energy_landscape:add(0.2):log()
   gnuplot.pngfigure(opt.log_directory .. '/energy_landscape.png') 
   gnuplot.imagesc(energy_landscape)

   gnuplot.pngfigure(opt.log_directory .. '/class_landscape.png') 
   gnuplot.imagesc(class_landscape)

   gnuplot.pngfigure(opt.log_directory .. '/energy_trajectories.png') 
   gnuplot.plot(unpack(plot_args))

   gnuplot.plotflush()
   gnuplot.closeall()
end


   








function plot_filters(opt, time_index, filter_list)
   local current_filter
   -- each entry of the filter_list, with the key set to the name of the filter, consists of a table of {filter, 'encoder' or 'decoder'}; some filters may be nil
   for name,filt_pair in pairs(filter_list) do 
      if filt_pair[1] then 
	 if filt_pair[2] == 'encoder' then
	    current_filter = filt_pair[1]:transpose(1,2)
	    if name == 'encoding pooling dictionary_1' then
	       save_filter(torch.mm(filter_list['decoding feature extraction dictionary_1'][1], filt_pair[1]:transpose(1,2)), 'encoder reconstruction', opt.log_directory)
	    elseif name == 'explaining away_1' then
	       save_filter(torch.mm(filter_list['encoding feature extraction dictionary_1'][1]:transpose(1,2), filt_pair[1]:transpose(1,2)), 'explaining away reconstruction', opt.log_directory)
	       --save_filter(torch.mm(filter_list['encoding feature extraction dictionary_1'][1]:transpose(1,2), torch.add(filt_pair[1]:transpose(1,2), torch.diag(torch.ones(filt_pair[1]:size(1))))), 'explaining away reconstruction', opt.log_directory)
	       save_filter(torch.mm(filter_list['decoding feature extraction dictionary_1'][1], filt_pair[1]:transpose(1,2)), 'explaining away dec reconstruction', opt.log_directory)
	       -- with ones added to the diag, as is done implicitly during the updates
	       --save_filter(torch.mm(filter_list['decoding feature extraction dictionary_1'][1], torch.add(filt_pair[1]:transpose(1,2), torch.diag(torch.ones(filt_pair[1]:size(1))))), 'explaining away dec reconstruction', opt.log_directory)
	       -- with diag forcibly removed
	       --save_filter(torch.mm(filter_list['decoding feature extraction dictionary_1'][1], torch.add(filt_pair[1]:transpose(1,2), -1, torch.diag(torch.diag(filt_pair[1]:transpose(1,2))))), 'explaining away dec reconstruction', opt.log_directory)
	    end
	 elseif filt_pair[2] == 'decoder' then
	    current_filter = filt_pair[1]
	    if name == 'decoding pooling dictionary_1' then
	       save_filter(torch.mm(filter_list['decoding feature extraction dictionary_1'][1], filt_pair[1]), 'decoder reconstruction', opt.log_directory)
	    end
	 else
	    error('filter pair[2] for filter ' .. name .. ' was incorrectly set to ' .. filt_pair[2])
	 end
	 
	 --print('saving filter ', name)
	 save_filter(current_filter, name, opt.log_directory)
      else
	 print('ignoring filter ' .. name)
      end
   end -- loop over filter names
end

function plot_reconstructions(opt, input, output)
   local image_list = {input, output}
   local current_image

   --print('sizes are ', image_list[1]:size(), image_list[2]:size())
   --io.read()

   for i = 1,#image_list do
      current_image = image_list[i]
      if current_image:dim() == 2 then -- if we're using minibatches, select the first element of the minibatch
	 current_image = current_image:select(1,1)
      end
      local current_image_side_length = math.sqrt(current_image:size(1))
      current_image = current_image:unfold(1,current_image_side_length, current_image_side_length)
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
