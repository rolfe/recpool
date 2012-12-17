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

local function save_filter(current_filter, filter_name, log_directory)
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
   local current_image = image.toDisplayTensor{input=current_filter,padding=1,nrow=10,symmetric=true}
   
   -- ideally, the pdf viewer should refresh automatically.  This 
   image.savePNG(paths.concat(log_directory, filter_name .. '.png'), current_image)
end

-- dataset is nExamples x input_dim
-- hidden_activation is nExamples x hidden_dim
function plot_optimal_dictionary(data_set, hidden_activation, filter_name, log_directory)
   --print('calling torch.gels on data of size')
   --print(data_set:size())
   --print(hidden_activation:size())
   --print(data_set:select(1,1):unfold(1,10,10))
   --print(hidden_activation:select(1,1):unfold(1,10,10))
   
   local num_active_units = 0
   local activation_norms = torch.Tensor(hidden_activation:size(2)):zero()
   for i=1,hidden_activation:size(2) do
      activation_norms[i] = hidden_activation:select(2,i):norm()
      if hidden_activation:select(2,i):norm() > 0.05 then
	 num_active_units = num_active_units + 1
      end
   end
   print('found ' .. num_active_units .. ' active units')
   --print('activations norms are ')
   --print(activation_norms:unfold(1,10,10))
   local conservative_hidden_activation = torch.Tensor(hidden_activation:size(1), num_active_units)
   num_active_units = 0
   for i=1,hidden_activation:size(2) do
      if hidden_activation:select(2,i):norm() > 0.05 then
	 num_active_units = num_active_units + 1
	 conservative_hidden_activation:select(2,num_active_units):copy(hidden_activation:select(2,i))
      end
   end

   -- add random noise to ensure that the matrix is invertible; otherwise, there's a problem if a unit is always silent
   local conservative_optimal_dictionary_matrix = torch.gels(data_set, conservative_hidden_activation) --:add(torch.rand(hidden_activation:size()):mul(1e-6))) 
   local optimal_dictionary_matrix = torch.Tensor(hidden_activation:size(2), data_set:size(2)):zero()
   
   num_active_units = 0
   for i=1,hidden_activation:size(2) do
      if hidden_activation:select(2,i):norm() > 0.05 then
	 num_active_units = num_active_units + 1
	 optimal_dictionary_matrix:select(1,i):copy(conservative_optimal_dictionary_matrix:select(1,num_active_units)) -- this ignores the extra rows --:narrow(1,1,hidden_activation:size(2)))
      end
   end

   --print('actual error is ' .. data_set:dist(conservative_hidden_activation*optimal_dictionary_matrix:narrow(1,1,conservative_hidden_activation:size(2))))
   print('actual error is ' .. data_set:dist(hidden_activation*optimal_dictionary_matrix))
   print('predicted error is ' .. math.sqrt(conservative_optimal_dictionary_matrix:narrow(1,conservative_hidden_activation:size(2)+1,
											  conservative_hidden_activation:size(1) - conservative_hidden_activation:size(2)):pow(2):sum()))
   --save_filter(optimal_dictionary_matrix:narrow(1,1,conservative_hidden_activation:size(2)):t(), filter_name, log_directory)
   save_filter(optimal_dictionary_matrix:t(), filter_name, log_directory)
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

      for i = 1,#accumulated_inputs do
	 local receptive_field_output = self:extract_receptive_fields(i)
	 save_filter(receptive_field_output, 'shrink receptive field ' .. i, opt.log_directory)
	 plot_optimal_dictionary(data_set_tensor, shrink_val_tensor:select(1,i), 'shrink dictionary ' .. i, opt.log_directory)
      end

      local activated_at_zero = torch.gt(shrink_val_tensor:select(1,1), 0):double():sum(1):select(1,1)
      local activated_at_one = torch.add(torch.gt(shrink_val_tensor:select(1,2), 0):double(), -1, torch.gt(shrink_val_tensor:select(1,1), 0):double()):maxZero():sum(1):select(1,1)
      local activated_at_end = torch.gt(shrink_val_tensor:select(1,shrink_val_tensor:size(1)), 0):double():sum(1):select(1,1)
      --local activated_after_zero = torch.gt(shrink_val_tensor:narrow(1,2,total_num_shrink_copies-1):sum(1):select(1,1), 0):double():sum(1):select(1,1) -- works since activities are non-negative
      local activated_ever = torch.gt(shrink_val_tensor:sum(1):select(1,1), 0):double():sum(1):select(1,1) -- works since activities are non-negative
      -- activated after zero but not at zero = activated_ever - activated_at_zero
      activated_ever[torch.le(activated_ever, 1)] = 1

      local percentage_late_activation = torch.cdiv(torch.add(activated_ever, -1, activated_at_zero), activated_ever)
      local percentage_first_iter_activation = torch.cdiv(activated_at_zero, activated_ever)
      local percentage_second_iter_activation = torch.cdiv(activated_at_one, activated_ever)
      local percentage_activated_at_end = torch.div(activated_at_end, shrink_val_tensor:size(2))
      --print('percentage late activation', percentage_late_activation:unfold(1,10,10))

      require 'gnuplot'
      local norm_vec = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      local enc_norm_vec = torch.Tensor(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight:size(1))
      local dec_norm_vec = torch.Tensor(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:size(2))
      local prod_norm_vec = torch.Tensor(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:size(2))
      local average_recurrent_pos_connection_angle = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      local average_recurrent_neg_connection_angle = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      local average_recurrent_pos_connection_categoricalness = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      local average_recurrent_neg_connection_categoricalness = torch.Tensor(model.layers[1].module_list.explaining_away.weight:size(1))
      --torch.diag(torch.mm(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight, model.layers[1].module_list.decoding_feature_extraction_dictionary.weight)), 
	 
      for i = 1,model.layers[1].module_list.explaining_away.weight:size(1) do
	 norm_vec[i] = model.layers[1].module_list.explaining_away.weight:select(1,i):norm()
	 enc_norm_vec[i] = model.layers[1].module_list.encoding_feature_extraction_dictionary.weight:select(1,i):norm()
	 dec_norm_vec[i] = model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:select(2,i):norm()
	 prod_norm_vec[i] = torch.dot(model.layers[1].module_list.encoding_feature_extraction_dictionary.weight:select(1,i), 
				      model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:select(2,i))
      end
      --print(norm_vec:unfold(1,10,10))
      local angle_between_encoder_and_decoder = torch.cdiv(prod_norm_vec, torch.cmul(enc_norm_vec, dec_norm_vec)):acos()

      for i = 1,model.layers[1].module_list.explaining_away.weight:size(1) do
	 local pos_norm, neg_norm, pos_weighted_sum_angle, neg_weighted_sum_angle, pos_weighted_sum_categoricalness, neg_weighted_sum_categoricalness = 0, 0, 0, 0, 0, 0
      	 for j = 1,model.layers[1].module_list.explaining_away.weight:size(2) do
	    if i ~= j then -- ignore the diagonal
	       local val_angle = math.abs(model.layers[1].module_list.explaining_away.weight[{i,j}]) *
		  math.acos(torch.dot(model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:select(2,i), 
				      model.layers[1].module_list.decoding_feature_extraction_dictionary.weight:select(2,j)) / 
			       (dec_norm_vec[i] * dec_norm_vec[j]))
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
	    end
	 end
	 pos_norm = (((pos_norm == 0) and 1) or pos_norm)
	 neg_norm = (((neg_norm == 0) and 1) or neg_norm)
	 average_recurrent_pos_connection_angle[i] = pos_weighted_sum_angle / pos_norm
	 average_recurrent_neg_connection_angle[i] = neg_weighted_sum_angle / neg_norm
	 average_recurrent_pos_connection_categoricalness[i] = pos_weighted_sum_categoricalness / pos_norm
	 average_recurrent_neg_connection_categoricalness[i] = neg_weighted_sum_categoricalness / neg_norm
      end		  	  

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
      gnuplot.plot(angle_between_encoder_and_decoder, norm_classification_connection)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('classification dictionary connection magnitude')

      gnuplot.figure() -- cos(angle) between encoder and decoder versus magnitude of recurrent input; categorical units have unaligned encoder/decoder pairs and larger recurrent connections
      gnuplot.plot(angle_between_encoder_and_decoder, average_recurrent_pos_connection_categoricalness)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('weighted average categoricalness between decoder and positively recurrently connected decoders')

      gnuplot.figure() -- cos(angle) between encoder and decoder versus magnitude of recurrent input; categorical units have unaligned encoder/decoder pairs and larger recurrent connections
      gnuplot.plot(angle_between_encoder_and_decoder, average_recurrent_neg_connection_categoricalness)
      gnuplot.xlabel('angle between encoder and decoder')
      gnuplot.ylabel('weighted average categoricalness between decoder and negatively recurrently connected decoders')



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

function plot_explaining_away_connections(decoding_filter, explaining_away_filter, opt)
   local num_sorted_inputs = 10
   local explaining_away_mag_filter = explaining_away_filter:clone():add(torch.diag(torch.ones(explaining_away_filter:size(2)))):abs() -- correct for the fact that the identity matrix needs to be added in manually
   local explaining_away_mag_filter_sorted, explaining_away_mag_filter_sort_indices = explaining_away_mag_filter:sort(2, true)
   local desired_indices = explaining_away_mag_filter_sort_indices:narrow(2,1,num_sorted_inputs)
   --local flattened_desired_indices = desired_indices:reshape(desired_indices:nElement()) -- takes element row-wise, and it is the rows that have been sorted and narrowed
   local sorted_recurrent_connection_filter = torch.Tensor(decoding_filter:size(1), num_sorted_inputs * explaining_away_filter:size(1))
   local output_filter_index = 1
   for i = 1,desired_indices:size(1) do
      for j = 1,desired_indices:size(2) do
	 local current_column = sorted_recurrent_connection_filter:select(2,output_filter_index)
	 current_column:copy(decoding_filter:select(2,desired_indices[{i,j}]))
	 current_column:mul(explaining_away_filter[{i,desired_indices[{i,j}]}]) -- + (((i == desired_indices[{i,j}]) and 1) or 0))
	 output_filter_index = output_filter_index + 1
      end
   end
   save_filter(sorted_recurrent_connection_filter, 'sorted recurrent connections', opt.log_directory) -- this depends upon save_filter using a row length of num_sorted_inputs!!!
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
