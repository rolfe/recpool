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


function receptive_field_builder_factory(nExamples, input_size, shrink_size, total_num_shrink_copies)
   local accumulated_inputs = {} -- array holding the (unscaled) receptive fields; initialized by the first call to accumulate_weighted_inputs
   local receptive_field_builder = {}
   local shrink_val_tensor = torch.Tensor(total_num_shrink_copies, nExamples, shrink_size)
   local data_set_tensor = torch.Tensor(nExamples, input_size)
   local data_set_index = 1

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
   
   function receptive_field_builder:accumulate_shrink_weighted_inputs(new_input, base_shrink, shrink_copies)
      local batch_size = new_input:size(1)
      if data_set_index >= nExamples then
	 error('accumulated ' .. data_set_index .. ' elements in the receptive field builder, but only expected ' .. nExamples)
      end

      data_set_tensor:narrow(1,data_set_index,batch_size):copy(new_input)

      self:accumulate_weighted_inputs(new_input, base_shrink.output, 1)
      shrink_val_tensor:select(1,1):narrow(1,data_set_index,batch_size):copy(base_shrink.output)
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
   end
   
   function receptive_field_builder:reset()
      data_set_index = 0
      for i = 1,#accumulated_inputs do
	 accumulated_inputs[i]:zero()
      end
   end
   
   return receptive_field_builder
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
	    --save_filter(torch.mm(filter_list[i-1], filter_list[i]:transpose(1,2)), 'explaining away dec reconstruction', opt.log_directory)
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
