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

function plot_filters(opt, time_index, filter_list, filter_enc_dec_list, filter_name_list)

   function save_filter(current_filter, filter_name, log_directory)
      local current_filter_side_length = math.sqrt(current_filter:size(1))
      current_filter = current_filter:unfold(1,current_filter_side_length, current_filter_side_length):transpose(1,2)
      local current_image = image.toDisplayTensor{input=current_filter,padding=1,nrow=10,symmetric=true}

      -- ideally, the pdf viewer should refresh automatically.  This 
      image.savePNG(paths.concat(log_directory, filter_name .. '.png'), current_image)
   end

   for i = 1,#filter_list do
      local current_filter
      if filter_enc_dec_list[i] == 'encoder' then
	 current_filter = filter_list[i]:transpose(1,2)
	 --print('processing ' .. filter_name_list[i] .. ' as a encoder', current_filter:size())
	 if filter_name_list[i] == 'encoding pooling dictionary_1' then
	    --print('combining ' .. filter_name_list[i-2] .. ' with ' .. filter_name_list[i])
	    --print(filter_list[i-2]:size(), filter_list[i]:transpose(1,2):size())
	    save_filter(torch.mm(filter_list[i-2], filter_list[i]:transpose(1,2)), 'encoder reconstruction', opt.log_directory)
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
   -- flatten the parameters for storage.  While this has already been done in the trainer, the trainer probably shouldn't be responsible for saving and loading the parameters
   --local flattened_parameters = model:getParameters() 

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

function load_parameters(flattened_parameters, file_name)
   -- flatten the parameters for loading from storage.  While this has already been done in the trainer, the trainer probably shouldn't be responsible for saving and loading the parameters
   --local flattened_parameters = model:getParameters() 

   print('loading flattened parameters from ' .. file_name)
   local mf = torch.DiskFile(file_name,'r'):binary()
   local saved_parameters = mf:readObject()
   flattened_parameters:copy(saved_parameters)
   mf:close()

   --flattened_parameters = nil
   saved_parameters = nil
   collectgarbage()
end
