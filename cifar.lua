----------------------------------------------------------------------
-- This script downloads and loads the CIFAR-10 dataset
-- http://www.cs.toronto.edu/~kriz/cifar.html
----------------------------------------------------------------------

cifar = {}
cifar.path_dataset = 'cifar-10-batches-t7'

local cifar_train = {}
cifar_train.path_data_batches = {}
for i = 1,5 do
   cifar_train.path_data_batches[i] = paths.concat(cifar.path_dataset, 'data_batch_' .. i .. '.t7')
end
cifar_train.set_size = 50000
cifar_train.batch_size = 10000

local cifar_test = {}
cifar_test.path_data_batches = {paths.concat(cifar.path_dataset, 'test_batch.t7')}
cifar_test.set_size = 10000
cifar_test.batch_size = 10000

function cifar:train_set_size()
   return 40000
end

function cifar:validation_set_size()
   return 10000
end

function cifar:test_set_size()
   return 10000
end



local function download()
   -- Note: files were converted from their original format
   -- to Torch's internal format.

   -- The CIFAR-10 dataset provides 6 files:
   --    + train: training data batches 1-5
   --    + test:  test data batch 1

   local tar = 'http://data.neuflow.org/data/cifar10.t7.tgz'
   
   if not paths.dirp('cifar-10-batches-t7') then
      print '==> downloading dataset'
      os.execute('wget ' .. tar)
      os.execute('tar xvf ' .. paths.basename(tar))
   end
end


-- max_load: number of elements to include in the loaded dataset
-- alternative_access_method: format with which the dataset should be returned 
-- offset specifies: number of elements of the dataset to skip (from the beginning); defaults to 0
-- RESTRICT_TO_WINDOW: array that holds the desire dimensions {width, height} of the desired window.  Currently must be square.  Defaults to 32x32, the full size of CIFAR
local function loadFlatDataset(desired_data_set, max_load, alternative_access_method, offset, RESTRICT_TO_WINDOW, DESIRED_WINDOW_SHIFTS)
   download()
   offset = offset or 0
   
   print('<cifar> loading dataset, requesting ' .. max_load .. ' elements')
   
   local data_set
   if desired_data_set == 'train' then
      data_set = cifar_train
   elseif desired_data_set == 'test' then
      data_set = cifar_test
   else
      error('unrecognized data set')
   end

   if max_load+offset > data_set.set_size then
      error('requested more data (' .. max_load + offset .. ') than is available (' .. data_set.set_size .. ')')
   end

   local side_length = 32 -- dataset is a closure, and so has access to side_length throughout
   local color_length = 3
   local dim = color_length*(side_length^2)
   
   local data, labels
   data = torch.Tensor(data_set.set_size, dim) -- the data consists of 3 32x32 color channels (R,G,B), with one full color channel (32x32) encoded before the next.
   labels = torch.Tensor(data_set.set_size)
   local windowed_side_length, side_offset, windowed_dim

   -- The dataset is broken into pieces on the harddrive.  Load the fewest number required, starting from the beginning.
   for i = 1,math.min(#data_set.path_data_batches, math.ceil((offset + max_load)/data_set.batch_size)) do
      subset = torch.load(data_set.path_data_batches[i], 'ascii')
      data[{ {(i-1)*data_set.batch_size+1, i*data_set.batch_size} }] = subset.data:t():double() -- this appears to overload narrow on the first dimension, followed by a copy
      labels[{ {(i-1)*data_set.batch_size+1, i*data_set.batch_size} }] = subset.labels[1]:double()
   end
   collectgarbage() -- since subset is unnecessary
   -- labels range for 0 to n-1; 1 is added by the access method

   
   data = data:narrow(1,offset+1,max_load) 
   labels = labels:narrow(1,offset+1,max_load) 

   local nExample, nWindowedExample = max_load, max_load
   local dataset = {} -- this is the object that is actually returned, and which mediates access to the local variables bound by the closure
   
   -- The dataset consumes an impractical amount of memory (hundreds of megabytes, extending to gigabytes) if all possible shifted windows are stored explicitly.  Instead, construct the shifted windows on the fly whenever the dataset is accessed.  This requires that the covariance matrix used for sphering/whitening be constructed manually, element-by-element, rather than using a matrix multiplication.  However, this need only be done once per run, and in fact could be done once and saved.
   if RESTRICT_TO_WINDOW then
      --require 'image'
      --image.display{image=torch.reshape(data[{ {1,256} }], 256,color_length,side_length,side_length), nrow=16, legend='Some samples from the data set before windowing'}
      
      if (type(RESTRICT_TO_WINDOW) ~= 'table') or #RESTRICT_TO_WINDOW ~= 2 then
	 error('RESTRICT_TO_WINDOW must be an array with two elements: {width, height}')
      elseif RESTRICT_TO_WINDOW[1] ~= RESTRICT_TO_WINDOW[2] then
	 error('Currently, restrict to window only allows square windows')
      end
      
      windowed_side_length = RESTRICT_TO_WINDOW[1]
      side_offset = math.floor((side_length - windowed_side_length)/2) --offset for the centered window; shifts are relative to this
      windowed_dim = color_length*(windowed_side_length^2)
      if DESIRED_WINDOW_SHIFTS then
	 if (type(DESIRED_WINDOW_SHIFTS) ~= 'table') or (#DESIRED_WINDOW_SHIFTS ~= 2) or (RESTRICT_TO_WINDOW[1] + 2*DESIRED_WINDOW_SHIFTS[1] > side_length) or 
	    (RESTRICT_TO_WINDOW[2] + 2*DESIRED_WINDOW_SHIFTS[2] > side_length) then
	    error('DESIRED_WINDOW_SHIFTS is incorrectly formatted')
	 end
	 nWindowedExample = max_load * (2*DESIRED_WINDOW_SHIFTS[1] + 1) * (2*DESIRED_WINDOW_SHIFTS[2] + 1)
      else
	 DESIRED_WINDOW_SHIFTS = {0,0}
	 nWindowedExample = max_load
      end
   end
      
   function dataset:convertToStaticGrayscale()
      if self.use_dynamic_grayscale then
	 error('cannot both use dynamic grayscale and convert to static grayscale')
      end

      if not(self.converted_to_static_grayscale) then -- only convert to grayscale once, since it actually contracts the data tensor
	 self.converted_to_static_grayscale = true

	 dim = dim/color_length
	 if windowed_dim then
	    windowed_dim = windowed_dim/color_length
	 end
	 color_length = 1
	 local new_data = torch.Tensor(data:size(1), 1, data:size(2)/color_length)
	 -- after unfolding, the original dimension iterates across groups; the last dimension iterates within groups; as a result, the two unfolds leave the color channel in the second dimension, and put x/y position in the third and fourth dimensions
	 new_data:sum(data:unfold(2,side_length^2,side_length^2), 2) --:sum(2):select(2,1))
	 new_data = new_data:select(2,1)

	 collectgarbage()
	 --image.display{image=torch.reshape(new_data[{ {1,256} }], 256,32,32), nrow=16, legend='Some samples from the data set after grayscale conversion'}

	 data = new_data
	 collectgarbage()
      end
   end

   function dataset:useDynamicGrayscale() -- render the dataset from rgb into grayscale in the access method
      if self.converted_to_static_grayscale then
	 error('cannot both use dynamic grayscale and convert to static grayscale')
      end
      self.use_dynamic_grayscale = true
   end


   
   function dataset:normalize(mean_, std_) -- mean-0, std-1 normalize each pixel separately
      local std = std_ or torch.std(data, 1, true):select(1,1) -- std and mean return 1xn tensors, so the useless first index needs to be stripped out
      local mean = mean_ or torch.mean(data, 1):select(1,1)
      -- do this ridiculous iteration so it's easy to add the same value to all elements across dimension 1; otherwise, we probably need to use ger
      for i=1,dim do 
         data:select(2, i):add(-mean[i])
         if std[i] > 0 then
            data:select(2, i):div(std[i])
         else
	    error('dimension ' .. i .. ' had standard deviation of 0')
	 end
      end
      return mean, std
   end

   function dataset:normalizeByColor(mean_) -- mean-0 each pixel separately, std-1 normalize each color
      local mean = mean_ or torch.mean(data, 1):select(1,1) -- eliminate the vestigial dimension over which we've averaged
      -- do this ridiculous iteration so it's easy to add the same value to all elements across dimension 1; otherwise, we probably need to use ger
      for i=1,dim do 
         data:select(2, i):add(-mean[i])
      end

      for i = 1,color_length do
	 local chosen_color = data:narrow(2,1+side_length*side_length*(i-1),side_length*side_length)
	 chosen_color:div(torch.std(chosen_color)) -- note that this is normalized by n-1 rather than n
      end
   end

   -- normalize pixel-wise and whiten the data using V * lambda^-0.5 * V^t; this eliminates correlations between pixels.  Otherwise, large spaces of correlated pixels dominate the L2 reconstruction loss, and most of the representational power of the model is wasted capturing the exact details of these correlated spaces, rather than learning independent pieces of information.  Note that the resulting center-surround filter (on images) resembles retinal/LGN receptive fields, which are subcortical and subject to heavy evolutionary pressure
   function dataset:sphere() 
      -- Even if the data will later be converted to grayscale, direct normalization is probably sufficient, since the three channels are then rendered mean-0, variance-1; and both mean and variance of independent random variables add. 
      local whitening_eigenvalue_offset = 0.1 -- the constant is necessary to keep eigenvalues near or equal to zero from exploding when they are raised to the power -0.5
      
      self.sphered_data = true
      self:convertToStaticGrayscale()
      print('Normalizing')
      self:normalize()
      print('zero check for normalization ' .. torch.sum(data,1):max() .. ', ' .. torch.sum(data,1):min())
      collectgarbage()
      --image.display{image=torch.reshape(data[{ {1,256} }], 256,32,32), nrow=16, legend='Some samples from the data set after normalization'}

      local sphere_transform
      print('Sphering')

      -- try to load sphering matrix
      local file_name = 'cifar_sphere_matrix.bin'
      local mf = torch.DiskFile(file_name,'r', true)
      if mf then
	 print('Loading sphering matrix from file')
	 mf = mf:binary()
	 sphere_transform = mf:readObject()
	 mf:close()
      else
	 local covariance = torch.Tensor(dim, dim):zero()
	 for i = 1,nExample do
	    covariance:addr(data:select(1,i), data:select(1,i))
	 end
	 covariance:div(nExample)
	 
	 local e_vals, e_vecs = torch.symeig(covariance, 'V')
	 
	 local zero_check = torch.add(torch.mm(torch.mm(covariance, e_vecs), torch.diag(torch.pow(e_vals, -1))), -1, e_vecs)
	 print('zero check for eigenvector decomposition ' .. torch.min(zero_check) .. ', ' .. torch.max(zero_check))
	 
	 local e_vals_sphere = torch.diag(torch.pow(torch.add(e_vals, whitening_eigenvalue_offset), -1/2)) -- the constant is necessary to keep eigenvalues near or equal to zero from exploding 
	 sphere_transform = torch.mm(e_vecs, torch.mm(e_vals_sphere, e_vecs:t()))
	 
	 --image.display{image=torch.reshape(e_vecs:narrow(2,1,256):t(), 256,32,32), nrow=16, legend='Some samples from the eigenvectors'}
	 --image.display{image=torch.reshape(sphere_transform:narrow(2,1,256):t(), 256,32,32), nrow=16, legend='Some samples from the whitening transform'}
	 --print(sphere_transform:select(2, 130):unfold(1,32,32))
	 
	 if nExample == cifar:train_set_size() then
	    -- save sphering matrix to file
	    print('Saving sphering matrix to file')
	    local mf = torch.DiskFile(file_name,'w'):binary()
	    mf:writeObject(sphere_transform)
	    mf:close()
	 end
      end

      local sphered_data = torch.Tensor():resizeAs(data)
      sphered_data:mm(data, sphere_transform:t()) -- keep in mind that the dimensions of data are nExample x dim
      data = sphered_data

      collectgarbage()

      --image.display{image=torch.reshape(data[{ {1,256} }], 256,32,32), nrow=16, legend='Some samples from the whitened data'}
   end

   function dataset:useDynamicNormalizeL2()
      self.use_dynamic_normalize_L2 = true
   end

   function dataset:normalizeStandard() -- standard normalization
      dataset:sphere()
      dataset:useDynamicNormalizeL2()
   end
      

   function dataset:normalizeL2(desired_norm) -- set all elements of the dataset to be norm-1 (or norm-desired_norm).  Each color channel is normalized separately
      self:normalizeByColor()

      desired_norm = desired_norm or 1
      print('normalizing: data has ' .. dim .. ' dimensions')

      local current_example
      for i=1,nExample do
         current_example = data:select(1, i)
	 current_example:div(torch.norm(current_example:unfold(1,side_length^2,side_length^2):sum(1):select(1,1)) / desired_norm)
	 if i % 1000 == 0 then
	    collectgarbage()
	 end
      end
   end

   function dataset:normalizeGlobal(mean_, std_) -- mean-0, std-1 normalize all pixels together
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:div(std)
      return mean, std
   end

   function dataset:nExample()
      if RESTRICT_TO_WINDOW then
	 return nWindowedExample 
      else
	 return nExample
      end
   end

   function dataset:dataSize()
      local output_dim
      if RESTRICT_TO_WINDOW then
	 output_dim = windowed_dim
      else
	 output_dim = dim
      end

      if self.use_dynamic_grayscale then
	 return output_dim/color_length
      else
	 return output_dim
      end
   end

   function dataset:labelSize()
      return 1
   end

   function dataset:nClass()
      return 10
   end


--[[

						   for x = -1*DESIRED_WINDOW_SHIFTS[1],DESIRED_WINDOW_SHIFTS[1] do
	 for y = -1*DESIRED_WINDOW_SHIFTS[2],DESIRED_WINDOW_SHIFTS[2] do
	    data_window = data:unfold(2,old_side_length,old_side_length):unfold(2,old_side_length,old_side_length):transpose(3,4):narrow(3,side_offset+x,side_length):narrow(4,side_offset+y,side_length)
	    new_data:narrow(1,1+data_offset*max_load,max_load):copy(data_window)
	    new_lables:narrow(1,1+data_offset*max_load,max_load):copy(labels)
	 end
						   end
						   --]]


   if alternative_access_method == 'recpool_net' then
      dataset.data = {}
      dataset.labels = {}
      local output_data_element = torch.Tensor()

      setmetatable(dataset.data, {__index = function(self, index)
				     local effective_dim = (RESTRICT_TO_WINDOW and windowed_dim) or dim
				     output_data_element:resize(effective_dim)

				     --local dataset_index = ((index - 1) % nExample) + 1 -- starts at 1
				     local nShifts = (2*DESIRED_WINDOW_SHIFTS[1] + 1) * (2*DESIRED_WINDOW_SHIFTS[2] + 1)
				     local dataset_index = math.floor((index - 1) / nShifts) + 1
				     if RESTRICT_TO_WINDOW then
					--local shift_index = math.floor((index - 1) / nExample) -- starts at 0
					local shift_index = math.floor((index - 1) % nShifts) 
					local num_x_shifts, num_y_shifts = 2*DESIRED_WINDOW_SHIFTS[1] + 1, 2*DESIRED_WINDOW_SHIFTS[2] + 1
					local shift_x = (shift_index % num_x_shifts) - DESIRED_WINDOW_SHIFTS[1]
					local shift_y = math.floor(shift_index  / num_x_shifts) - DESIRED_WINDOW_SHIFTS[2]
					
					-- after unfolding, the original dimension iterates across groups; the last dimension iterates within groups; as a result, the two unfolds leave the color channel in the second dimension, and put x/y position in the third and fourth dimensions
					local data_element = data[dataset_index]:unfold(1,side_length,side_length):unfold(1,side_length,side_length):transpose(2,3)
					data_element = data_element:narrow(2,side_offset+shift_y,windowed_side_length):narrow(3,side_offset+shift_x,windowed_side_length) -- first dim is color channel
					output_data_element:copy(data_element) -- effectively refold
				     else
					output_data_element:copy(data[dataset_index]) -- so we can normalize in place without altering the original data
				     end

				     -- can't use self.use_dynamic_grayscale, since the self passed to the __index function is dataset.data, rather than dataset
				     if dataset.use_dynamic_grayscale then -- sum over the color dimension
					-- after unfolding, the original dimension iterates across groups; the last dimension iterates within groups
					output_data_element = output_data_element:unfold(1,side_length^2,side_length^2):sum(1):select(1,1)
				     end
				     if dataset.use_dynamic_normalize_L2 then
					output_data_element:div(output_data_element:norm())
				     end
				     return output_data_element
      end})
      setmetatable(dataset.labels, {__index = function(self, index)
				       local dataset_index = ((index - 1) % nExample) + 1 -- starts at 1
				       return labels[dataset_index]+1
      end})
   elseif alternative_access_method == 'recpool_net_L2_classification' then
      --print('using L2 classification')
      --io.read()
      dataset.data = {}
      dataset.labels = {}
      local label_vector = torch.zeros(10)
      setmetatable(dataset.data, {__index = function(self, index)
				     return data[index]
      end})
      setmetatable(dataset.labels, {__index = function(self, index)
				       label_vector:zero()
				       local class = labels[index]+1
				       label_vector[class] = 1
				       --print('cifar access will return ', label_vector)
				       return label_vector
      end})
   else
      local labelvector = torch.zeros(10)
      setmetatable(dataset, {__index = function(self, index)
                                       local input = data[index]
                                       local class = labels[index]+1
                                       local this_label = labelvector:zero()
                                       this_label[class] = 1
                                       local example = {input, this_label}
                                       return example
                                    end})
   end

   return dataset
end


function cifar.loadTrainSet(maxLoad, alternative_access_method, offset, RESTRICT_TO_WINDOW, DESIRED_WINDOW_SHIFTS)
   return loadFlatDataset('train', maxLoad, alternative_access_method, offset, RESTRICT_TO_WINDOW, DESIRED_WINDOW_SHIFTS)
end

function cifar.loadTestSet(maxLoad, alternative_access_method, offset, RESTRICT_TO_WINDOW, DESIRED_WINDOW_SHIFTS)
   return loadFlatDataset('test', maxLoad, alternative_access_method, offset, RESTRICT_TO_WINDOW, DESIRED_WINDOW_SHIFTS)
end

function cifar.loadDataSet(params)
   return loadFlatDataset(params.train_or_test, params.maxLoad, params.alternative_access_method, params.offset, params.RESTRICT_TO_WINDOW, params.DESIRED_WINDOW_SHIFTS)
end

function cifar.debug()
   require 'image'
   local w_size = 16
   local dataset = cifar.loadTrainSet(40000, 'recpool_net', 0, {w_size, w_size}, {3, 3})
   dataset:sphere()
   dataset:useDynamicNormalizeL2()

   local test_images = torch.Tensor(256, dataset:dataSize())
   for i = 1,256 do
      test_images:select(1,i):copy(dataset.data[i])
   end
   image.display{image=torch.reshape(test_images[{ {1,256} }], 256,w_size,w_size), nrow=16, legend='Sequential samples from the whitened data'}

   for i = 1,256 do
      test_images:select(1,i):copy(dataset.data[math.random(dataset:nExample())])
   end
   image.display{image=torch.reshape(test_images[{ {1,256} }], 256,w_size,w_size), nrow=16, legend='Random samples from the whitened data'}
end

--return cifar