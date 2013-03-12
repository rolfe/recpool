----------------------------------------------------------------------
-- This script downloads and loads the CIFAR-10 dataset
-- http://www.cs.toronto.edu/~kriz/cifar.html
----------------------------------------------------------------------

require 'xlua'    -- xlua provides useful tools, like progress bars

cifar_spec = {}
cifar_spec.path_dataset = 'cifar-10-batches-t7'

local cifar_train_spec = {}
cifar_train_spec.path_data_batches = {}
for i = 1,5 do
   cifar_train_spec.path_data_batches[i] = paths.concat(cifar_spec.path_dataset, 'data_batch_' .. i .. '.t7')
end
cifar_train_spec.set_size = 50000
cifar_train_spec.batch_size = 10000
cifar_train_spec.whitening_filter_name = 'cifar'

local cifar_test_spec = {}
cifar_test_spec.path_data_batches = {paths.concat(cifar_spec.path_dataset, 'test_batch.t7')}
cifar_test_spec.set_size = 10000
cifar_test_spec.batch_size = 10000
cifar_test_spec.whitening_filter_name = 'cifar'

function cifar_spec:train_set_size()
   return 40000
end

function cifar_spec:validation_set_size()
   return 10000
end

function cifar_spec:validation_set_offset()
   return self:train_set_size()
end

function cifar_spec:test_set_size()
   return 10000
end

-- return restrict_to_window, desired_window_shifts, window_shift_increment, desired_whitened_output_window
function cifar_spec:window_params()
   return {16, 16}, {4,4}, {2, 2}, {12, 12}
end

-- returns set size, window_shifts for quick and full diagnostic
function cifar_spec:diagnostic_params()
   return 20000, {0, 0}
end

-- returns set size, window_shifts for reconstruction_connections
function cifar_spec:reconstruction_params()
   return 100, {1, 1}, {1, 1}
end


berkeley_spec = {}
--berkeley_spec.path_dataset = '/Users/rolfe/Code/datasets/BSDS300/images/'
berkeley_spec.path_dataset = 'BSDS300/images/'
local berkeley_train_spec = {}
berkeley_train_spec.path_data_batches = {paths.concat(berkeley_spec.path_dataset, 'train'), paths.concat(berkeley_spec.path_dataset, 'test')}
berkeley_train_spec.set_size = 300
berkeley_train_spec.whitening_filter_name = 'berkeley'

function berkeley_spec:train_set_size()
   return 300
end

function berkeley_spec:validation_set_size()
   return 300
end

function berkeley_spec:validation_set_offset()
   return 0
end

function berkeley_spec:test_set_size()
   return 300
end

-- return restrict_to_window, desired_window_shifts, window_shift_increment, desired_whitened_output_window
function berkeley_spec:window_params()
   return {16, 16}, {9,9}, {16, 16}, {12, 12}
end

-- returns set size, window_shifts
function berkeley_spec:diagnostic_params()
   return 50, {9, 9}
end

-- returns set size, window_shifts, window_shift_increment
function berkeley_spec:reconstruction_params()
   return 10, {3, 3}, {16, 16}
   --return 60, {1, 1}, {1, 1}
end


local function download_cifar()
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

local function loadCifar(data_set_spec, offset, max_load)
   local side_length = 32 -- dataset is a closure, and so has access to side_length throughout
   local color_length = 3
   local dim = color_length*(side_length^2)
   
   local data, labels
   data = torch.Tensor(data_set_spec.set_size, dim) -- the data consists of 3 32x32 color channels (R,G,B), with one full color channel (32x32) encoded before the next.
   labels = torch.Tensor(data_set_spec.set_size)

   -- The dataset is broken into pieces on the harddrive.  Load the fewest number required, starting from the beginning.
   for i = 1,math.min(#data_set_spec.path_data_batches, math.ceil((offset + max_load)/data_set_spec.batch_size)) do
      subset = torch.load(data_set_spec.path_data_batches[i], 'ascii')
      data[{ {(i-1)*data_set_spec.batch_size+1, i*data_set_spec.batch_size} }] = subset.data:t():double() -- this appears to overload narrow on the first dimension, followed by a copy
      labels[{ {(i-1)*data_set_spec.batch_size+1, i*data_set_spec.batch_size} }] = subset.labels[1]:double()
   end
   collectgarbage() -- since subset is unnecessary
   -- labels range for 0 to n-1; 1 is added by the access method

   
   data = data:narrow(1,offset+1,max_load) 
   labels = labels:narrow(1,offset+1,max_load) 
   return side_length, color_length, dim, data, labels
end

local function scanDir(directory)
    local i, t, popen = 0, {}, io.popen
    for filename in popen('ls "'..directory..'"'):lines() do -- was ls -a
        i = i + 1
        t[i] = filename
    end
    return t
end

local function loadBerkeley(data_set_spec, offset, max_load)
   require 'image'

   local min_rows, min_cols
   local num_images = 0

   -- determine the minimum dimensions shared by all images
   for _,dir in pairs(berkeley_train_spec.path_data_batches) do
      local filenames = scanDir(dir)
      for _,file in pairs(filenames) do
	 local color_size, row_size, col_size = image.getJPGsize(dir .. '/' .. file)
	 min_rows = math.min(min_rows or row_size, row_size)
	 min_cols = math.min(min_cols or col_size, col_size)
	 num_images = num_images + 1
      end
   end

   print('Berkeley dataset contains ' .. num_images .. ' images')

   local side_length = math.min(min_rows, min_cols) -- extract a square common region
   local color_length = 3
   local dim = color_length*(side_length^2)

   local data, labels
   data = torch.Tensor(num_images, dim) -- the data consists of 3 nxn color channels (R,G,B), with one full color channel (32x32) encoded before the next.
   labels = torch.Tensor(num_images):zero()

   print('extracting regions of size ' .. side_length)
   local image_index = 0
   for _,dir in pairs(berkeley_train_spec.path_data_batches) do
      local filenames = scanDir(dir)
      for _,file in pairs(filenames) do
	 image_index = image_index + 1
	 local jpeg = image.loadJPG(dir .. '/' .. file)
	 local selected_region = jpeg:narrow(2,math.floor((jpeg:size(2) - side_length)/2) + 1, side_length):narrow(3,math.floor((jpeg:size(3) - side_length)/2) + 1, side_length)
	 data:select(1,image_index):copy(selected_region) -- should iterate in the correct order over the color and spatial dimensions
	 labels[image_index] = 0 -- this is an unlabeled dataset
	 
	 if image_index % 20 == 0 then
	    collectgarbage()
	 end
	 
	 --if image_index < 4 then
	 --   image.display(torch.reshape(data:select(1,image_index), color_length, side_length, side_length))
	 --end
      end
   end

   if image_index ~= num_images then
      error('expected ' .. num_images .. ' image files, but read ' .. image_index)
   else
      print('loaded ' .. num_images .. ' berkeley images, from both train and test sets')
   end
   collectgarbage()

   print('using offset ' .. offset .. ' max_load ' .. max_load)
   data = data:narrow(1,offset+1,max_load) 
   labels = labels:narrow(1,offset+1,max_load) 
   
   return side_length, color_length, dim, data, labels
end

function tb()
   loadBerkeley(berkeley_train_spec, 0, 300)
end
      

-- extract the center from all images into a common tensor.  This allows compatibility with cifar code, which assumes all images have the same size and so can efficiently be stored in a single tensor.  The other option would be to have a table to tensors, with one tensor for each image.  This is more flexible, and should be considered for later.  In the end, we will return random windows for each image, so restricting the size just alters the range over which the window can be chosen.

-- max_load: number of elements to include in the loaded dataset
-- alternative_access_method: format with which the dataset should be returned 
-- offset specifies: number of elements of the dataset to skip (from the beginning); defaults to 0
-- restrict_to_window: array that holds the desire dimensions {width, height} of the desired window.  Currently must be square.  Defaults to 32x32, the full size of CIFAR
-- desired_window_shifts: the number of shifts to perform (+ and -) on each dimension
-- window_shift_increment: the size of the shift to perform on each dimension (performed 2*desired_window_shifts + 1 times)
-- whitened_output_window: narrow the output further after applying the sphering/whitening transformation; this ensures that the pixels on the edge of the output window can be whitened based upon pixels in all directions
local function loadFlatDataset(desired_data_set_name, max_load, alternative_access_method, offset, restrict_to_window, desired_window_shifts, window_shift_increment, desired_whitened_output_window)
   download_cifar()
   offset = offset or 0
   
   print('<cifar> loading dataset, requesting ' .. max_load .. ' elements')
   
   local data_set_spec, side_length, color_length, dim, data, labels
   local use_all_data = false
   if (desired_data_set_name == 'train') or (desired_data_set_name == 'test') then
      if desired_data_set_name == 'train' then
	 data_set_spec = cifar_train_spec
      elseif desired_data_set_name == 'test' then
	 data_set_spec = cifar_test_spec
      else
	 error('unrecognized data set ' .. desired_data_set_name)
      end
      
      if max_load == cifar_spec:train_set_size() then
	 use_all_data = true
      end

      side_length, color_length, dim, data, labels = loadCifar(data_set_spec, offset, max_load)
   elseif desired_data_set_name == 'berkeley' then
      data_set_spec = berkeley_train_spec
      if max_load == berkeley_spec:train_set_size() then
	 use_all_data = true
      end
      side_length, color_length, dim, data, labels = loadBerkeley(data_set_spec, offset, max_load)
   else
      error('unrecognized data set' .. desired_data_set_name)
   end

   if max_load+offset > data_set_spec.set_size then
      error('requested more data (' .. max_load + offset .. ') than is available (' .. data_set_spec.set_size .. ')')
   end

   local windowed_side_length, windowed_side_offset, windowed_dim -- these local variables are part of the closure, but most accesses should probably be through the access functions defined for dataset below
   local output_windowed_side_length, output_windowed_side_offset, output_windowed_dim
   local nExample, nWindowedExample = max_load, max_load
   local dataset = {} -- this is the object that is actually returned, and which mediates access to the local variables bound by the closure
   
   -- The dataset consumes an impractical amount of memory (hundreds of megabytes, extending to gigabytes) if all possible shifted windows are stored explicitly.  Instead, construct the shifted windows on the fly whenever the dataset is accessed.  This requires that the covariance matrix used for sphering/whitening be constructed manually, element-by-element, rather than using a matrix multiplication.  However, this need only be done once per run, and in fact could be done once and saved.
   if restrict_to_window then
      --require 'image'
      --image.display{image=torch.reshape(data[{ {1,256} }], 256,color_length,side_length,side_length), nrow=16, legend='Some samples from the data set before windowing'}
      
      if (type(restrict_to_window) ~= 'table') or #restrict_to_window ~= 2 then
	 error('restrict_to_window must be an array with two elements: {width, height}')
      elseif restrict_to_window[1] ~= restrict_to_window[2] then
	 error('Currently, restrict to window only allows square windows')
      end
      
      windowed_side_length = restrict_to_window[1]
      windowed_side_offset = math.floor((side_length - windowed_side_length)/2) + 1 --offset for the centered window; shifts are relative to this
      windowed_dim = color_length*(windowed_side_length^2)
      if desired_window_shifts then
	 window_shift_increment = window_shift_increment or {1,1} -- default to shifting by a single pixel
	 if (type(window_shift_increment) ~= 'table') or (#window_shift_increment ~= 2) then
	    error('window_shift_increment is incorrectly formatted')
	 end
	 if (type(desired_window_shifts) ~= 'table') or (#desired_window_shifts ~= 2) or (restrict_to_window[1] + 2*desired_window_shifts[1]*window_shift_increment[1] > side_length) or 
	    (restrict_to_window[2] + 2*desired_window_shifts[2]*window_shift_increment[2] > side_length) then
	    error('desired_window_shifts is incorrectly formatted')
	 end
	 nWindowedExample = max_load * (2*desired_window_shifts[1] + 1) * (2*desired_window_shifts[2] + 1)
      else
	 desired_window_shifts = {0,0}
	 window_shift_increment = {0,0}
	 nWindowedExample = max_load
      end

   else
      windowed_side_length = side_length
      windowed_dim = dim
   end

   function dataset:setWhitenedOutputWindow(new_whitened_output_window)
      whitened_output_window = new_whitened_output_window
      if (type(whitened_output_window) ~= 'table') or #whitened_output_window ~= 2 then
	 error('whitened_output_window must be an array with two elements: {width, height}')
      elseif whitened_output_window[1] ~= whitened_output_window[2] then
	 error('Currently, restrict to window only allows square windows')
      end
      
      output_windowed_side_length = whitened_output_window[1]
      output_windowed_side_offset = math.floor((windowed_side_length - output_windowed_side_length)/2) + 1
      output_windowed_dim = color_length*(output_windowed_side_length^2)
   end


   if desired_whitened_output_window then -- this function must be defined before it can be called 
      dataset:setWhitenedOutputWindow(desired_whitened_output_window)
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
	 if output_windowed_dim then
	    output_windowed_dim = output_windowed_dim/color_length
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


   
   function dataset:normalizePixel(mean_, std_) -- mean-0, std-1 normalize each pixel separately
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

   -- mean-0, std-1 normalize, using a common value for all pixels.  This makes sense for a small set of large images.  We effectively assume that the image statistics are stationary/ergodic
   function dataset:normalizeGlobal(mean_, std_) 
      local std = std_ or torch.std(data) -- ideally, this would normalize by n rather than n-1, using the final true argument, but this is only available when the std is taken over a single dimension
      local mean = mean_ or torch.mean(data)
      data:add(-mean)
      if std > 0 then
	 data:div(std)
      else
	 error('standard deviation was <= 0')
      end
      return mean, std
   end


   function dataset:normalizeByColor(mean_) -- mean-0 each pixel and color separately, std-1 normalize each color
      local mean = mean_ or torch.mean(data, 1):select(1,1) -- eliminate the vestigial dimension over which we've averaged
      -- do this ridiculous iteration so it's easy to add the same value to all elements across dimension 1; otherwise, we probably need to use ger
      for i=1,dim do 
         data:select(2, i):add(-mean[i])
      end

      for i = 1,color_length do
	 local chosen_color = data:narrow(2,1+side_length*side_length*(i-1),side_length*side_length)
	 --chosen_color:add(-torch.mean(chosen_color)) -- mean-0 globally by color
	 chosen_color:div(torch.std(chosen_color)) -- note that this is normalized by n-1 rather than n
      end
   end

   -- normalize pixel-wise and whiten the data using V * lambda^-0.5 * V^t; this eliminates correlations between pixels.  Otherwise, large spaces of correlated pixels dominate the L2 reconstruction loss, and most of the representational power of the model is wasted capturing the exact details of these correlated spaces, rather than learning independent pieces of information.  Note that the resulting center-surround filter (on images) resembles retinal/LGN receptive fields, which are subcortical and subject to heavy evolutionary pressure
   function dataset:sphere() 
      if self.sphere_transform then
	 error('it is not possible to reconstruction the sphere transform, since it is built using a data access method that makes use of the previous sphere transform')
      end

      local orig_output_window = whitened_output_window -- the output window must be expanded to its full size while constructing the whitening filter
      whitened_output_window = nil

      -- Even if the data will later be converted to grayscale, direct normalization is probably sufficient, since the three channels are then rendered mean-0, variance-1; and both mean and variance of independent random variables add. 
      local whitening_eigenvalue_offset = 0.1 -- the constant is necessary to keep eigenvalues near or equal to zero from exploding when they are raised to the power -0.5
      
      self:convertToStaticGrayscale()
      print('Normalizing')
      self:normalizeGlobal()
      --self:normalizePixel()
      print('zero check for normalization ' .. torch.sum(data,1):max() .. ', ' .. torch.sum(data,1):min())
      collectgarbage()
      --image.display{image=torch.reshape(data[{ {1,256} }], 256,32,32), nrow=16, legend='Some samples from the data set after normalization'}

      local sphere_transform
      print('Sphering')

      -- try to load sphering matrix
      local file_name = data_set_spec.whitening_filter_name .. '_window_' .. windowed_side_length .. '_color_' .. color_length .. '_sphere_matrix.bin'
      local mf = torch.DiskFile(file_name,'r', true)
      if mf then
	 print('Loading sphering matrix from file')
	 mf = mf:binary()
	 sphere_transform = mf:readObject()
	 mf:close()
      else
	 print('construct covariance matrix using ' .. self:nExample() .. ' examples')
	 local covariance = torch.Tensor(self:dataSize(), self:dataSize()):zero()
	 for i = 1,self:nExample() do -- if windowed, iterated over all windows
	    local current_example = self.data[i] --data:select(1,i)
	    covariance:addr(current_example, current_example)
	 end
	 covariance:div(self:nExample())
	 
	 print('perform eigendecomposition')
	 local e_vals, e_vecs = torch.symeig(covariance, 'V')
	 
	 local zero_check = torch.add(torch.mm(torch.mm(covariance, e_vecs), torch.diag(torch.pow(e_vals, -1))), -1, e_vecs)
	 print('zero check for eigenvector decomposition ' .. torch.min(zero_check) .. ', ' .. torch.max(zero_check))
	 
	 local e_vals_sphere = torch.diag(torch.pow(torch.add(e_vals, whitening_eigenvalue_offset), -1/2)) -- the constant is necessary to keep eigenvalues near or equal to zero from exploding 
	 sphere_transform = torch.mm(e_vecs, torch.mm(e_vals_sphere, e_vecs:t()))
	 
	 --image.display{image=torch.reshape(e_vecs:narrow(2,1,256):t(), 256,windowed_side_length,windowed_side_length), nrow=16, legend='Some samples from the eigenvectors'}
	 --image.display{image=torch.reshape(sphere_transform:narrow(2,1,256):t(), 256,windowed_side_length,windowed_side_length), nrow=16, legend='Some samples from the whitening transform'}
	 --print(sphere_transform:select(2, 130):unfold(1,32,32))
	 
	 if use_all_data then -- save sphering matrix to file
	    print('Saving sphering matrix to file')
	    local mf = torch.DiskFile(file_name,'w'):binary()
	    mf:writeObject(sphere_transform)
	    mf:close()
	 end
      end

      self.sphere_transform = sphere_transform
      whitened_output_window = orig_output_window

      -- rather than statically transforming the data in place, apply the whitening matrix dynamically whenever the data is requested.  This is necessary with dynamic windowing, and with a sphering matrix only defined within each window.  This may be problematic since pixels on the edges of the window are only whitened based upon the elements in the center of the window.  Try it and find out!
      --local sphered_data = torch.Tensor():resizeAs(data)
      --sphered_data:mm(data, sphere_transform:t()) -- keep in mind that the dimensions of data are nExample x dim
      --data = sphered_data

      collectgarbage()

      --image.display{image=torch.reshape(data[{ {1,256} }], 256,32,32), nrow=16, legend='Some samples from the whitened data'}
   end

   function dataset:useDynamicNormalizeL2(desired_norm)
      self.use_dynamic_normalize_L2 = true
      if desired_norm then
	 self.dynamic_norm = desired_norm
      end
   end

   -- normalizing each window separately corresponds to something like local contrast normalization, since the L2 norm of the hidden units must be roughly proportional to the L2 norm of the input, given the constraint on the decoding matrix columns and the L2 reconstruction loss.  However, this implies that the network should amplify the structure in windows that are basically flat and featureless.  As a result, it will devote more energy to representing "textures" and noise, rather than objects.  Since MNIST digits are canonical objects, if we want to see corresponding development of categorical- and part-units, we should probably avoid amplifying the noise in featureless windows.  
   function dataset:findMaxL2Norm()
      local max_l2_norm = 0
      print('finding max L2 norm in dataset')
      for i = 1,self:nExample() do -- if windowed, iterated over all windows
	 max_l2_norm = max_l2_norm + self.data[i]:norm() --math.max(max_l2_norm, self.data[i]:norm())
      end
      max_l2_norm = max_l2_norm / self:nExample()
      return max_l2_norm
   end      

   function dataset:normalizeStandard() -- standard normalization
      dataset:sphere()
      --dataset:useDynamicNormalizeL2()
      local max_L2_norm = dataset:findMaxL2Norm()
      dataset:useDynamicNormalizeL2(max_L2_norm) -- this is equivalent to directly scaling the inputs.  However, it results in the variance per pixel being a value other than one
   end
      

   function dataset:normalizeL2(desired_norm) -- set all elements of the dataset to be norm-1 (or norm-desired_norm).  Each color channel is normalized separately
      self:normalizeByColor()

      desired_norm = desired_norm or 1
      print('normalizing: data has ' .. dim .. ' dimensions')

      local current_example
      for i=1,nExample do
         current_example = data:select(1, i)
	 current_example:div(torch.norm(current_example) / desired_norm) -- version for 12/7 CIFAR run
	 --current_example:div(torch.norm(current_example:unfold(1,side_length^2,side_length^2):sum(1):select(1,1)) / desired_norm) -- current version
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
      if restrict_to_window then
	 return nWindowedExample 
      else
	 return nExample
      end
   end

   function dataset:dataSize()
      local output_dim
      if whitened_output_window then
	 output_dim = output_windowed_dim
      elseif restrict_to_window then
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

						   for x = -1*desired_window_shifts[1],desired_window_shifts[1] do
	 for y = -1*desired_window_shifts[2],desired_window_shifts[2] do
	    data_window = data:unfold(2,old_side_length,old_side_length):unfold(2,old_side_length,old_side_length):transpose(3,4):narrow(3,windowed_side_offset+x,side_length):narrow(4,windowed_side_offset+y,side_length)
	    new_data:narrow(1,1+data_offset*max_load,max_load):copy(data_window)
	    new_lables:narrow(1,1+data_offset*max_load,max_load):copy(labels)
	 end
						   end
						   --]]


   if alternative_access_method == 'recpool_net' then
      dataset.data = {}
      dataset.labels = {}
      local output_data_element = torch.Tensor()
      local windowed_output_data_element = torch.Tensor()

      setmetatable(dataset.data, {__index = function(self, index)
				     local effective_dim = (restrict_to_window and windowed_dim) or dim
				     output_data_element:resize(effective_dim)
				     if whitened_output_window then
					windowed_output_data_element:resize(output_windowed_dim)
				     end

				     --local dataset_index = ((index - 1) % nExample) + 1 -- starts at 1
				     local nShifts = (2*desired_window_shifts[1] + 1) * (2*desired_window_shifts[2] + 1)
				     local dataset_index = math.floor((index - 1) / nShifts) + 1
				     if restrict_to_window then
					--local shift_index = math.floor((index - 1) / nExample) -- starts at 0
					local shift_index = math.floor((index - 1) % nShifts) 
					local num_x_shifts, num_y_shifts = 2*desired_window_shifts[1] + 1, 2*desired_window_shifts[2] + 1
					local shift_x = ((shift_index % num_x_shifts) - desired_window_shifts[1]) * window_shift_increment[1]
					local shift_y = (math.floor(shift_index  / num_x_shifts) - desired_window_shifts[2]) * window_shift_increment[2]
					
					-- after unfolding, the original dimension iterates across groups; the last dimension iterates within groups; as a result, the two unfolds leave the color channel in the second dimension, and put x/y position in the third and fourth dimensions
					local data_element = data[dataset_index]:unfold(1,side_length,side_length):unfold(1,side_length,side_length):transpose(2,3)
					-- first dim is color channel
					data_element = data_element:narrow(2,windowed_side_offset+shift_y,windowed_side_length):narrow(3,windowed_side_offset+shift_x,windowed_side_length) 
					output_data_element:copy(data_element) -- effectively refold
				     else
					output_data_element:copy(data[dataset_index]) -- so we can normalize in place without altering the original data
				     end

				     -- can't use self.use_dynamic_grayscale, since the self passed to the __index function is dataset.data, rather than dataset
				     if dataset.use_dynamic_grayscale then -- sum over the color dimension
					-- after unfolding, the original dimension iterates across groups; the last dimension iterates within groups
					output_data_element = output_data_element:unfold(1,side_length^2,side_length^2):sum(1):select(1,1)
				     end

				     if dataset.sphere_transform then -- this may be problematic if this access method is used in constructing the sphere transform repeatedly!
					dataset.sphere_temp = torch.Tensor():resizeAs(output_data_element) -- avoid allocating new memory on each operation
					dataset.sphere_temp:mv(dataset.sphere_transform, output_data_element) 
					output_data_element:copy(dataset.sphere_temp)
				     end

				     local final_output_val
				     if whitened_output_window then
					local nar_out = output_data_element:unfold(1,windowed_side_length,windowed_side_length):unfold(1,windowed_side_length,windowed_side_length):transpose(2,3)
					nar_out = nar_out:narrow(2,output_windowed_side_offset,output_windowed_side_length):narrow(3,output_windowed_side_offset,output_windowed_side_length) 
					windowed_output_data_element:copy(nar_out)
					final_output_val = windowed_output_data_element
				     else
					final_output_val = output_data_element
				     end

				     --print('norm is ' .. final_output_val:norm())
				     if dataset.use_dynamic_normalize_L2 then
					if dataset.dynamic_norm then
					   final_output_val:div(dataset.dynamic_norm)
					else
					   final_output_val:div(final_output_val:norm())
					end
				     end

				     zero_mean = false
				     if zero_mean then
					print('mean is : ' .. final_output_val:mean() .. ', L2 norm is: ' .. final_output_val:norm())
					final_output_val:add(-1 * final_output_val:mean())
				     end

				     return final_output_val
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


function cifar_spec:loadTrainSet(maxLoad, alternative_access_method, offset, restrict_to_window, desired_window_shifts)
   return loadFlatDataset('train', maxLoad, alternative_access_method, offset, restrict_to_window, desired_window_shifts)
end

function cifar_spec:loadTestSet(maxLoad, alternative_access_method, offset, restrict_to_window, desired_window_shifts)
   return loadFlatDataset('test', maxLoad, alternative_access_method, offset, restrict_to_window, desired_window_shifts)
end

function cifar_spec:loadDataSet(params)
   return loadFlatDataset(params.train_or_test, params.maxLoad, params.alternative_access_method, params.offset, params.restrict_to_window, params.desired_window_shifts, params.window_shift_increment, params.desired_whitened_output_window)
end

function berkeley_spec:loadDataSet(params)
   return loadFlatDataset('berkeley', params.maxLoad, 'recpool_net', params.offset, params.restrict_to_window, params.desired_window_shifts, params.window_shift_increment, params.desired_whitened_output_window)
end


function cd()
   require 'image'
   local w_size = 16
   local dataset = cifar_spec:loadTrainSet(40000, 'recpool_net', 0, {w_size, w_size}, {3, 3})
   dataset:sphere()
   local max_L2_norm = dataset:findMaxL2Norm()
   dataset:useDynamicNormalizeL2(max_L2_norm)

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

function bd()
   require 'image'
   local w_size = 16
   local ow_size = 12
   --loadFlatDataset(desired_data_set_name, max_load, alternative_access_method, offset, restrict_to_window, desired_window_shifts, window_shift_increment, desired_whitened_output_window)
   local dataset = loadFlatDataset('berkeley', 300, 'recpool_net', 0, {w_size, w_size}, {9,9}, {w_size, w_size}, {ow_size, ow_size})

   local test_images = torch.Tensor(256, dataset:dataSize())
   for i = 1,256 do
      test_images:select(1,i):copy(dataset.data[i])
   end
   image.display{image=torch.reshape(test_images[{ {1,256} }], 256,3,ow_size,ow_size), nrow=19, legend='Sequential samples from the original data'}

   dataset:sphere()
   dataset:useDynamicNormalizeL2()
   --local max_L2_norm = dataset:findMaxL2Norm()
   --dataset:useDynamicNormalizeL2(max_L2_norm)

   test_images = torch.Tensor(256, dataset:dataSize())
   for i = 1,256 do
      test_images:select(1,i):copy(dataset.data[i])
   end
   image.display{image=torch.reshape(test_images[{ {1,256} }], 256,ow_size,ow_size), nrow=19, legend='Sequential samples from the whitened data'}

   for i = 1,500 do
      local rand_sample = dataset.data[math.random(dataset:nExample())]
      print('mean is ' .. rand_sample:mean() .. ' norm is ' .. rand_sample:norm())
   end
end

--return cifar_spec