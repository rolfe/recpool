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
local function loadFlatDataset(desired_data_set, max_load, alternative_access_method, offset, RESTRICT_TO_WINDOW)
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
   data = torch.Tensor(data_set.set_size, dim) -- the data consists of 3 32x32 color channels (R,G,B), with one full color channel encoded before the next.
   labels = torch.Tensor(data_set.set_size)

   -- The dataset is broken into pieces on the harddrive.  Load the fewest number required, starting from the beginning.
   for i = 1,math.min(#data_set.path_data_batches, math.ceil((offset + max_load)/data_set.batch_size)) do
      subset = torch.load(data_set.path_data_batches[i], 'ascii')
      data[{ {(i-1)*data_set.batch_size+1, i*data_set.batch_size} }] = subset.data:t():double() -- this appears to overload narrow on the first dimension, followed by a copy
      labels[{ {(i-1)*data_set.batch_size+1, i*data_set.batch_size} }] = subset.labels[1]:double()
   end
   -- labels range for 0 to n-1; 1 is added by the access method
   
   data = data:narrow(1,offset+1,max_load) --:reshape(max_load,3,32,32)
   labels = labels:narrow(1,offset+1,max_load) --:reshape(max_load,3,32,32)

   local nExample = max_load
   local dataset = {} -- this is the object that is actually returned, and which mediates access to the local variables bound by the closure
   
   if RESTRICT_TO_WINDOW then
      --require 'image'
      --image.display{image=torch.reshape(data[{ {1,256} }], 256,color_length,side_length,side_length), nrow=16, legend='Some samples from the data set before windowing'}
      
      if (type(RESTRICT_TO_WINDOW) ~= 'table') or #RESTRICT_TO_WINDOW ~= 2 then
	 error('RESTRICT_TO_WINDOW must be an array with two elements: {width, height}')
      elseif RESTRICT_TO_WINDOW[1] ~= RESTRICT_TO_WINDOW[2] then
	 error('Currently, restrict to window only allows square windows')
      end

      local old_side_length = side_length
      side_length = RESTRICT_TO_WINDOW[1]
      local side_offset = math.floor((old_side_length - side_length)/2)
      dim = color_length*(side_length^2)
      -- after unfolding, the original dimension iterates across groups; the last dimension iterates within groups; as a result, the two unfolds leave the color channel in the second dimension, and put x/y position in the third and fourth dimensions
      local data_window = data:unfold(2,old_side_length,old_side_length):unfold(2,old_side_length,old_side_length):transpose(3,4):narrow(3,side_offset,side_length):narrow(4,side_offset,side_length)
      data = torch.Tensor(max_load, dim)
      data:copy(data_window)
      
      --image.display{image=torch.reshape(data[{ {1,256} }], 256,color_length,side_length,side_length), nrow=16, legend='Some samples from the data set after windowing'}
   end

   --require 'image'
   --image.display{image=torch.reshape(data[{ {1,256} }], 256,3,32,32), nrow=16, legend='Some samples from the data set'}
   --image.display{image=torch.reshape(data[{ {1,256} }], 256,3*32,32), nrow=16, legend='Some samples from the data set'}
   
   function dataset:normalize(mean_, std_) -- mean-0, std-1 normalize each pixel separately
      local std = std_ or torch.std(data, 1, true) -- std and mean return 1xn tensors, so the useless first index needs to be stripped out below
      local mean = mean_ or torch.mean(data, 1)
      -- do this ridiculous iteration so it's easy to add the same value to all elements across dimension 1; otherwise, we probably need to use ger
      for i=1,dim do 
         data:select(2, i):add(-mean[1][i])
         if std[1][i] > 0 then
            data:select(2, i):div(std[1][i])
         end
      end
      return mean, std
   end

   function dataset:normalizeByColor(mean_) -- mean-0 each pixel separately, std-1 normalize each color
      local mean = mean_ or torch.mean(data, 1)
      mean = mean:select(1,1) -- eliminate the vestigial dimension over which we've averaged
      -- do this ridiculous iteration so it's easy to add the same value to all elements across dimension 1; otherwise, we probably need to use ger
      for i=1,dim do 
         data:select(2, i):add(-mean[i])
      end

      for i = 1,3 do
	 local chosen_color = data:narrow(2,1+side_length*side_length*(i-1),side_length*side_length)
	 chosen_color:div(torch.std(chosen_color)) -- note that this is normalized by n-1 rather than n
      end
   end

   function dataset:normalizeL2(desired_norm) -- set all elements of the dataset to be norm-1
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

   function dataset:useGrayscale() -- render the dataset from rgb into grayscale in the access method
      dataset.grayscale = true
   end


   function dataset:normalizeGlobal(mean_, std_) -- mean-0, std-1 normalize all pixels together
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:div(std)
      return mean, std
   end

   function dataset:nExample()
      return nExample
   end

   function dataset:dataSize()
      if dataset.grayscale then
	 return dim/3
      else
	 return dim
      end
   end

   function dataset:labelSize()
      return 1
   end

   function dataset:nClass()
      return 10
   end

   if alternative_access_method == 'recpool_net' then
      dataset.data = {}
      dataset.labels = {}
      setmetatable(dataset.data, {__index = function(self, index)
				                if dataset.grayscale then -- sum over the color dimension
						   -- after unfolding, the original dimension iterates across groups; the last dimension iterates within groups
						   return data[index]:unfold(1,side_length^2,side_length^2):sum(1):select(1,1)
						else
						   return data[index]
						end
                                             end})
      setmetatable(dataset.labels, {__index = function(self, index)
                                                return labels[index]+1
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


function cifar.loadTrainSet(maxLoad, alternative_access_method, offset)
   return loadFlatDataset('train', maxLoad, alternative_access_method, offset)
end

function cifar.loadTestSet(maxLoad, alternative_access_method, offset)
   return loadFlatDataset('test', maxLoad, alternative_access_method, offset)
end

function cifar.loadDataSet(params)
   return loadFlatDataset(params.train_or_test, params.maxLoad, params.alternative_access_method, params.offset, params.RESTRICT_TO_WINDOW)
end


--return cifar