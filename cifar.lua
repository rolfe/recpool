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


-- alternative_access_method specifies the format with which the dataset should be returned 
local function loadFlatDataset(desired_data_set, max_load, alternative_access_method, offset)
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
   
   local data, labels
   data = torch.Tensor(data_set.set_size, 3*32*32)
   labels = torch.Tensor(data_set.set_size)

   for i = 1,math.min(#data_set.path_data_batches, math.ceil((offset + max_load)/data_set.batch_size)) do
      subset = torch.load(data_set.path_data_batches[i], 'ascii')
      data[{ {(i-1)*data_set.batch_size+1, i*data_set.batch_size} }] = subset.data:t():double()
      labels[{ {(i-1)*data_set.batch_size+1, i*data_set.batch_size} }] = subset.labels[1]:double()
   end
   --labels:add(1) -- this is done by the access method
   
   data = data:narrow(1,offset+1,max_load) --:reshape(max_load,3,32,32)
   labels = labels:narrow(1,offset+1,max_load) --:reshape(max_load,3,32,32)

   -- Visualization is quite easy, using image.display(). Check out:
   -- help(image.display), for more info about options.

   --require 'image'
   --image.display{image=torch.reshape(data[{ {1,256} }], 256,3,32,32), nrow=16, legend='Some samples from the data set'}
   --image.display{image=torch.reshape(data[{ {1,256} }], 256,3*32,32), nrow=16, legend='Some samples from the data set'}
   
   local dim = 3*32*32
   local nExample = max_load
   local dataset = {} -- this is the object that is actually returned, and which mediates access to the local variables bound by the closure

   function dataset:normalize(mean_, std_) -- mean-0, std-1 normalize each pixel separately
      local std = std_ or torch.std(data, 1, true) -- std and mean return 1xn tensors, so the useless first index needs to be stripped out below
      local mean = mean_ or torch.mean(data, 1)
      for i=1,dim do
         data:select(2, i):add(-mean[1][i])
         if std[1][i] > 0 then
            data:select(2, i):mul(1/std[1][i])
         end
      end
      return mean, std
   end

   function dataset:normalize_by_color(mean_) -- mean-0, std-1 normalize each pixel separately
      local mean = mean_ or torch.mean(data, 1)
      for i=1,dim do
         data:select(2, i):add(-mean[1][i])
      end

      for i = 1,3 do
	 local chosen_color = data:narrow(2,1+32*32*(i-1),32*32)
	 chosen_color:div(torch.std(chosen_color)) -- note that this is normalized by n-1 rather than n
      end
   end

   function dataset:normalizeL2(desired_norm) -- set all elements of the dataset to be norm-1
      self:normalize_by_color()
      
      desired_norm = desired_norm or 1
      print('normalizing: data has ' .. dim .. ' dimensions')

      local current_example
      for i=1,nExample do
         current_example = data:select(1, i)
	 current_example:div(torch.norm(current_example) / desired_norm)
	 if i % 1000 == 0 then
	    collectgarbage()
	 end
      end
   end


   function dataset:normalizeGlobal(mean_, std_) -- mean-0, std-1 normalize all pixels together
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   function dataset:nExample()
      return nExample
   end

   function dataset:dataSize()
      return dim
   end

   function dataset:labelSize()
      return 1
   end

   if alternative_access_method == 'recpool_net' then
      dataset.data = {}
      dataset.labels = {}
      setmetatable(dataset.data, {__index = function(self, index)
				                return data[index] --:narrow(1,1,32*32)
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


--return cifar