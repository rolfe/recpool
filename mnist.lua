require 'torch'
require 'paths'

mnist_spec = {}

mnist_spec.path_remote = 'http://data.neuflow.org/data/mnist-th7.tgz'
mnist_spec.path_dataset = 'mnist-th7'
mnist_spec.path_trainset = paths.concat(mnist_spec.path_dataset, 'train.th7')
mnist_spec.path_testset = paths.concat(mnist_spec.path_dataset, 'test.th7')

function mnist_spec:train_set_size()
   return 50000
end

function mnist_spec:validation_set_size()
   return 10000
end

--function mnist_spec:train_set_size()
--   return self:train_set_size()
--end


function mnist_spec:test_set_size()
   return 10000
end

-- return restrict_to_window, desired_window_shifts, window_shift_increment, desired_whitened_output_window
function mnist_spec:window_params()
   return nil, nil, nil, nil
end

-- returns set size, window_shifts for quick and full diagnostic
function mnist_spec:diagnostic_params()
   return 10000, nil
end

-- returns set size, window_shifts for reconstruction_connections
function mnist_spec:reconstruction_params()
   return 500, nil, nil
end



local function download()
   if not paths.filep(mnist_spec.path_trainset) or not paths.filep(mnist_spec.path_testset) then
      local remote = mnist_spec.path_remote
      local tar = paths.basename(remote)
      os.execute('wget ' .. remote .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
   end
end

-- alternative_access_method specifies the format with which the dataset should be returned 
local function loadFlatDataset(fileName, maxLoad, alternative_access_method, offset)
   download()

   local f = torch.DiskFile(fileName, 'r')
   f:binary()

   local nExample = f:readInt()
   local dim = f:readInt()
   print('<mnist> dataset has ' .. nExample .. ' elements of dimension ' .. dim-1 .. '+1')
   if offset then
      maxLoad = maxLoad + offset
   end

   if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<mnist> loading only ' .. nExample .. ' examples')
   elseif maxLoad and maxLoad == nExample then
      print('<mnist> loading all ' .. nExample .. ' examples')
   end

   print('<mnist> reading ' .. nExample .. ' examples with ' .. dim-1 .. '+1 dimensions...')
   local tensor = torch.Tensor(nExample, dim)
   tensor:storage():copy(f:readFloat(maxLoad*dim))
   print('<mnist> done')

   if offset then
      nExample = nExample - offset
   end
   local dataset = {}
   dataset.tensor = tensor
   if offset then
      tensor = tensor:narrow(1,offset+1, nExample)
   end
   if nExample ~= tensor:size(1) then
      error('dataset was not properly offset: nExample = ' .. nExample .. ' but tensor:size(1) = ' .. tensor:size(1))
   end
   

   function dataset:normalize(mean_, std_)
      local data = tensor:narrow(2, 1, dim-1)
      local std = std_ or torch.std(data, 1, true)
      local mean = mean_ or torch.mean(data, 1)
      for i=1,dim-1 do
         data:select(2, i):add(-mean[1][i])
         if std[1][i] > 0 then
            data:select(2, i):mul(1/std[1][i])
         end
      end
      return mean, std
   end

   function dataset:normalizeL2(desired_norm) -- added by Jason 6/8/12
      desired_norm = desired_norm or 1
      print('normalizing: data has ' .. dim-1 .. ' dimensions')
      local data = tensor:narrow(2, 1, dim-1)
      --local L2norm = torch.norm(data)
      local current_example
      for i=1,nExample do
         current_example = data:select(1, i)
	 current_example:div(torch.norm(current_example) / desired_norm)
      end
   end

   function dataset:useGrayscale() -- this need not do anything, since MNIST is already in grayscale
   end

   function dataset:normalizeStandard(desired_norm)
      self:normalizeL2(desired_norm)
      self:useGrayscale()
   end

   function dataset:normalizeGlobal(mean_, std_)
      local data = tensor:narrow(2, 1, dim-1)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   dataset.dim = dim-1

   function dataset:nExample()
      return nExample
   end

   function dataset:dataSize()
      return dim-1
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
                                                return tensor[index]:narrow(1, 1, dim-1)
                                             end})
      setmetatable(dataset.labels, {__index = function(self, index)
                                                return tensor[index][dim]+1
                                             end})
   elseif alternative_access_method == 'recpool_net_L2_classification' then
      --print('using L2 classification')
      --io.read()
      dataset.data = {}
      dataset.labels = {}
      local label_vector = torch.zeros(10)
      setmetatable(dataset.data, {__index = function(self, index)
				     return tensor[index]:narrow(1, 1, dim-1)
      end})
      setmetatable(dataset.labels, {__index = function(self, index)
				       label_vector:zero()
				       local class = tensor[index][dim]+1
				       label_vector[class] = 1
				       --print('mnist access will return ', label_vector)
				       return label_vector
      end})
   else
      local labelvector = torch.zeros(10)
      setmetatable(dataset, {__index = function(self, index)
                                       local input = tensor[index]:narrow(1, 1, dim-1)
                                       local class = tensor[index][dim]+1
                                       local label = labelvector:zero()
                                       label[class] = 1
                                       local example = {input, label}
                                       return example
                                    end})
   end

   return dataset
end

function mnist_spec:loadTrainSet(maxLoad, alternative_access_method, offset)
   return loadFlatDataset(mnist_spec.path_trainset, maxLoad, alternative_access_method, offset)
end

function mnist_spec:loadTestSet(maxLoad, alternative_access_method, offset)
   return loadFlatDataset(mnist_spec.path_testset, maxLoad, alternative_access_method, offset)
end

function mnist_spec:loadDataSet(params)
   local chosen_path
   if params.train_or_test == 'train' then
      chosen_path = mnist_spec.path_trainset
   elseif params.train_or_test == 'test' then
      chosen_path = mnist_spec.path_testset
   else
      error('unrecognized train/test request')
   end
   return loadFlatDataset(chosen_path, params.maxLoad, params.alternative_access_method, params.offset)
end


return mnist_spec