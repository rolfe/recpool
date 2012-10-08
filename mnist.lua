require 'torch'
require 'paths'

mnist = {}

mnist.path_remote = 'http://data.neuflow.org/data/mnist-th7.tgz'
mnist.path_dataset = 'mnist-th7'
mnist.path_trainset = paths.concat(mnist.path_dataset, 'train.th7')
mnist.path_testset = paths.concat(mnist.path_dataset, 'test.th7')

local function download()
   if not paths.filep(mnist.path_trainset) or not paths.filep(mnist.path_testset) then
      local remote = mnist.path_remote
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
         tensor:select(2, i):add(-mean[1][i])
         if std[1][i] > 0 then
            tensor:select(2, i):mul(1/std[1][i])
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


   function dataset:normalizeGlobal(mean_, std_)
      local data = tensor:narrow(2, 1, dim-1)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   dataset.dim = dim-1

   function dataset:size()
      return nExample
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

function mnist.loadTrainSet(maxLoad, alternative_access_method, offset)
   return loadFlatDataset(mnist.path_trainset, maxLoad, alternative_access_method, offset)
end

function mnist.loadTestSet(maxLoad, alternative_access_method, offset)
   return loadFlatDataset(mnist.path_testset, maxLoad, alternative_access_method, offset)
end


return mnist