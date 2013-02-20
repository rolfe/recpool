require 'torch'
require 'paths'

spiral_2d = {}

function spiral_2d:train_set_size()
   return 50000
end

function spiral_2d:validation_set_size()
   return 10000
end

function spiral_2d:test_set_size()
   return 10000
end


-- alternative_access_method specifies the format with which the dataset should be returned 
-- if project_onto_sphere is true, then use spherical coordinates to project the 2d dataset onto the surface of a sphere of radius 1
local function loadFlatDataset(maxLoad, alternative_access_method, project_onto_sphere)
   local dataset = {}
   local dim = 3 -- these variables are part of the closure of the access functions defined below
   local internal_dim = 2
   if project_onto_sphere then
      dim = 4
      function dataset:project_onto_sphere(data_row, current_val, radius)
	 data_row[1] = radius * math.sin(current_val[1]) * math.cos(current_val[2])
	 data_row[2] = radius * math.sin(current_val[1]) * math.sin(current_val[2])
	 data_row[3] = radius * math.cos(current_val[1])
      end
   end

   local nExample = maxLoad
   local tensor = torch.Tensor(nExample, dim)
   local min_vals = torch.Tensor(internal_dim)
   local max_vals = torch.Tensor(internal_dim)
   local current_val = torch.Tensor(internal_dim)

   for i = 1,maxLoad do
      local data_row = tensor:select(1,i)
      data_row[dim] = math.random(2) - 1 -- categories are integers starting from 0; these are shifted up by 1 by the access method
      local angle = math.random() * math.pi -- the dataset consists of two interleaved, shifted and flipped, semicircles; points are generated based upon the angle around the semicircle
      
      current_val[1] = (math.cos(angle) + (data_row[dim] - 0.5)*1) / ((project_onto_sphere and 3) or 1) + ((project_onto_sphere and 0.75) or 0) --2
      current_val[2] = (2*(data_row[dim] - 0.5)*math.sin(angle) - 0.75*(data_row[dim] - 0.5)*1) / ((project_onto_sphere and 1.5) or 1) + ((project_onto_sphere and 0.5) or 0)

      if project_onto_sphere then
	 if (current_val:max() > math.pi) or (current_val:min() < -math.pi) then
	    error('When projecting onto a sphere, the original 2d coordinate must be bounded within (-pi,pi)x(-pi,pi): ' .. current_val[1] .. ', ' .. current_val[2] .. ' violates these bounds')
	 end
	 local radius = 1
	 dataset:project_onto_sphere(data_row, current_val, radius)
      else
	 data_row:narrow(1,1,2):copy(current_val)
      end

      -- we need to know the range of the 2d coordinates, even when we project into three dimensions; the safest way to do this is to extract it empirically when creating the dataset
      if i == 1 then 
	 min_vals:copy(current_val)
	 max_vals:copy(current_val)
      else
	 for j = 1,internal_dim do
	    min_vals[j] = math.min(min_vals[j], current_val[j])
	    max_vals[j] = math.max(max_vals[j], current_val[j])
	 end
      end
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

   function dataset:normalizeL2(desired_norm)
      --don't actually normalize this dataset!  Otherwise, it just falls on a 2d circle!
   end

   function dataset:normalizeL2orig(desired_norm) 
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

   function dataset:useGrayscale() -- this need not do anything, since SPIRAL_2D is already in grayscale
   end


   function dataset:normalizeGlobal(mean_, std_)
      local data = tensor:narrow(2, 1, dim-1)
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
      return dim-1
   end

   function dataset:labelSize()
      return 1
   end

   function dataset:nClass()
      return 2
   end
   
   function dataset:max(index)
      -- dimensions of tensor are: nExample, dim
      --local max_val = tensor:select(2,index):max()
      --local min_val = tensor:select(2,index):min()
      local min_val, max_val = min_vals[index], max_vals[index]
      return max_val + 0.5 * (max_val - min_val)
   end

   function dataset:min(index)
      --local max_val = tensor:select(2,index):max()
      --local min_val = tensor:select(2,index):min()
      local min_val, max_val = min_vals[index], max_vals[index]
      return min_val - 0.5 * (max_val - min_val)
   end


   if alternative_access_method == 'recpool_net' then
      dataset.data = {}
      dataset.labels = {}
      setmetatable(dataset.data, {__index = function(self, index)
				     if type(index) == 'number' then
					return tensor[index]:narrow(1, 1, dim-1)
				     elseif type(index) == 'table' then
					if (#index ~= 2) or (index[1] < 0) or (index[1] > 1) or (index[2] < 0) or (index[2] > 1) then
					   error('index must consist of two numbers between 0 and 1, indicating the fractional position within the range of the 2D dataset')
					end
					local output = torch.Tensor(dataset:dataSize())
					local internal = torch.Tensor(internal_dim)
					
					for i = 1,internal_dim do
					   internal[i] = dataset:min(i) + index[i] * (dataset:max(i) - dataset:min(i))
					end
					
					if project_onto_sphere then
					   dataset:project_onto_sphere(output, internal, 1)
					else
					   output:copy(internal)
					end
					return output
				     else
					error('index must be either an integer, or a table of two numbers between 0 and 1')
				     end
      end})
      
      -- the correct class isn't defined for arbitrary points in the domain of the dataset, so return a placeholder value that is guaranteed to be allowable.  The scaling of the classification loss *must* be set to zero for the loss to be meaningful
      setmetatable(dataset.labels, {__index = function(self, index)
				       if type(index) == 'number' then
					  return tensor[index][dim]+1
				       elseif type(index) == 'table' then
					  return math.random(dataset:nClass())
				       else
					  error('index must be either an integer, or a table of two numbers between 0 and 1')
				       end
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
				       --print('spiral_2d access will return ', label_vector)
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

function spiral_2d.loadTrainSet(maxLoad, alternative_access_method)
   return loadFlatDataset(maxLoad, alternative_access_method)
end

function spiral_2d.loadTestSet(maxLoad, alternative_access_method)
   return loadFlatDataset(maxLoad, alternative_access_method)
end

function spiral_2d.loadDataSet(params)
   return loadFlatDataset(params.maxLoad, params.alternative_access_method, true)
end


