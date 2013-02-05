-- Wrap a criterion, for which only a single input comes from the network, into a module.  The criterion can also take a fixed target, specified by setTarget.  DEBUG_MODE, presently disable, potentially outputs diagnostic messages during each operation.

local L1OverL2Cost, parent = torch.class('nn.L1OverL2Cost', 'nn.Module')

function L1OverL2Cost:__init()
   self.output = 0
   self.gradInput = torch.Tensor()

   self.L1_norm_vec = torch.Tensor()
   self.L2_norm_vec = torch.Tensor()
   self.abs_input = torch.Tensor()
   self.abs_input_squared = torch.Tensor()
   self.L1_over_L2 = torch.Tensor()

   self.sign_over_L2 = torch.Tensor()
   self.L1_matrix = torch.Tensor()
   self.L2_matrix = torch.Tensor()
   self.L1_times_input_over_L2_cubed = torch.Tensor()

end

function L1OverL2Cost:updateOutput(input) 
   self.abs_input:resizeAs(input):copy(input):abs()
   self.abs_input_squared:resizeAs(input):copy(self.abs_input):pow(2)
   torch.sum(self.L1_norm_vec, self.abs_input, input:dim()) -- resize automatically; keep in mind that the dimension over which the sum is performed is still present, but has extent of 1
   torch.sum(self.L2_norm_vec, self.abs_input_squared, input:dim())
   if input:dim() == 2 then
      self.L1_norm_vec = self.L1_norm_vec:select(input:dim(),1) -- eliminate the vestigial dimension -- not necessary if we sum
      self.L2_norm_vec = self.L2_norm_vec:select(input:dim(),1)
   end
   self.L2_norm_vec:sqrt()
   self.L1_over_L2:resizeAs(self.L1_norm_vec):cdiv(self.L1_norm_vec, self.L2_norm_vec)
   self.output = torch.sum(self.L1_over_L2)
   
   return self.output
end
    
function L1OverL2Cost:updateGradInput(input, gradOutput) 
   self.sign_over_L2:resizeAs(input)
   self.L1_matrix:resize(self.L1_norm_vec:size(1), input:size(input:dim()))
   self.L2_matrix:resize(self.L1_norm_vec:size(1), input:size(input:dim()))

   self.L1_times_input_over_L2_cubed:resizeAs(input)
   

   self.gradInput:resizeAs(input)

   --print(self.L2_norm_vec:size(), input:size(input:dim()), self.L2_matrix:size())
   torch.ger(self.L2_matrix, self.L2_norm_vec, torch.ones(input:size(input:dim())))
   self.sign_over_L2:copy(input):sign():cdiv(self.L2_matrix)

   torch.ger(self.L1_matrix, self.L1_norm_vec, torch.ones(input:size(input:dim())))
   self.L2_matrix:pow(3)
   self.L1_times_input_over_L2_cubed:copy(input):cmul(self.L1_matrix):cdiv(self.L2_matrix)

   torch.add(self.gradInput, self.sign_over_L2, -1, self.L1_times_input_over_L2_cubed)
   if gradOutput then
      self.gradInput:mul(gradOutput[1])
   end

   return self.gradInput
end 


--[[ -- originally, I was going to implement this using nn modules in the main processing chain, but it's difficult to handle both batched and non-batched inputs, since the L1 and L2 norm need to calculated within but not between batches
feature_extraction_sparsifying_module = nn.Sequential()
--wrap in table; split into L1 and L2 norms, each returned in a table; divide the tables
-- THIS DOESN'T WORK, SINCE WHEN THE INPUTS ARE BATCHED, THE L1 AND L2 NORMS SUM OVER THE BATCHES, WHEREAS THEY SHOULD RETURN VECTORS
feature_extraction_sparsifying_module:add(nn.IdentityTable()) -- wrap the tensor in a table
local L1_over_L2_parallel = nn.ParallelDistributingTable() 
local L1_seq = nn.Sequential()
L1_seq:add(nn.SelectTable{1})
L1_seq:add(nn.L1CriterionModule(nn.L1Cost(), lambdas.ista_L1_lambda)) 
L1_seq:add(nn.IdentityTensor()) -- wrap the scalar in a tensor
local L2_seq = nn.Sequential()
L2_seq:add(nn.SelectTable{1})
L2_seq:add(nn.L1CriterionModule(nn.L2Cost(), 1)) -- scaling by lambda is done in the numerator, rather than the denominator
L2_seq:add(nn.IdentityTensor()) -- wrap the scalar in a tensor
L1_over_L2_parallel:add(L1_seq)
L1_over_L2_parallel:add(L2_seq)
feature_extraction_sparsifying_module:add(L1_over_L2_parallel)
feature_extraction_sparsifying_module:add(nn.CDivTable())
feature_extraction_sparsifying_module:add() -- unwrap scalar from tensor -- CONTINUE HERE!!!
--]]