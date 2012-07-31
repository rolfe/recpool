local HLinearPsd, parent = torch.class('unsup.HLinearPsd','unsup.Hpsd')

-- input_size    : size of input
-- internal_size : size of code
-- target_size   : size of classification output
-- lambda        : sparsity coefficient
-- beta	         : prediction coefficient
-- params        : optim.FistaLS parameters
function HLinearPsd:__init(input_size, internal_size, target_size, lambda, beta, params)
   
   -- prediction weight
   self.beta = beta

   -- decoder is L1 solution
   --self.decoder = unsup.LinearFistaL1(inputSize, outputSize, lambda, params)
   --local encoder_outputSize = outputSize

   local cmul_code_size = internal_size
   local L1_code_size = internal_size/2
   local L2_code_size = internal_size
   self.decoder = unsup.FactoredSparseCoder(input_size, cmul_code_size, L1_code_size, target_size, lambda, params)
   local encoder_code_size = L2_code_size + L1_code_size -- really, we should directly predict the cmul code, and then infer the L1 and L2 codes from it

   -- encoder
   params = params or {}
   self.params = params
   self.params.encoderType = params.encoderType or 'linear'

   if params.encoderType == 'linear' then
      self.encoder = nn.ScaledGradLinear(input_size,encoder_code_size)
   elseif params.encoderType == 'tanh' then
      self.encoder = nn.Sequential()
      self.encoder:add(nn.ScaledGradLinear(input_size,encoder_code_size))
      self.encoder:add(nn.Tanh())
      self.encoder:add(nn.ScaledGradDiag(encoder_code_size))
   elseif params.encoderType == 'tanh_shrink' then
      self.encoder = nn.Sequential()
      self.encoder:add(nn.ScaledGradLinear(input_size,encoder_code_size))
      self.encoder:add(nn.TanhShrink())
      self.encoder:add(nn.ScaledGradDiag(encoder_code_size))
      -- resetting weights here is useless, since this is done again by PSD:__init()
   elseif params.encoderType == 'tanh_shrink_parallel' then
      self.encoder = nn.Sequential()
      self.encoder:add(nn.Replicate(2))
      
      self.L2_encoder = nn.Sequential()
      self.L2_encoder:add(nn.ScaledGradLinear(input_size,L2_code_size, 2e-3)) -- the final argument is a learning rate scaling factor.  This is necessary since, at least when the L2 layer is disabled, the error in reconstructing these units tends to be orders of magnitude greater than for the L1 units
      self.L2_encoder:add(nn.TanhShrink())
      self.L2_encoder:add(nn.ScaledGradDiag(L2_code_size, 2e-3))

      self.L1_encoder = nn.Sequential()
      self.L1_encoder:add(nn.ScaledGradLinear(input_size,L1_code_size))
      self.L1_encoder:add(nn.TanhShrink())
      self.L1_encoder:add(nn.ScaledGradDiag(L1_code_size))

      self.factored_processor = nn.Parallel(1,1)
      self.factored_processor:add(self.L2_encoder)
      self.factored_processor:add(self.L1_encoder)
      self.encoder:add(self.factored_processor)

      -- resetting weights here is useless, since this is done again by PSD:__init()
   else
      error('params.encoderType unknown " ' .. params.encoderType)
   end

   parent.__init(self, self.encoder, self.decoder, self.beta, self.params)

end



function HLinearPsd:reset(stdv)
   parent.reset(self, stdv)

   if self.params.encoderType == 'tanh_shrink_parallel' then
      --self.encoder:get(2):get(1):get(1):reset(2)
   end
   
   --self.encoder:get(1):reset(1) -- BE CAREFUL WITH THIS!
   --self.encoder:get(1).weight[{{1,200},{}}]:fill(0) -- BE CAREFUL WITH THIS!
   --self.encoder:get(1).bias[{{1,200}}]:fill(1) -- BE CAREFUL WITH THIS!
   --print(self.encoder:get(1).bias) -- debug
end

