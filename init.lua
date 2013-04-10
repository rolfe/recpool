require 'kex'
-- modifying kex
--torch.include('nn', 'ScaledGradDiag.lua')
dofile('ScaledGradDiag.lua')

require 'nn'
-- modifying nn
--torch.include('nn', 'ScaledGradLinear.lua')
--torch.include('nn', 'CauchyCriterion.lua')
--torch.include('nn', 'InputCMul.lua')
dofile('ScaledGradLinear.lua')
dofile('CauchyCriterion.lua')
dofile('InputCMul.lua')
dofile('ParallelDistributingTable.lua')
dofile('SelectTable.lua')
dofile('CopyTable.lua')
dofile('IdentityTable.lua')
dofile('IdentityTensor.lua')
dofile('SafeIdentity.lua')
dofile('Transpose.lua')
dofile('AddConstant.lua')
dofile('MulConstant.lua')
dofile('AppendConstant.lua')
dofile('LowPass.lua')
dofile('MultiplicativeFilter.lua')
dofile('SafePower.lua')
dofile('SafeLog.lua')
dofile('SafeEntropy.lua')
dofile('FixedShrink.lua')
dofile('SafeCMulTable.lua')
dofile('NormalizeTable.lua')
dofile('NormalizeTensor.lua')
dofile('NormalizeTensorL1.lua')
dofile('PrintModule.lua')
dofile('ParameterizedShrink.lua')
dofile('ParameterizedL1Cost.lua')
dofile('Ignore.lua')
dofile('ZeroModule.lua')
dofile('L1CriterionModule.lua')
dofile('SumCriterionModule.lua')
dofile('L2Cost.lua')
dofile('L1OverL2Cost.lua')
dofile('SoftClassNLLCriterion.lua')
dofile('HingeClassNLLCriterion.lua')
dofile('CauchyCost.lua')
--dofile('DebugSquare.lua')
dofile('ConstrainedLinear.lua')

require 'optim'
dofile('asgd_no_weight_decay.lua')
dofile('sgd_decayed_weight_decay.lua')

require 'unsup'
-- modifying unsup
--torch.include('unsup', 'HLinearPsd.lua')
--torch.include('unsup', 'Hpsd.lua')
dofile('Hpsd.lua')
dofile('HLinearPsd.lua')
dofile('FactoredSparseCoder.lua')
dofile('init_fsc_cost_functions.lua')

dofile('ModifiedJacobian.lua')
--dofile('test.lua')