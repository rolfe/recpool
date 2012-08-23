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
dofile('ParameterizedShrink.lua')
dofile('Ignore.lua')
dofile('ZeroModule.lua')
dofile('L1CriterionModule.lua')
dofile('L2Cost.lua')

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