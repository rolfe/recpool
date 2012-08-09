-- modifying kex
--torch.include('nn', 'ScaledGradDiag.lua')
dofile('ScaledGradDiag.lua')

-- modifying nn
--torch.include('nn', 'ScaledGradLinear.lua')
--torch.include('nn', 'CauchyCriterion.lua')
--torch.include('nn', 'InputCMul.lua')
dofile('ScaledGradLinear.lua')
dofile('CauchyCriterion.lua')
dofile('InputCMul.lua')



-- modifying unsup
--torch.include('unsup', 'HLinearPsd.lua')
--torch.include('unsup', 'Hpsd.lua')
dofile('Hpsd.lua')
dofile('HLinearPsd.lua')
dofile('FactoredSparseCoder.lua')
dofile('init_fsc_cost_functions.lua')

dofile('ModifiedJacobian.lua')
dofile('test.lua')