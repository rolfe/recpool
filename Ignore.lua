local Ignore, parent = torch.class('nn.Ignore', 'nn.Module')

-- Throw away the input.  Used in conjunction with a Criteria in a ParallelDistributingTable.  A Criterion returns the loss value, but this should not be passed on to the rest of the network; a Criterion requires no gradOutput to updateGradInput.  Since ParallelDistributingTable sets the elements of the output array according to the outputs of each stream, and setting an array value to nil makes it as if it isn't there, this just eliminates the output from the array (and reduces the size of the array by one).  Be careful when the ignored element is in the middle of the array, since the length operator and the ipairs iterator stop at the first nil entry.  

function Ignore:updateOutput(input)
   self.output = nil
   return self.output
end


function Ignore:updateGradInput(input, gradOutput)
   self.gradInput = nil
   return self.gradInput
end
