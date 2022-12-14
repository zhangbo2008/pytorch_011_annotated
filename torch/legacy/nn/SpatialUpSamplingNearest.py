import torch
from .Module import Module

class SpatialUpSamplingNearest(Module):
    """
    Applies a 2D up-sampling over an input image composed of several input planes.

    The upsampling is.ne using the simple nearest neighbor technique.

    The Y and X dimensions are assumed to be the last 2 tensor dimensions.  For
    instance, if the tensor is 4D,: dim 3 is the y dimension and dim 4 is the x.

    owidth  = width*scale_factor
    oheight  = height*scale_factor
    """

    def __init__(self, scale):
        super(SpatialUpSamplingNearest, self).__init__()

        self.scale_factor = scale
        if self.scale_factor < 1:
            raise ValueError('scale_factor must be greater than 1')
        if self.scale_factor % 1 != 0:
            raise ValueError('scale_factor must be integer')

        self.inputSize = torch.LongStorage(4)
        self.outputSize = torch.LongStorage(4)

    def updateOutput(self, input):
        assert input.dim() == 4

        # Copy the input size
        xdim = input.dim() - 1
        ydim = input.dim() - 2
        for i in range(input.dim()):
          self.inputSize[i] = input.size(i)
          self.outputSize[i] = input.size(i)

        self.outputSize[ydim] = self.outputSize[ydim] * self.scale_factor
        self.outputSize[xdim] = self.outputSize[xdim] * self.scale_factor
        self.output.resize_(self.outputSize)
        self._backend.SpatialUpSamplingNearest_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.scale_factor
        )
        return self.output


    def updateGradInput(self, input, gradOutput):
        self.gradInput.resizeAs_(input)
        self._backend.SpatialUpSamplingNearest_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.scale_factor
        )
        return self.gradInput

