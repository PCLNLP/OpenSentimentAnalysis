import math
import numpy as np
from mindspore.common.initializer import Initializer, _calculate_fan_in_and_fan_out


def _assignment(arr, num):
    """Assign the value of `num` to `arr`."""
    if arr.shape == ():
        arr = arr.reshape(1)
        arr[:] = num
        arr = arr.reshape(())
    else:
        if isinstance(num, np.ndarray):
            arr[:] = num[:]
        else:
            arr[:] = num
    return arr


class XavierNormal(Initializer):
    r"""
    Generates an array with values sampled from Xavier normal distribution
    :math:`{U}(-\text{boundary}, \text{boundary})` in order to initialize a tensor, where

    .. math::
        boundary = gain * \sqrt{\frac{2}{n_{in} + n_{out}}}

    where :math:`gain` is an optional scaling factor, :math:`n_{in}` is the number of input units in the weight tensor,
    :math:`n_{out}` is the number of output units in the weight tensor.

    Args:
        gain (float): An optional scaling factor. Default: 1.
    """
    def __init__(self, gain=1):
        super(XavierNormal, self).__init__(gain=gain)
        self.gain = gain

    def _initialize(self, arr):
        n_in, n_out = _calculate_fan_in_and_fan_out(arr.shape)

        boundary = self.gain * math.sqrt(2.0 / (n_in + n_out))
        data = np.random.normal(0, boundary, arr.shape)

        _assignment(arr, data)
