"""
    Regularizers allow you to apply penalties.
    These penalties are summed into the loss function that the network optimizes.
"""


class WeightRegularizer:
    def __call__(self, *args, **kwargs):
        pass


class Tikonov(WeightRegularizer):

    def __init__(self, LAMBDA):
        self.LAMBDA = LAMBDA

    def __call__(self, w):
        """
        :param w: weights
        :return:
        """
        return w - 2 * self.LAMBDA * w
