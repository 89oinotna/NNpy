"""
    Regularizers allow you to apply penalties.
    These penalties are summed into the loss function that the network optimizes.
"""


class Tikonov:
    """
    Loss = error + penalty to separate eta
    """

    def __init__(self, LAMBDA):
        self.LAMBDA = LAMBDA

    def __call__(self, w):
        """

        :param w: weights
        :return:
        """
        return - self.LAMBDA * w
