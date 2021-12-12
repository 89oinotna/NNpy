import numpy as np


def metric(type_init, **kwargs):
    init = {
        'mee': MEE,
        'simple_class':lambda **kwargs: SimpleClassification()
    }
    matrix = init[type_init](**kwargs)
    return matrix


class Metric:
    def __call__(self, label, output):
        pass


class RegressionMetric(Metric):
    def __call__(self, label, output):
        return self.error(label, output)

    def error(self, label, output):
        pass


class ClassificationMetric(Metric):
    def __call__(self, label, output):
        return self.accuracy(label, output)

    def accuracy(self, label, output):
        pass


class MEE(RegressionMetric):
    def error(self, label, output):
        return np.mean(np.sqrt(np.sum(np.square(label - output), axis=1)))


class SimpleClassification(ClassificationMetric):
    def accuracy(self, label: np.ndarray, output: np.ndarray):
        output = np.around(output)
        return np.sum(([1 if out == lab else 0 for lab, out in zip(label, output)])) / len(label)
