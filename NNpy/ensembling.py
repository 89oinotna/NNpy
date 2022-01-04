import numpy as np
from random import randint
from network import NeuralNetwork


class Bagging:

    def __init__(self, sample_size, max_epochs_tr=1500, bootstrap=True, models=None):
        if models is None:
            models = []
        self.sample_size = sample_size
        self.models = models
        self.bootstrap = bootstrap
        self.max_epochs_tr = max_epochs_tr

    def _generate_sample(self, data, labels):
        """performs bootstrap with resampling

        Args:
            data (list): data over which perform the bootstrap
            labels (list): list of labels over which perform the bootstrap
        Returns:
            list
        """
        data_sample = []
        labels_sample = []
        for i in range(0, self.sample_size):
            x = randint(0, self.sample_size)
            data_sample.append(data[x])
            labels_sample.append(labels[x])

        return data_sample, labels_sample

    def add_neural_network(self, model):
        """adds a Neural Network to the ensemble

        Args:
            model (NeuralNetwork): neural network to add to the ensemble
        """
        self.models.append(model)

    def fit(self, tr_data, tr_label, validation_data=None, validation_label=None, test_set=None, max_epochs=0):
        """performs training of the models

        Args:
            tr_data (np.array): data used for training
            tr_label (np.array): labels of training set
            validation_set (np.array, optional): list used for validation. Defaults to None.

        Returns:
            Report: report that contains information about the training 
        """
        training_results = []
        # training
        for i in range(0, len(self.models)):
            # if bootstrap is true then perform _generate_sample(Bootstrap with resampling), otherwise we simply use
            # the original training set
            results_model_i = self.models[i].fit(tr_data, tr_label, validation_data, validation_label, early_stopping=True, return_outputs=True) \
                if self.bootstrap else self.models[i].fit(tr_data, tr_label, validation_data, validation_label, early_stopping=True, return_outputs=True)

            training_results.append(results_model_i)

        # fill values
        for i, (tr_metric, tr_loss), (vl_metric, vl_loss), (tr_outputs, vl_outputs) in training_results:
            ll =len(tr_metric)
            if ll < max_epochs:
                diff = (max_epochs - ll)
                tr_metric.extend([tr_metric[-1]] * diff)
                vl_metric.extend([vl_metric[-1]] * diff)
                tr_loss.extend([tr_loss[-1]] * diff)
                vl_loss.extend([vl_loss[-1]] * diff)
                tr_outputs.extend([tr_outputs[-1]] * diff)
                vl_outputs.extend([vl_outputs[-1]] * diff)



        # calculate the mean of every result
        final_training_error = np.mean([training_result[0][1][-1] for training_result in training_results], axis=0)
        final_validation_error = np.mean([training_result[1][1][-1] for training_result in training_results], axis=0)
        final_training_accuracy = np.mean(
            [training_result[0][0][-1] for training_result in training_results], axis=0)
        final_validation_accuracy = np.mean(
            [training_result[1][0][-1] for training_result in training_results], axis=0)

        #TODO: the same for test and print training accuracy

        return final_training_error, final_validation_error, final_training_accuracy, final_validation_accuracy

    def predict(self, sample):
        """
            This method implements the predict operation to make prediction
            on the output of a sample

            Parameters:
                sample: represents the feature space of a sample

            Return: the predicted target over the sample
        """
        return np.mean([model.feed_forward(sample) for model in self.models], axis=0)

