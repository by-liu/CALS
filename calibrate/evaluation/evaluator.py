from abc import ABCMeta, abstractmethod


class Evaluator(metaclass=ABCMeta):
    """
    Base class for a evaluator
    """
    @abstractmethod
    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update status given a mini-batch results
        """
        pass

    def curr_score(self):
        """
        Return curr/latest score
        """
        pass

    @abstractmethod
    def mean_score(self):
        """
        Return mean score across all classes/samples
        """
        pass

    @abstractmethod
    def num_samples(self):
        """
        return the total number of evaluated samples
        """
        pass

    @abstractmethod
    def main_metric(self):
        "return the name of the main metric"
        pass

    def class_score(self):
        """
        Return score for different classes
        """
        pass
