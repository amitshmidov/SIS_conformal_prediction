import numpy as np
import torch
from torch import nn as nn
from collections import defaultdict
from functools import lru_cache
import abc
from conformal_prediction.sis_quantile_estimator import SISQuantileEstimator
from mlp import MLP
from torch.nn import functional as F


class ConformalPredictor(abc.ABC):
    def __init__(self, model, calibration_set, labels):
        self.model = model
        self.calibration_set = calibration_set
        self.n = len(calibration_set)
        self.labels = labels
        self.scores = None

    @abc.abstractmethod
    def make_set(self, x, alpha) -> set:
        pass


class StandardPredictionSet(ConformalPredictor):
    def __init__(self, model, calibration_set, labels):
        super(StandardPredictionSet, self).__init__(model, calibration_set, labels)
        self.scores = []
        for x, y in calibration_set:
            self.scores.append(1 - model(x)[y])

    @lru_cache(maxsize=None)
    def score_quantile(self, q):
        return np.quantile(self.scores, q)

    def make_set(self, x, alpha):
        final_set = set()
        model_preds = self.model(x)
        quantile = self.score_quantile((1 - alpha) * (1 + 1 / self.n))
        for y in self.labels:
            if 1 - model_preds[y] <= quantile:
                final_set.add(y)


class LabelCondSet(ConformalPredictor):
    def __init__(self, model, calibration_set, labels):
        super(LabelCondSet, self).__init__(model, calibration_set, labels)
        self.scores = defaultdict(list)
        for x, y in calibration_set:
            self.scores[y].append(1 - model(x)[y])

    @lru_cache(maxsize=None)
    def score_quantile(self, y, q):
        return np.quantile(self.scores[y], q)

    def make_set(self, x, alpha):
        final_set = set()
        model_preds = self.model(x)
        for y in self.labels:
            quantile = self.score_quantile(y, (1 - alpha) * (1 + 1 / self.n))
            if 1 - model_preds[y] <= quantile:
                final_set.add(y)


class RandomizedSet(ConformalPredictor):
    def __init__(self, model, calibration_set, labels):
        super(RandomizedSet, self).__init__(model, calibration_set, labels)
        u_array = np.random.random(self.n)
        self.scores = []
        for (x, y), u_i in zip(calibration_set, u_array):
            model_preds = list(self.model(x))
            sorted_labels = np.flipud(np.argsort(model_preds))
            y_rank = np.where(sorted_labels == y)[0]
            top_labels = sorted_labels[:y_rank]
            score = sum(model_preds[label] for label in top_labels) + model_preds[y] - u_i * model_preds[y]
            self.scores.append(score)

    @lru_cache(maxsize=None)
    def score_quantile(self, q):
        return np.quantile(self.scores, q)

    # l is the minimum number of labels to reach probability mass of tau
    # v is the probability to remove last element
    @staticmethod
    def l_v_function(model_preds_ord, tau):
        assert 0 <= tau <= 1
        if tau == 0:
            return 0, 0
        l = 1
        prob_sum = 0
        for p in model_preds_ord:
            prob_sum += p
            if prob_sum >= tau:
                break
            l += 1

        v = 1 / model_preds_ord[l - 1] * (sum(model_preds_ord[:l]) - tau)
        return l, v

    def make_set(self, x, alpha) -> set:
        quantile = self.score_quantile((1 - alpha) * (1 + 1 / self.n))
        u = np.random.random(1)
        model_preds = self.model(x)
        sorted_labels = np.flipud(np.argsort(model_preds))
        model_preds_ord = model_preds[sorted_labels]
        l, v = self.l_v_function(model_preds_ord, quantile)
        top_labels = sorted_labels[:max(l-1, 1)] if u < v else sorted_labels[:max(l, 1)]
        return set(top_labels)


class QuantilePredictionSet(ConformalPredictor):
    def __init__(self, model, calibration_set, labels, sis_model, estimator: SISQuantileEstimator):
        super(QuantilePredictionSet, self).__init__(model, calibration_set, labels)
        self.estimator = estimator
        self.sis_model = sis_model
        self.scores = defaultdict(list)
        for x, y in calibration_set:
            self.scores[y].append(estimator(x, y) - sis_model(x, y))

    @lru_cache(maxsize=None)
    def score_quantile(self, y, q):
        return np.quantile(self.scores[y], q)

    def make_set(self, x, alpha):
        final_set = set()
        for y in self.labels:
            quantile = self.score_quantile(y, (1 - alpha) * (1 + 1 / self.n))
            if self.sis_model(x, y) >= self.estimator.get_sis_quantile(x, y, alpha) - quantile:
                final_set.add(y)
        return final_set



