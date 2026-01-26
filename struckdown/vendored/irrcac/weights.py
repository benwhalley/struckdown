"""Vendored from irrCAC - weight schemes for agreement coefficients."""

import numpy as np
import pandas as pd


class Weights:
    """Methods for computing weights for a set of categories."""

    def __init__(self, categories):
        if isinstance(categories, list):
            self.q = len(categories)
        elif isinstance(categories, (np.ndarray, pd.DataFrame)):
            self.q = categories.shape[-1]
        else:
            raise ValueError(
                "Valid input for `categories` is one of "
                "list, numpy array, or pandas data frame."
            )
        if all(isinstance(n, (int, float)) for n in categories):
            self.categ_vec = sorted(categories)
        else:
            self.categ_vec = list(range(1, len(categories) + 1))
        self.xmin, self.xmax = min(self.categ_vec), max(self.categ_vec)

    def __getitem__(self, item):
        if item == "bipolar":
            return self.bipolar()
        elif item == "circular":
            return self.circular()
        elif item == "identity":
            return self.identity()
        elif item == "linear":
            return self.linear()
        elif item == "ordinal":
            return self.ordinal()
        elif item == "quadratic":
            return self.quadratic()
        elif item == "radical":
            return self.radical()
        elif item == "ratio":
            return self.ratio()
        else:
            raise ValueError(f'"{item} is an unknown type of weights.')

    def __str__(self):
        return f"Weights for {self.q} categories."

    def bipolar(self):
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                if k != el:
                    weights[k][el] = pow(self.categ_vec[k] - self.categ_vec[el], 2) / (
                        (self.categ_vec[k] + self.categ_vec[el] - 2 * self.xmin)
                        * (2 * self.xmax - self.categ_vec[k] - self.categ_vec[el])
                    )
                else:
                    weights[k][el] = 0
        weights = 1 - weights / np.max(weights)
        return weights

    def circular(self):
        weights = np.eye(self.q)
        U = self.xmax - self.xmin + 1
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = pow(
                    np.sin(np.pi * (self.categ_vec[k] - self.categ_vec[el]) / U), 2
                )
        weights = 1 - weights / np.max(weights)
        return weights

    def identity(self):
        weights = np.eye(self.q)
        return weights

    def linear(self):
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = 1 - abs(self.categ_vec[k] - self.categ_vec[el]) / abs(
                    self.xmax - self.xmin
                )
        return weights

    def ordinal(self):
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                nkl = max(k, el) - min(k, el) + 1
                weights[k][el] = nkl * (nkl - 1) / 2
        weights = 1 - weights / np.max(weights)
        return weights

    def quadratic(self):
        weights = np.eye(self.q)
        diff = self.xmax - self.xmin
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = 1 - pow(
                    (self.categ_vec[k] - self.categ_vec[el]) / diff, 2
                )
        return weights

    def radical(self):
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = 1 - np.sqrt(
                    abs(self.categ_vec[k] - self.categ_vec[el])
                ) / np.sqrt(abs(self.xmax - self.xmin))
        return weights

    def ratio(self):
        if 0 in self.categ_vec:
            raise ValueError(
                "You have 0 as a category. Please do not use"
                " 0 as a category because it produce a"
                " division by 0."
            )
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = 1 - (
                    pow(
                        (self.categ_vec[k] - self.categ_vec[el])
                        / (self.categ_vec[k] + self.categ_vec[el]),
                        2,
                    )
                ) / pow((self.xmax - self.xmin) / (self.xmax + self.xmin), 2)
        return weights
