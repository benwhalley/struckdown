"""Vendored from irrCAC - Chance-corrected Agreement Coefficients."""

from copy import deepcopy

import numpy as np
from scipy import stats

from .weights import Weights


class CAC:
    """Chance-corrected Agreement Coefficients (CAC)

    Calculates various chance-corrected agreement coefficients (CAC) among 2 or
    more raters. Coefficients include Fleiss' kappa, Gwet's AC1/AC2, and
    Krippendorff's Alpha.
    """

    def __init__(
        self,
        ratings,
        weights="identity",
        categories=None,
        confidence_level=0.95,
        N=np.inf,
        digits=5,
    ):
        weights_choices = (
            "identity",
            "quadratic",
            "ordinal",
            "linear",
            "radical",
            "ratio",
            "circular",
            "bipolar",
        )
        if not 0.9 <= confidence_level <= 0.99:
            raise ValueError("Please provide a value in range [0.90, 0.99].")
        self.confidence_level = confidence_level

        self.ratings = ratings.dropna(how="all")
        self.ratings.replace(to_replace="", value=np.nan, inplace=True)
        self.n, self.r = self.ratings.shape
        self.f = self.n / N
        if categories is None:
            self.categories = sorted(self.ratings.stack().unique().tolist())
        else:
            self.categories = categories
        self.q = len(self.categories)
        if isinstance(weights, str):
            if weights not in weights_choices:
                raise ValueError(f"weights values can be any of {weights_choices}")
            self.weights_name = weights
            weights_functions = Weights(self.categories)
            self.weights_mat = weights_functions[self.weights_name]
        else:
            self.weights_name = "Custom Weights"
            self.weights_mat = np.asarray(weights)
            rows, cols = self.weights_mat.shape
            if not (rows == self.q and cols == self.q):
                raise ValueError(
                    f"Expected weights matrix shape is {self.q}x{self.q}. "
                    f"Given size is {rows}x{cols}."
                )
        self.digits = digits
        self.coefficient_value = 0
        self.coefficient_name = None
        self.confidence_interval = (0, 0)
        self.p_value = 0
        self.z = 0
        self.se = 0
        self.pa = 0
        self.pe = 0
        self.agreement = {
            "est": dict(
                coefficient_value=0,
                coefficient_name=None,
                confidence_interval=(0, 0),
                p_value=0,
                z=0,
                se=0,
                pa=0,
                pe=0,
            ),
            "weights": self.weights_mat,
            "categories": self.categories,
        }

    def gwet(self):
        """Gwet's AC1/AC2 coefficient."""
        agree_mat = np.zeros(shape=(self.n, self.q))
        for k in range(self.q):
            agree_mat[:, k] = self.ratings[self.ratings == self.categories[k]].count(
                axis=1
            )
        agree_mat_w = np.transpose(np.matmul(self.weights_mat, agree_mat.T))
        ri_vec = agree_mat.sum(axis=1)
        sum_q = (agree_mat * (agree_mat_w - 1)).sum(axis=1)
        n2more = sum(ri_vec >= 2)
        pa = sum(sum_q[ri_vec >= 2] / (ri_vec * (ri_vec - 1))[ri_vec >= 2]) / n2more
        pi_vec = (
            np.repeat(1 / self.n, self.n).reshape(self.n, 1)
            * (agree_mat / np.repeat(ri_vec, self.q).reshape(self.n, self.q))
        ).T.sum(axis=1)
        weights_mat_sum = np.sum(np.sum(self.weights_mat))
        if self.q >= 2:
            pe = weights_mat_sum * sum(pi_vec * (1 - pi_vec)) / (self.q * (self.q - 1))
        else:
            pe = 1 - 1e-15
        ac1 = (pa - pe) / (1 - pe)
        den_ivec = ri_vec * (ri_vec - 1)
        den_ivec = den_ivec - (den_ivec == 0)
        pa_ivec = sum_q / den_ivec
        pe_r2 = pe * (ri_vec >= 2)
        ac1_ivec = (self.n / n2more) * (pa_ivec - pe_r2) / (1 - pe)
        pe_ivec = (
            (weights_mat_sum / (self.q * (self.q - 1)))
            * np.matmul(agree_mat, (1 - pi_vec))
            / ri_vec
        )
        ac1_ivec_x = ac1_ivec - 2 * (1 - ac1) * (pe_ivec - pe) / (1 - pe)
        var_ac1 = (1 - self.f) / (self.n * (self.n - 1)) * sum((ac1_ivec_x - ac1) ** 2)
        stderr = np.sqrt(var_ac1)
        if stderr == 0.0:
            stderr = 1e-15
        p_value = 2 * (1 - stats.t.cdf(abs(ac1 / stderr), self.n - 1))
        lcb, ucb = stats.t.interval(
            self.confidence_level, df=self.n - 1, scale=stderr, loc=ac1
        )
        ucb = min(1, ucb)

        if weights_mat_sum == self.q:
            coeff_name = "AC1"
        else:
            coeff_name = "AC2"

        self.coefficient_value = np.round(ac1, self.digits)
        self.coefficient_name = coeff_name
        self.confidence_interval = (round(lcb, self.digits), round(ucb, self.digits))
        self.p_value = p_value
        self.z = round(ac1 / stderr, self.digits)
        self.se = round(stderr, self.digits)
        self.pa = round(pa, self.digits)
        self.pe = round(pe, self.digits)
        self.agreement["est"].update(
            dict(
                coefficient_name=coeff_name,
                pa=self.pa,
                pe=self.pe,
                se=self.se,
                z=self.z,
                coefficient_value=self.coefficient_value,
                confidence_interval=self.confidence_interval,
                p_value=self.p_value,
            )
        )
        return deepcopy(self.agreement)

    def fleiss(self):
        """Fleiss' generalized kappa coefficient."""
        agree_mat = np.zeros(shape=(self.n, self.q))
        for c in range(self.q):
            agree_mat[:, c] = self.ratings[self.ratings == self.categories[c]].count(
                axis=1
            )
        agree_mat_w = np.transpose(np.matmul(self.weights_mat, agree_mat.T))
        ri_vec = agree_mat.sum(axis=1)
        sum_q = (agree_mat * (agree_mat_w - 1)).sum(axis=1)
        n2more = sum(ri_vec >= 2)
        pa = float(
            sum(sum_q[ri_vec >= 2] / (ri_vec * (ri_vec - 1))[ri_vec >= 2]) / n2more
        )
        pi_vec = (
            np.repeat(1 / self.n, self.n).reshape(self.n, 1)
            * (agree_mat / np.repeat(ri_vec, self.q).reshape(self.n, self.q))
        ).T.sum(axis=1)
        pe = float(
            np.sum(
                self.weights_mat
                * (pi_vec.reshape(self.q, 1) * pi_vec.reshape(1, self.q))
            )
        )
        fleiss_kappa = (pa - pe) / (1 - pe)
        den_ivec = ri_vec * (ri_vec - 1)
        den_ivec = den_ivec - (den_ivec == 0)
        pa_ivec = sum_q / den_ivec
        pe_r2 = pe * (ri_vec >= 2)
        kappa_ivec = (self.n / n2more) * (pa_ivec - pe_r2) / (1 - pe)
        pi_vec_wk_ = np.matmul(self.weights_mat, pi_vec)
        pi_vec_w_k = np.matmul(self.weights_mat.T, pi_vec)
        pi_vec_w = (pi_vec_wk_ + pi_vec_w_k) / 2
        pe_ivec = np.matmul(agree_mat, pi_vec_w) / ri_vec
        kappa_ivec_x = kappa_ivec - 2 * (1 - fleiss_kappa) * (pe_ivec - pe) / (1 - pe)
        var_fleiss = (
            (1 - self.f)
            / (self.n * (self.n - 1))
            * sum((kappa_ivec_x - fleiss_kappa) ** 2)
        )
        stderr = np.sqrt(var_fleiss)
        if stderr == 0.0:
            stderr = 1e-15
        p_value = float(2 * (1 - stats.t.cdf(abs(fleiss_kappa / stderr), self.n - 1)))
        lcb, ucb = stats.t.interval(
            self.confidence_level, df=self.n - 1, scale=stderr, loc=fleiss_kappa
        )
        ucb = min(1, ucb)

        self.coefficient_value = round(fleiss_kappa, self.digits)
        self.coefficient_name = "Fleiss' kappa"
        self.confidence_interval = (round(lcb, self.digits), round(ucb, self.digits))
        self.p_value = p_value
        self.z = round(fleiss_kappa / stderr, self.digits)
        self.se = round(stderr, self.digits)
        self.pa = round(pa, self.digits)
        self.pe = round(pe, self.digits)
        self.agreement["est"].update(
            dict(
                coefficient_name=self.coefficient_name,
                pa=self.pa,
                pe=self.pe,
                se=self.se,
                z=self.z,
                coefficient_value=self.coefficient_value,
                confidence_interval=self.confidence_interval,
                p_value=self.p_value,
            )
        )
        return deepcopy(self.agreement)

    def krippendorff(self):
        """Krippendorff's alpha coefficient."""
        agree_mat = np.zeros(shape=(self.n, self.q))
        for k in range(self.q):
            agree_mat[:, k] = self.ratings[self.ratings == self.categories[k]].count(
                axis=1
            )
        agree_mat_w = np.transpose(np.matmul(self.weights_mat, agree_mat.T))
        ri_vec = agree_mat.sum(axis=1)
        agree_mat = agree_mat[ri_vec >= 2]
        agree_mat_w = agree_mat_w[ri_vec >= 2]
        ri_vec = ri_vec[ri_vec >= 2]
        ri_mean = np.mean(ri_vec)
        n = agree_mat.shape[0]
        epsi = 1 / np.sum(ri_vec)
        sum_q = (agree_mat * (agree_mat_w - 1)).sum(axis=1)
        paprime = np.sum(sum_q / (ri_mean * (ri_vec - 1))) / n
        pa = float((1 - epsi) * paprime + epsi)
        pi_vec = np.matmul(np.repeat(1 / n, n).reshape(1, n), agree_mat / ri_mean).T
        pe = float(np.sum(self.weights_mat * np.matmul(pi_vec, pi_vec.T)))
        krippen_alpha = (pa - pe) / (1 - pe)
        krippen_alpha_est = np.round(krippen_alpha, self.digits)
        krippen_alpha_prime = (paprime - pe) / (1 - pe)
        pa_ivec = (
            sum_q / (ri_mean * (ri_vec - 1)) - pa * (ri_vec - ri_mean) / ri_mean
        ).reshape(-1, 1)
        krippen_ivec = (pa_ivec - pe) / (1 - pe)
        pi_vec_wk_ = np.matmul(self.weights_mat, pi_vec)
        pi_vec_w_k = np.matmul(self.weights_mat.T, pi_vec)
        pi_vec_w = (pi_vec_wk_ + pi_vec_w_k) / 2
        pe_ivec = np.matmul(agree_mat, pi_vec_w) / ri_mean - (
            pe * (ri_vec - ri_mean) / ri_mean
        ).reshape(-1, 1)
        krippen_ivec_x = krippen_ivec - 2 * (1 - krippen_alpha_prime) * (
            pe_ivec - pe
        ) / (1 - pe)
        var_krippen = (
            (1 - self.f)
            / (n * (n - 1))
            * sum((krippen_ivec_x - krippen_alpha_prime) ** 2)
        )
        stderr = np.sqrt(float(var_krippen.item()))
        if stderr == 0.0:
            stderr = 1e-15
        p_value = 2 * (1 - stats.t.cdf(abs(krippen_alpha / stderr), n - 1))
        lcb, ucb = stats.t.interval(
            self.confidence_level, df=n - 1, scale=stderr, loc=krippen_alpha
        )
        ucb = min(1, ucb)
        self.coefficient_value = round(krippen_alpha_est, self.digits)
        self.coefficient_name = "Krippendorff's Alpha"
        self.confidence_interval = (round(lcb, self.digits), round(ucb, self.digits))
        self.p_value = p_value
        self.z = round(krippen_alpha_est / stderr, self.digits)
        self.se = round(stderr, self.digits)
        self.pa = round(pa, self.digits)
        self.pe = round(pe, self.digits)
        self.agreement["est"].update(
            dict(
                coefficient_name=self.coefficient_name,
                pa=self.pa,
                pe=self.pe,
                se=self.se,
                z=self.z,
                coefficient_value=self.coefficient_value,
                confidence_interval=self.confidence_interval,
                p_value=self.p_value,
            )
        )
        return deepcopy(self.agreement)
