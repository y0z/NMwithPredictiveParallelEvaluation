from .utility import array64, CacheError, Function
from collections import OrderedDict
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as sk_kern
import warnings


class NelderMead:
    def __init__(self,
                 f,
                 gamma_s=0.5,
                 delta_ic=-0.5,
                 delta_oc=0.5,
                 delta_r=1.,
                 delta_e=2.,
                 speculative_exec=False,
                 num_montecarlo=100,
                 num_speculative_iter=3,
                 num_parallels=1,
                 max_gp_samples=100):
        self.f = Function(f, num_parallels)
        self.gamma_s = gamma_s
        self.delta_ic = delta_ic
        self.delta_oc = delta_oc
        self.delta_r = delta_r
        self.delta_e = delta_e
        self.num_montecarlo = num_montecarlo
        self.speculative_exec = speculative_exec
        self.num_speculative_iter = num_speculative_iter
        self.num_parallels = num_parallels
        self.max_gp_samples = max_gp_samples

    def optimize(self, initial_simplex, max_iter=100, min_diam=1e-9):
        values = array64(self.f.bulk_call(initial_simplex, False))
        simplex, values = self._order(initial_simplex, values)
        return self._optimize(simplex, values, max_iter, min_diam, 'r')

    def _order(self, simplex, values):
        tmp = sorted(zip(simplex, values, range(len(simplex))), key=lambda x: x[1])
        s = array64([i[0] for i in tmp])
        v = array64([i[1] for i in tmp])
        return s, v

    def _reflect(self, simplex):
        yc = array64(sum(simplex[:-1]) / (len(simplex) - 1))
        yr = array64(yc + self.delta_r * (yc - simplex[-1]))
        return yr

    def _expand(self, simplex):
        yc = array64(sum(simplex[:-1]) / (len(simplex) - 1))
        ye = array64(yc + self.delta_e * (yc - simplex[-1]))
        return ye

    def _outside_contract(self, simplex):
        yc = array64(sum(simplex[:-1]) / (len(simplex) - 1))
        yoc = array64(yc + self.delta_oc * (yc - simplex[-1]))
        return yoc

    def _inside_contract(self, simplex):
        yc = array64(sum(simplex[:-1]) / (len(simplex) - 1))
        yic = array64(yc + self.delta_ic * (yc - simplex[-1]))
        return yic

    def _shrink(self, simplex):
        ys = array64([simplex[0] + self.gamma_s * (x - simplex[0]) for x in simplex])
        return ys

    def _diam(self, simplex):
        d = np.inf
        for i in range(len(simplex) - 1):
            x1 = simplex[i]
            for j in range(i + 1, len(simplex)):
                x2 = simplex[j]
                d = np.minimum(d, np.linalg.norm(x1 - x2))
        return d

    def _optimize(self, simplex, values, max_iter, min_diam, next_operation):
        k = 0
        prev_simplex = simplex.copy()
        while k < max_iter and self._diam(simplex) > min_diam:
            try:
                o = next_operation
                if o == 'r':
                    yr = self._reflect(simplex)
                    fr = self.f(yr, raise_if_not_found=self.speculative_exec)
                    if (values[0] <= fr < values[-2]):
                        next_operation = 'R'
                    elif fr < values[0]:
                        next_operation = 'e'
                    elif fr < values[-1]:
                        next_operation = 'o'
                    elif fr >= values[-1]:
                        next_operation = 'i'
                elif o == 'R':
                    yr = self._reflect(simplex)
                    fr = self.f(yr, raise_if_not_found=self.speculative_exec)
                    simplex[-1] = yr
                    values[-1] = fr
                    next_operation = 'r'
                elif o == 'e':
                    yr = self._reflect(simplex)
                    fr = self.f(yr, raise_if_not_found=self.speculative_exec)
                    ye = self._expand(simplex)
                    fe = self.f(ye, raise_if_not_found=self.speculative_exec)
                    if (fe < fr):
                        next_operation = 'E'
                    else:
                        next_operation = 'R'
                elif o == 'E':
                    ye = self._expand(simplex)
                    fe = self.f(ye, raise_if_not_found=self.speculative_exec)
                    simplex[-1] = ye
                    values[-1] = fe
                    next_operation = 'r'
                elif o == 'o':
                    yr = self._reflect(simplex)
                    fr = self.f(yr, raise_if_not_found=self.speculative_exec)
                    yoc = self._outside_contract(simplex)
                    foc = self.f(yoc, raise_if_not_found=self.speculative_exec)
                    if foc <= fr:
                        next_operation = 'O'
                    else:
                        next_operation = 's'
                elif o == 'O':
                    yoc = self._outside_contract(simplex)
                    foc = self.f(yoc, raise_if_not_found=self.speculative_exec)
                    simplex[-1] = yoc
                    values[-1] = foc
                    next_operation = 'r'
                elif o == 'i':
                    yic = self._inside_contract(simplex)
                    fic = self.f(yic, raise_if_not_found=self.speculative_exec)
                    if fic < values[-1]:
                        next_operation = 'I'
                    else:
                        next_operation = 's'
                elif o == 'I':
                    yic = self._inside_contract(simplex)
                    fic = self.f(yic, raise_if_not_found=self.speculative_exec)
                    simplex[-1] = yic
                    values[-1] = fic
                    next_operation = 'r'
                elif o == 's':
                    simplex = self._shrink(simplex)
                    if self.speculative_exec:
                        values = array64(
                            self.f.bulk_call(simplex, raise_if_not_found=self.speculative_exec))
                    else:
                        values = array64(self.f.bulk_call(simplex, False))
                    next_operation = 'r'
                simplex, values = self._order(simplex, values)
            except CacheError as e:
                if self.speculative_exec:
                    self._speculative_exec(
                        simplex.copy(), values.copy(), o, k, max_iter, min_diam)
            if not np.array_equal(prev_simplex, simplex):
                prev_simplex = simplex.copy()
                k += 1
        simplex, values = self._order(simplex, values)
        return simplex[0], values[0], k

    def _speculative_exec(self, simplex, values, next_operation, k, max_iter, min_diam):
        gp = self._gp(self.f.keys[-self.max_gp_samples:], self.f.values[-self.max_gp_samples:])
        f_ = self._speculative_evaluator(gp)

        candidates = OrderedDict()
        for _ in range(self.num_montecarlo):
            nm = NelderMead(f_, speculative_exec=False, num_parallels=1)
            nm._optimize(simplex,
                         values,
                         min(self.num_speculative_iter, max_iter - k),
                         min_diam,
                         next_operation)
            for c in nm.f.keys:
                if self.f.check_cache(c):
                    continue
                key = tuple(c)
                if not key in candidates:
                    candidates[key] = 0
                candidates[key] += 1
        candidates = [
            array64(x[0]) for x in sorted(candidates.items(), key=lambda x: x[1], reverse=True)]
        self.f.bulk_call(candidates[:self.num_parallels])

    def _speculative_evaluator(self, gp):
        def _impl(y):
            mu, std = gp(y)
            return norm.rvs(mu, std)
        return _impl

    def _gp(self, x, y):
        kernel = sk_kern.Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
        clf = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=0,
            normalize_y=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(x, y)
        def _impl(x):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mu, std = clf.predict(x.reshape(1, -1), return_std=True)
            return mu[0], std[0]
        return _impl

