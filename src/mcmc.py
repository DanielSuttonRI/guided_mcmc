import numpy as np
import seaborn as sns
from inspect import signature
from functools import reduce
from scipy.stats import norm, uniform, bernoulli
from operator import mul

class GuidedMCMC:
    def __init__(self, target):
        self.target = target
        self.n_dim = self._get_target_dimensions()
        self.step = self._guided_step_resolver()
    
    def _sample_target(self, temp, *args):
        return self.target(*args) ** temp

    def _get_target_dimensions(self):
        return len(signature(self.target).parameters)
    
    def _guided_step_resolver(self):
        if self.n_dim == 1:
            step = self._guided_metropolis_step
        else:
            step = self._guided_metropolis_within_gibbs_step
        return step

    def _guided_metropolis_step(self, xt, temp, p):
        z = abs(norm.rvs(0, 1 / (temp + 1), 1))
        y = xt + p*z # Proposal
        alpha = self._sample_target(temp, y) / self._sample_target(temp, xt)

        if uniform.rvs(size=1) < alpha:
            return y, p
        else:
            return xt, -p


    def _guided_metropolis_within_gibbs_step(self, xt, temp, p):
        x_curr = xt.copy()
        z = abs(norm.rvs(0, 1 / temp, self.n_dim))

        for i in range(self.n_dim):
            y = x_curr[i] + p[i] * z[i] # Proposal
            x_proposal = x_curr.copy()
            x_proposal[i] = y
            alpha = self._sample_target(temp, *x_proposal) / self._sample_target(temp, *x_curr)

            if uniform.rvs(size=1) < alpha:
                x_curr[i] = y
            else:
                p[i] = -p[i]

        return x_curr, p

    def metropolis_guided_walk(self, n):
        x = [uniform.rvs(0, 1, 1)] # Starting point
        p = 2 * bernoulli.rvs(p=0.5, size=1) - 1 # Initialise p

        for _ in range(n):
            xt, p = self.step(x[-1], 1, p)
            x.append(xt)
        x = np.array(x)
        return x

    def metropolis_within_gibbs_guided_walk(self, n):
        x = [uniform.rvs(0, 1, self.n_dim)] # Starting point
        p = 2 * bernoulli.rvs(p=0.5, size=self.n_dim) - 1 # Initialise p

        for _ in range(n):
            xt, p = self.step(x[-1], 1, p)
            x.append(xt)
        x = np.array(x)
        return x

    def tempered_transitions(self, i, d, p):
        if i == 0:
            return 1
        elif i == (d - 1):
            return d - 2
        else:
            return i + p

    def simulated_tempering_guided_walk(self, n, temp, pi):
        temp_p = 1 # Trajectory for temperatures
        d = len(temp) # Number of temperatures

        x = uniform.rvs(0, 1, self.n_dim) # Inhomogenous Markov Chain with samples from all temperatures
        w = [x.copy()] # Homogenous Markov Chain with samples from target
        p = 2 * bernoulli.rvs(p=0.5, size=self.n_dim) - 1 # Initialise p

        l = 1 # Count for chain with only target
        i = 0 # Indexing for temperature. 0 is target distribution

        while l < n:
            x, p = self.step(x, temp[0], p)

            if i == 0:
                w.append(x) # Keep sample if in target temperature
                l += 1

            # Attempt temperature change
            j = self.tempered_transitions(i, d, temp_p)
            alpha = self._sample_target(temp[i] * pi[j], *x) / self._sample_target(temp[i] * pi[i], *x)

            if uniform.rvs(size=1) < alpha:
                i = j
            else:
                temp_p = -temp_p
        w = np.array(w)
        return w



    def tempered_transitions_guided_walk(self, n, temp):
        n_heated_temp = len(temp) - 1
        x = [uniform.rvs(0, 1, self.n_dim)] # Homogenous Markov Chain
        xhat = [None for i in range(2 * n_heated_temp + 1)] # Inhomogenous Markov Chain
        p = 2 * bernoulli.rvs(p=0.5, size=self.n_dim) - 1 # Initialise p

        for _ in range(n):
            xhat[0] = x[-1]

            for i in range(n_heated_temp):
                xhat[i + 1], p = self.step(xhat[i], temp[i + 1], p)
            
            # Middle step - Reverse p
            p = -p

            for i in range(n_heated_temp):
                xhat[i + n_heated_temp + 1], p = self.step(xhat[i + n_heated_temp], temp[n_heated_temp - i], p)
            
            # Reverse p again to maintain invariance
            p = -p

            numerator_returns = [
                self._sample_target(
                    temp[n_heated_temp - abs((i + 1) % (2 * n_heated_temp) - n_heated_temp)], 
                    *xhat[i + (i >= n_heated_temp)]
                )
                for i in range(2 * n_heated_temp)
            ]
            denominator_returns = [
                self._sample_target(
                    temp[n_heated_temp - abs(((2 * n_heated_temp) - i) % (2 * n_heated_temp) - n_heated_temp)], 
                    *xhat[i + (i >= n_heated_temp)]
                )
                for i in range(2 * n_heated_temp)
            ]
            alpha = reduce(mul, numerator_returns) / reduce(mul, denominator_returns)

            if uniform.rvs(size=1) < alpha:
                x.append(xhat[-1])
            else:
                x.append(x[-1])
        x = np.array(x)
        return x

    def shortcut_tempered_transitions_guided_walk(self, n, temp, pi, c_up, c_down):
        x = uniform.rvs(0, 1, 1) # Homogenous Markov Chain with samples from all temperatures
        w = [x.copy()] # Homogenous Markov Chain with samples from target
        
        xhat = [None for i in range(5)] # Inhomogenous Markov Chain
        p = 2 * bernoulli.rvs(p=0.5, size=1) - 1 # Initialise p

        l = 1 # Count for chain with only target
        i = 0 # Indexing for temperature. 0 is target distribution

        while l < n:
            # Algorithm for target temperature
            if i == 0:
                x, p = self.step(x, temp[0], p)
                w.append(x)
                l += 1

                # Part tempered transitions step
                if uniform.rvs(size=1) < c_up:
                    xhat[0] = x
                    xhat[1], p = self.step(xhat[0], temp[1], p)
                    xhat[2], p = self.step(xhat[1], temp[2], p)
                    xhat[3], p = self.step(xhat[2], temp[3], p)
                    xhat[4], p = self.step(xhat[3], temp[4], p)

                    numerator = self._sample_target(temp[1], xhat[0]) * self._sample_target(temp[2], xhat[1]) * self._sample_target(temp[3], xhat[2]) * \
                        self._sample_target(temp[4], xhat[3]) * pi[0] * c_down
                    denominator = self._sample_target(temp[0], xhat[0]) * self._sample_target(temp[1], xhat[1]) * self._sample_target(temp[2], xhat[2]) * \
                        self._sample_target(temp[3], xhat[3]) * pi[1] * c_up

                    alpha = numerator / denominator
                    if uniform.rvs(size=1) < alpha:
                        x = xhat[4]
                        i = 1

            elif i == 1:
                x, p = self.step(x, temp[4], p)

                if uniform.rvs(size=1) < c_down:
                    xhat[4] = x
                    xhat[3], p = self.step(xhat[4], temp[4], p)
                    xhat[2], p = self.step(xhat[3], temp[3], p)
                    xhat[1], p = self.step(xhat[2], temp[2], p)
                    xhat[0], p = self.step(xhat[1], temp[1], p)

                    numerator = self._sample_target(temp[1], xhat[0]) * self._sample_target(temp[2], xhat[1]) * self._sample_target(temp[3], xhat[2]) * \
                        self._sample_target(temp[4], xhat[3]) * pi[1] * c_down
                    denominator = self._sample_target(temp[0], xhat[0]) * self._sample_target(temp[1], xhat[1]) * self._sample_target(temp[2], xhat[2]) * \
                        self._sample_target(temp[3], xhat[3]) * pi[0] * c_up

                    alpha = denominator / numerator # Reciprocal
                    if uniform.rvs(size=1) < alpha:
                        x = xhat[0]
                        i = 0
        w = np.array(w)
        return w