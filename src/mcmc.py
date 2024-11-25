import numpy as np
import seaborn as sns
from scipy.stats import norm, uniform, bernoulli, multivariate_normal

sample_target_1D = lambda x, temp: (norm.pdf(x, -3, 1) + norm.pdf(x, 3, 1)) ** temp
sample_target_2D_2 = lambda x, y, temp: multivariate_normal.pdf([x, y], [0, 0], [[1, 0.9], [0.9, 1]]) ** temp
sample_target_2D = lambda x, y, temp: (multivariate_normal.pdf([x, y], [-3, -3], [[1, 0], [0, 1]]) + multivariate_normal.pdf([x, y], [3, 3], [[1, 0], [0, 1]])) ** temp


def guided_metropolis_step(xt, temp, p):
    z = abs(norm.rvs(0, 1 / (temp + 1), 1))
    y = xt + p*z # Proposal
    alpha = sample_target_1D(y, temp) / sample_target_1D(xt, temp)

    if uniform.rvs(size=1) < alpha:
        return y, p
    else:
        return xt, -p

def metropolis_guided_walk(n):
    x = [uniform.rvs(0, 1, 1)] # Starting point
    p = 2 * bernoulli.rvs(p=0.5, size=1) - 1 # Initialise p

    for _ in range(n):
        xt, p = guided_metropolis_step(x[-1], 1, p)
        x.append(xt)
    x = np.array(x)
    return x

def guided_metropolis_within_gibbs_step(x, temp, p):
    # Fixed at 2 dimensions
    x_curr = x.copy()

    # Update x1
    z = abs(norm.rvs(0, 1 / temp, 1))
    y = x_curr[0] + p[0] * z[0] # Proposal
    alpha = sample_target_2D(y, x_curr[1], temp) / sample_target_2D(x_curr[0], x_curr[1], temp)

    if uniform.rvs(size=1) < alpha:
        x_curr[0] = y
    else:
        p[0] = -p[0]

    # Update x2
    z = abs(norm.rvs(0, 1 / temp, 1))
    y = x_curr[1] + p[1] * z[0] # Proposal
    alpha = sample_target_2D(x_curr[0], y, temp) / sample_target_2D(x_curr[0], x_curr[1], temp)

    if uniform.rvs(size=1) < alpha:
        x_curr[1] = y
    else:
        p[1] = -p[1]
    
    return x_curr, p

def metropolis_within_gibbs_guided_walk(n):
    x = [uniform.rvs(0, 1, 2)] # Starting point
    p = 2 * bernoulli.rvs(p=0.5, size=2) - 1 # Initialise p

    for _ in range(n):
        xt, p = guided_metropolis_within_gibbs_step(x[-1], 1, p)
        x.append(xt)
    x = np.array(x)
    return x

def tempered_transitions(i, d, p):
    if i == 0:
        return 1
    elif i == (d - 1):
        return d - 2
    else:
        return i + p

def simulated_tempering_guided_walk(n, temp, pi):
    temp_p = 1 # Trajectory for temperatures
    d = len(temp) # Number of temperatures

    x = uniform.rvs(0, 1, 1) # Inhomogenous Markov Chain with samples from all temperatures
    w = [x.copy()] # Homogenous Markov Chain with samples from target
    p = 2 * bernoulli.rvs(p=0.5, size=1) - 1 # Initialise p

    l = 1 # Count for chain with only target
    i = 0 # Indexing for temperature. 0 is target distribution

    while l < n:
        x, p = guided_metropolis_step(x, temp[0], p)

        if i == 0:
            w.append(x) # Keep sample if in target temperature
            l += 1

        # Attempt temperature change
        j = tempered_transitions(i, d, temp_p)
        alpha = sample_target_1D(x, temp[i] * pi[j]) / sample_target_1D(x, temp[i] * pi[i])

        if uniform.rvs(size=1) < alpha:
            i = j
        else:
            temp_p = -temp_p
    w = np.array(w)
    return w



def tempered_transitions_guided_walk(n, temp):
    x = [uniform.rvs(0, 1, 1)] # Homogenous Markov Chain
    xhat = [None for i in range(9)] # Inhomogenous Markov Chain
    p = 2 * bernoulli.rvs(p=0.5, size=1) - 1 # Initialise p

    for _ in range(n):
        xhat[0] = x[-1]
        xhat[1], p = guided_metropolis_step(xhat[0], temp[1], p)
        xhat[2], p = guided_metropolis_step(xhat[1], temp[2], p)
        xhat[3], p = guided_metropolis_step(xhat[2], temp[3], p)
        xhat[4], p = guided_metropolis_step(xhat[3], temp[4], p)

        # Middle step - Reverse p
        p = -p

        xhat[5], p = guided_metropolis_step(xhat[4], temp[4], p)
        xhat[6], p = guided_metropolis_step(xhat[5], temp[3], p)
        xhat[7], p = guided_metropolis_step(xhat[6], temp[2], p)
        xhat[8], p = guided_metropolis_step(xhat[7], temp[1], p)

        # Reverse p again to maintain invariance
        p = -p

        numerator = sample_target_1D(xhat[0], temp[1]) * sample_target_1D(xhat[1], temp[2]) * sample_target_1D(xhat[2], temp[3]) * \
            sample_target_1D(xhat[3], temp[4]) * sample_target_1D(xhat[5], temp[3]) * sample_target_1D(xhat[6], temp[2]) * \
            sample_target_1D(xhat[7], temp[1]) * sample_target_1D(xhat[8], temp[0])

        denominator = sample_target_1D(xhat[0], temp[0]) * sample_target_1D(xhat[1], temp[1]) * sample_target_1D(xhat[2], temp[2]) * \
            sample_target_1D(xhat[3], temp[3]) * sample_target_1D(xhat[5], temp[4]) * sample_target_1D(xhat[6], temp[3]) * \
            sample_target_1D(xhat[7], temp[2]) * sample_target_1D(xhat[8], temp[1])
        
        alpha = numerator / denominator
        if uniform.rvs(size=1) < alpha:
            x.append(xhat[-1])
        else:
            x.append(x[-1])
    x = np.array(x)
    return x

def shortcut_tempered_transitions_guided_walk(n, temp, pi, c_up, c_down):
    x = uniform.rvs(0, 1, 1) # Homogenous Markov Chain with samples from all temperatures
    w = [x.copy()] # Homogenous Markov Chain with samples from target
    
    xhat = [None for i in range(5)] # Inhomogenous Markov Chain
    p = 2 * bernoulli.rvs(p=0.5, size=1) - 1 # Initialise p

    l = 1 # Count for chain with only target
    i = 0 # Indexing for temperature. 0 is target distribution

    while l < n:
        # Algorithm for target temperature
        if i == 0:
            x, p = guided_metropolis_step(x, temp[0], p)
            w.append(x)
            l += 1

            # Part tempered transitions step
            if uniform.rvs(size=1) < c_up:
                xhat[0] = x
                xhat[1], p = guided_metropolis_step(xhat[0], temp[1], p)
                xhat[2], p = guided_metropolis_step(xhat[1], temp[2], p)
                xhat[3], p = guided_metropolis_step(xhat[2], temp[3], p)
                xhat[4], p = guided_metropolis_step(xhat[3], temp[4], p)

                numerator = sample_target_1D(xhat[0], temp[1]) * sample_target_1D(xhat[1], temp[2]) * sample_target_1D(xhat[2], temp[3]) * \
                    sample_target_1D(xhat[3], temp[4]) * pi[0] * c_down
                denominator = sample_target_1D(xhat[0], temp[0]) * sample_target_1D(xhat[1], temp[1]) * sample_target_1D(xhat[2], temp[2]) * \
                    sample_target_1D(xhat[3], temp[3]) * pi[1] * c_up

                alpha = numerator / denominator
                if uniform.rvs(size=1) < alpha:
                    x = xhat[4]
                    i = 1

        elif i == 1:
            x, p = guided_metropolis_step(x, temp[4], p)

            if uniform.rvs(size=1) < c_down:
                xhat[4] = x
                xhat[3], p = guided_metropolis_step(xhat[4], temp[4], p)
                xhat[2], p = guided_metropolis_step(xhat[3], temp[3], p)
                xhat[1], p = guided_metropolis_step(xhat[2], temp[2], p)
                xhat[0], p = guided_metropolis_step(xhat[1], temp[1], p)

                numerator = sample_target_1D(xhat[0], temp[1]) * sample_target_1D(xhat[1], temp[2]) * sample_target_1D(xhat[2], temp[3]) * \
                    sample_target_1D(xhat[3], temp[4]) * pi[1] * c_down
                denominator = sample_target_1D(xhat[0], temp[0]) * sample_target_1D(xhat[1], temp[1]) * sample_target_1D(xhat[2], temp[2]) * \
                    sample_target_1D(xhat[3], temp[3]) * pi[0] * c_up

                alpha = denominator / numerator # Reciprocal
                if uniform.rvs(size=1) < alpha:
                    x = xhat[0]
                    i = 0
    w = np.array(w)
    return w