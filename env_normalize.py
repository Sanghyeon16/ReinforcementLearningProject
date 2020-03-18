import numpy as np

class RunningStd:
    def __init__(self, coef, shape, epsilon):
        self.coef = coef
        self.var = np.ones(shape)
        self.epsilon = epsilon
        self.count = 0

    def update(self, x):
        batch_var  = np.mean(x**2, 0)
        diff_var  = batch_var - self.var
        w = (1-self.coef)/(1-self.coef**(self.count+1))
        self.var  = self.var + diff_var*w
        self.count += 1

    def normalize(self, x):
        return x/np.sqrt(self.var + self.epsilon)

class RunningMeanStd:
    def __init__(self, coef, shape, epsilon):
        self.coef = coef
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.epsilon = epsilon
        self.count = 0

    def update(self, x):
        batch_mean = np.mean(x, 0)
        batch_var  = np.var(x, 0)
        diff_mean = batch_mean - self.mean
        diff_var  = batch_var - self.var
        w = (1-self.coef)/(1-self.coef**(self.count+1))
        self.mean = self.mean + diff_mean*w
        self.var  = self.var + diff_var*w + diff_mean**2*(w*(1-w))
        self.count += 1

    def normalize(self, x):
        return (x-self.mean)/np.sqrt(self.var + self.epsilon)


class EnvNormalizer():
    def __init__(self, coef, obs_shape, gamma, norm_rew, clipval = 10, epsilon = 1e-4):
        self.norm_rew = norm_rew
        self.clipval = clipval
        self.obs_normalizer = RunningMeanStd(coef, obs_shape, epsilon)
        self.gamma = gamma
        if norm_rew:
            self.rew_normalizer = RunningStd(coef, 1, epsilon)

    def update_obs(self, obs):
        self.obs_normalizer.update(obs)

    def update_rew(self, rew):
        if self.norm_rew:
            self.rew_normalizer.update(rew)

    def normalize_obs(self, obs):
        return np.clip(self.obs_normalizer.normalize(obs), -self.clipval, self.clipval)

    def normalize_rew(self, rew):
        if self.norm_rew:
            return np.clip(self.rew_normalizer.normalize(rew)*(1-self.gamma), -self.clipval, self.clipval)
        else:
            return rew

