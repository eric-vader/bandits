import numpy as np

class MultiArmedBandit(object):
    """
    A Multi-armed Bandit
    """
    def __init__(self, k):
        self.k = k
        self.action_values = np.zeros(k)
        self.optimal = 0

    def reset(self):
        self.action_values = np.zeros(self.k)
        self.optimal = 0

    def pull(self, action):
        return 0, True


class GaussianBandit(MultiArmedBandit):
    """
    Gaussian bandits model the reward of a given arm as normal distribution with
    provided mean and standard deviation.
    """
    def __init__(self, k, mu=0, sigma=1):
        super(GaussianBandit, self).__init__(k)
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.action_values = np.random.normal(self.mu, self.sigma, self.k)
        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        return (np.random.normal(self.action_values[action]),
                action == self.optimal)


class BinomialBandit(MultiArmedBandit):
    """
    The Binomial distribution models the probability of an event occurring with
    p probability k times over N trials i.e. get heads on a p-coin k times on
    N flips.

    In the bandit scenario, this can be used to approximate a discrete user
    rating or "strength" of response to a single event.
    """
    def __init__(self, k, n, p=None, t=None, rng_seed=0):
        super(BinomialBandit, self).__init__(k)
        self.n = n
        self.p = p

        self.rng = np.random.default_rng(rng_seed)

        self._n = n*np.ones(k, dtype=np.int)
        self._p = self.rng.uniform(size=self.k) # np.ones(self.k)/self.n

        self.t = t
        
        self._samples = None
        self._cursor = 0

        self.reset()

    def reset(self):
        if self.p is None:
            self.action_values = self.rng.uniform(size=self.k) 
        else:
            self.action_values = self.p
        
        self._p = self.action_values

        if self.t is not None:
            self._samples = self.rng.binomial(self._n, self._p, size=(self.t,self.k))
            self._cursor = 0

        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        return self.sample[action], action == self.optimal

    @property
    def sample(self):
        if self._samples is None:
            return self.rng.binomial(self._n, self._p, size=(self.k,))
        else:
            val = self._samples[self._cursor]
            self._cursor += 1
            return val


class BernoulliBandit(BinomialBandit):
    """
    The Bernoulli distribution models the probability of a single event
    occurring with p probability i.e. get heads on a single p-coin flip. This is
    the special case of the Binomial distribution where N=1.

    In the bandit scenario, this can be used to approximate a hit or miss event,
    such as if a user clicks on a headline, ad, or recommended product.
    """
    def __init__(self, **kwargs):
        super(BernoulliBandit, self).__init__(n=1, **kwargs)
