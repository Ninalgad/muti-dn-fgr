# %%writefile /content/src/ladp.py
import numpy as np
from utils import one_hot2dist


class LADP:
    def __init__(self, gamma=0.95, sd=1):
        self.gamma = gamma  # [0, 1]
        self.sd = sd

    def apply(self, image, mask):
        dist = one_hot2dist(mask)

        c = np.random.beta(1, 1)  # [0,1] creat distance
        c = (c - 0.5) * 2  # [-1.1]
        m = np.min(dist)
        if c > 0:
            lam = c * m / 2  # λl = -1/2|min(dis_array)|
        else:
            lam = c * m
        mask = (dist < lam).astype('float32')  # creat M

        s0, s1 = 0.8, 1.5
        y = np.random.normal(scale=self.sd, size=image.shape)

        z = y.copy()
        z = z * (1 - mask) * s0 + z * mask * s1
        noisy_image = pow(self.gamma, .5) * image.copy() + pow(1 - self.gamma, .5) * z

        return noisy_image, y
