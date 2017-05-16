#-*- coding: utf-8 -*-
import random

import numpy as np
import matplotlib.pyplot as plt


s = np.random.normal(3, 0.1, 1000000)
plt.subplot(221)
plt.hist(s, 30, normed=True)


a = []
for i in range(1000000):
    a.append(random.uniform(10.0, 15.0))
a.sort()
print a[500000 - 5:500000 + 100]
plt.subplot(222)
plt.hist(a, 30, normed=True)

b = a*s
plt.subplot(223)
plt.hist(b, 30, normed=True)
print b.shape
print len(b)
b.sort()
print b[500000-5:500000+100]
plt.show()