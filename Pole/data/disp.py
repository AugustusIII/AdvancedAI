import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab

s1 = np.load('pole-pg-g_1-dt_100.npy')
s2 = np.load('pole-pg-g_1-dt_50.npy')
s3 = np.load('pole-pg-g_2-dt_100.npy')
s4 = np.load('pole-pg-g_2-dt_50.npy')
t = np.arange(s1.size)
plt.rcParams['figure.figsize'] = [15, 5]
plt.plot(t, s1, 'r', label='g = 1, dt = 100')
plt.plot(t, s2, 'b', label='g = 1, dt = 50')
plt.plot(t, s3, 'g', label='g = 2, dt = 100')
plt.plot(t, s4, 'y', label='g = 2, dt = 50')
plt.title('Comparison of Learning over Various Parameters in Pole Balancing')
plt.xlabel('episodes of training')
plt.ylabel('average return over each episode')
pylab.legend(loc='upper left')
#plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()