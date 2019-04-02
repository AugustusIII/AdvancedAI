import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab

s1 = np.load('mc-pg-g_1-dt_500.npy')
s2 = np.load('mc-pg-g_1-dt_250.npy')
s3 = np.load('mc-pg-g_12-dt_500.npy')
s4 = np.load('mc-pg-g_12-dt_250.npy')
t1 = np.arange(s1.size)
t = np.arange(s2.size)
plt.rcParams['figure.figsize'] = [15, 5]
plt.plot(t1, s1, 'r', label='g = 1, dt = 100')
plt.plot(t, s2, 'b', label='g = 1, dt = 50')
plt.plot(t, s3, 'g', label='g = 1.2, dt = 100')
plt.plot(t, s4, 'y', label='g = 1.2, dt = 50')
plt.title('Comparison of Learning over Various Parameters in Mountain Climbing')
plt.xlabel('episodes of training')
plt.ylabel('average return over each episode')
pylab.legend(loc='upper left')
#plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()