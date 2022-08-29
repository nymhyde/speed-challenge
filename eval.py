import numpy as np
import matplotlib.pyplot as plt

test = np.loadtxt('test.txt')

# plot and save
plt.figure(1)
plt.plot(test, c='black', label='predicted test speed')
plt.xlabel('Frame Number')
plt.ylabel('Speed [m/s]')
plt.legend(loc='best')
plt.savefig('test_result.png', format='png', dpi=1200)
plt.show()

