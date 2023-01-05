import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
x  = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)
print(x)


plt.plot(x,y,"y.")
plt.xlabel("$x_1$",fontsize=18)
plt.ylabel("$y$",rotation=0,fontsize=18)
plt.axis([0,2,0,15])
plt.show()


x_b = np.c_[np.ones((100,1)),x]
# normal equation - linear regression => (xT*x)^-1*xT*y
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T.dot(y))
print(theta_best)
