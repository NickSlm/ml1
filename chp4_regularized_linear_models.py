import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

x = 3 * np.random.rand(20, 1)
y = 0.5 * x + 1 + np.random.randn(20, 1) / 1.5

