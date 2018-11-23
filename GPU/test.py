import numpy as np 
from scipy.stats import beta 
from matplotlib.pyplot import hist, plot, show 
obs = beta.rvs(5, 5, size=2000)  # 2000 observations 
hist(obs, bins=40, normed=True) 
grid = np.linspace(0.01, 0.99, 100) 
plot(grid,beta.pdf(grid, 5, 5), 'k-', linewidth=2) 
show()
