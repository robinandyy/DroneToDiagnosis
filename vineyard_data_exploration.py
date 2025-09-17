import numpy as np
import matplotlib.pyplot as plt

X = np.genfromtxt('TabMDA\\vineyard_data.csv', skip_header=1, usecols=(0,1,2,3), delimiter=',')
y = np.genfromtxt('TabMDA\\vineyard_data.csv', skip_header=1, usecols=(4), delimiter=',').reshape(150,1)
all = np.genfromtxt('TabMDA\\vineyard_data.csv', skip_header=1, usecols=(0,1,2,3,4), delimiter=',')

NDRE = X[:, 0] 


# Mask where last column == 1.0
disease_mask = all[:, -1] == 1.0  
healthy_mask = all[:, -1] == 0.0


# Apply mask to first feature (col 0)
NDRE_if_diseased = all[disease_mask, 0]
NDRE_if_healthy = all[healthy_mask, 0]
NDRE_disease_avg = np.mean(NDRE_if_diseased)
NDRE_healthy_avg = np.mean(NDRE_if_healthy)

print(NDRE_if_diseased[:5])
print(NDRE_if_healthy[:5])



# NDRE_disease_min
# NDRE_healthy_min

# flatten in case they are column vectors
d = NDRE_if_diseased.ravel()
h = NDRE_if_healthy.ravel()

plt.hist(d, bins='auto', alpha=0.5, label='Diseased')
plt.hist(h, bins='auto', alpha=0.5, label='Healthy')
plt.xlabel("NDRE")
plt.ylabel("Count")
plt.title("NDRE distributions (real data)")
plt.legend()
plt.show()

