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



# NDRE_disease_min
# NDRE_healthy_min

# flatten in case they are column vectors
d = NDRE_if_diseased.ravel()
h = NDRE_if_healthy.ravel()


# compute stats
stats_d = {
    "mean": np.mean(d),
    "min": np.min(d),
    "max": np.max(d),
    "std": np.std(d, ddof=1)
}
stats_h = {
    "mean": np.mean(h),
    "min": np.min(h),
    "max": np.max(h),
    "std": np.std(h, ddof=1)
}

# format legend labels
label_d = (f"Diseased\n"
           f"mean={stats_d['mean']:.3f}, "
           f"min={stats_d['min']:.3f}, "
           f"max={stats_d['max']:.3f}, "
           f"std={stats_d['std']:.3f}")

label_h = (f"Healthy\n"
           f"mean={stats_h['mean']:.3f}, "
           f"min={stats_h['min']:.3f}, "
           f"max={stats_h['max']:.3f}, "
           f"std={stats_h['std']:.3f}")

plt.hist(d, bins='auto', alpha=0.5, label=label_d)
plt.hist(h, bins='auto', alpha=0.5, label=label_h)
plt.xlabel("NDRE")
plt.ylabel("Count")
plt.title("NDRE distributions (real data)")
plt.legend(fontsize=8, loc="upper right")
plt.show()





# Mask where last column == 1.0
disease_mask = all[:, -1] == 1.0  
healthy_mask = all[:, -1] == 0.0


# Apply mask to first feature (col 0)
CHM_if_diseased = all[disease_mask, 1]
CHM_if_healthy = all[healthy_mask, 1]
CHM_disease_avg = np.mean(CHM_if_diseased)
CHM_healthy_avg = np.mean(CHM_if_healthy)


# flatten in case they are column vectors
chm_d = CHM_if_diseased.ravel()
chm_h = CHM_if_healthy.ravel()

# compute stats
stats_d = {
    "mean": np.mean(chm_d),
    "min": np.min(chm_d),
    "max": np.max(chm_d),
    "std": np.std(chm_d, ddof=1)
}
stats_h = {
    "mean": np.mean(chm_h),
    "min": np.min(chm_h),
    "max": np.max(chm_h),
    "std": np.std(chm_h, ddof=1)
}

# format legend labels
label_d = (f"Diseased\n"
           f"mean={stats_d['mean']:.3f}, "
           f"min={stats_d['min']:.3f}, "
           f"max={stats_d['max']:.3f}, "
           f"std={stats_d['std']:.3f}")

label_h = (f"Healthy\n"
           f"mean={stats_h['mean']:.3f}, "
           f"min={stats_h['min']:.3f}, "
           f"max={stats_h['max']:.3f}, "
           f"std={stats_h['std']:.3f}")

plt.hist(chm_d, bins='auto', alpha=0.5, label=label_d)
plt.hist(chm_h, bins='auto', alpha=0.5, label=label_h)
plt.xlabel("CHM")
plt.ylabel("Count")
plt.title("CHM distributions (real data)")
plt.legend(fontsize=8, loc="upper right")
plt.show()






# Mask where last column == 1.0
disease_mask = all[:, -1] == 1.0  
healthy_mask = all[:, -1] == 0.0


# Apply mask to first feature (col 0)
LAI_if_diseased = all[disease_mask, 2]
LAI_if_healthy = all[healthy_mask, 2]
LAI_disease_avg = np.mean(CHM_if_diseased)
LAI_healthy_avg = np.mean(CHM_if_healthy)


# flatten in case they are column vectors
lai_d = LAI_if_diseased.ravel()
lai_h = LAI_if_healthy.ravel()

# compute stats
stats_d = {
    "mean": np.mean(lai_d),
    "min": np.min(lai_d),
    "max": np.max(lai_d),
    "std": np.std(lai_d, ddof=1)
}
stats_h = {
    "mean": np.mean(lai_h),
    "min": np.min(lai_h),
    "max": np.max(lai_h),
    "std": np.std(lai_h, ddof=1)
}

# format legend labels
label_d = (f"Diseased\n"
           f"mean={stats_d['mean']:.3f}, "
           f"min={stats_d['min']:.3f}, "
           f"max={stats_d['max']:.3f}, "
           f"std={stats_d['std']:.3f}")

label_h = (f"Healthy\n"
           f"mean={stats_h['mean']:.3f}, "
           f"min={stats_h['min']:.3f}, "
           f"max={stats_h['max']:.3f}, "
           f"std={stats_h['std']:.3f}")

plt.hist(lai_d, bins='auto', alpha=0.5, label=label_d)
plt.hist(lai_h, bins='auto', alpha=0.5, label=label_h)
plt.xlabel("LAI")
plt.ylabel("Count")
plt.title("LAI distributions (real data)")
plt.legend(fontsize=8, loc="upper right")
plt.show()





# Mask where last column == 1.0
disease_mask = all[:, -1] == 1.0  
healthy_mask = all[:, -1] == 0.0


# Apply mask to first feature (col 0)
DTM_if_diseased = all[disease_mask, 3]
DTM_if_healthy = all[healthy_mask, 3]
DTM_disease_avg = np.mean(LAI_if_diseased)
DTM_healthy_avg = np.mean(LAI_if_healthy)


# flatten in case they are column vectors
dtm_d = DTM_if_diseased.ravel()
dtm_h = DTM_if_healthy.ravel()

# compute stats
stats_d = {
    "mean": np.mean(dtm_d),
    "min": np.min(dtm_d),
    "max": np.max(dtm_d),
    "std": np.std(dtm_d, ddof=1)
}
stats_h = {
    "mean": np.mean(dtm_h),
    "min": np.min(dtm_h),
    "max": np.max(dtm_h),
    "std": np.std(dtm_h, ddof=1)
}

# format strings for legend
label_d = (f"Diseased\n"
           f"mean={stats_d['mean']:.3f}, "
           f"min={stats_d['min']:.3f}, "
           f"max={stats_d['max']:.3f}, "
           f"std={stats_d['std']:.3f}")

label_h = (f"Healthy\n"
           f"mean={stats_h['mean']:.3f}, "
           f"min={stats_h['min']:.3f}, "
           f"max={stats_h['max']:.3f}, "
           f"std={stats_h['std']:.3f}")

plt.hist(dtm_d, bins='auto', alpha=0.5, label=label_d)
plt.hist(dtm_h, bins='auto', alpha=0.5, label=label_h)
plt.xlabel("DTM")
plt.ylabel("Count")
plt.title("DTM distributions (real data)")
plt.legend(fontsize=8, loc="upper left")
plt.show()

