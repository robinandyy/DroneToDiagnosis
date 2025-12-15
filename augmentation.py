import numpy as np
import matplotlib.pyplot as plt
from ctgan import CTGAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



X = np.genfromtxt('Augmentation\\vineyard_data.csv', skip_header=1, usecols=(0,1,2,3), delimiter=',')
y = np.genfromtxt('Augmentation\\vineyard_data.csv', skip_header=1, usecols=(4), delimiter=',').reshape(150,1)


all_data = np.genfromtxt('Augmentation\\vineyard_data.csv', skip_header=1, usecols=(0,1,2,3,4), delimiter=',')

discrete_columns = [4]

train_data = all_data[:113]
test_data = all_data[113:]

print(len(train_data),len(test_data))




# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=42)

# y_train = y_train.ravel()
# y_test = y_test.ravel()

ctgan = CTGAN(epochs=10000)
ctgan.fit(train_data=train_data, discrete_columns=discrete_columns)

synthetic_data = ctgan.sample(300)

X_train = synthetic_data[:,:4]
y_train = synthetic_data[:,-1:].ravel()
X_test = test_data[:,:4]
y_test = test_data[:,-1:].ravel()

rf = RandomForestClassifier(n_estimators=500, oob_score=True)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))
