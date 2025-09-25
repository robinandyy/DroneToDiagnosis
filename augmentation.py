import numpy as np
import matplotlib.pyplot as plt
from ctgan import CTGAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.genfromtxt('vineyard_data.csv', skip_header=1, usecols=(0,1,2,3), delimiter=',')
y = np.genfromtxt('vineyard_data.csv', skip_header=1, usecols=(4), delimiter=',').reshape(150,1)


all_data = np.genfromtxt('vineyard_data.csv', skip_header=1, usecols=(0,1,2,3,4), delimiter=',')

discrete_columns = [4]

ctgan = CTGAN(epochs=10)
ctgan.fit(all_data, discrete_columns)

synthetic_data = ctgan.sample(3000)


X = all_data[:,:4]
y = all_data[:,-1:].ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

rf = RandomForestClassifier(n_estimators=500, oob_score=True)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))

