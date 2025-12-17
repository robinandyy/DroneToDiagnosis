import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# read vineyard data
all_data = pd.read_csv('Augmentation\\vineyard_data.csv')
discrete_columns = [4]


# train/test split 0.75 and 0.25
train_data = all_data[:113]
test_data = all_data[113:]

metadata = Metadata.detect_from_dataframe(train_data)

#split train data between healthy and disease to maintain relationships between features within each class
label_col = train_data.columns[-1]
healthy = train_data[train_data[label_col] == 0]
diseased = train_data[train_data[label_col] == 1]

# create synthetic data for healthy and diseased grapevines
syn_h = GaussianCopulaSynthesizer(
    Metadata.detect_from_dataframe(healthy)
)
syn_h.fit(healthy)

syn_d = GaussianCopulaSynthesizer(
    Metadata.detect_from_dataframe(diseased)
)
syn_d.fit(diseased)

synthetic_data = pd.concat([
    syn_h.sample(10),
    syn_d.sample(10)
])


# store real and synthetic training data
X_train_real = train_data.iloc[:,0:4]
X_train_syn = synthetic_data.iloc[:,0:4]
y_train_real = train_data.iloc[:,-1:].squeeze()
y_train_syn = synthetic_data.iloc[:,-1:].squeeze()

# combine
X_train = pd.concat([X_train_real, X_train_syn]) 
y_train = pd.concat([y_train_real, y_train_syn]) 


X_test = test_data.iloc[:,:4]
y_test = test_data.iloc[:,-1:].squeeze()

# train Random Forest model
rf = RandomForestClassifier(n_estimators=500, oob_score=True)
rf.fit(X_train, y_train)

# check accuracy
y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))
