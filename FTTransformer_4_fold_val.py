import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from mambular.models import FTTransformerClassifier
from prettytable import PrettyTable

# Set random seed for reproducibility
np.random.seed(0)

# Number of samples
n_samples = 1000
n_features = 5

# Generate random features
# X = np.random.randn(n_samples, n_features)
# coefficients = np.random.randn(n_features)

# Generate target variable
# y = np.dot(X, coefficients) + np.random.randn(n_samples)
# # Convert y to multiclass by categorizing into quartiles
# y = pd.qcut(y, 4, labels=False)

# Create a DataFrame to store the data
all_data = np.genfromtxt('Augmentation\\300_epochs_175_samples_vineyard_data.csv', skip_header=1, usecols=(1,2,3,4,5), delimiter=',')
print(f"all data: {all_data[:10]}")

X = pd.read_csv('Augmentation\\vineyard_data.csv').iloc[:,0:4]
y = pd.read_csv('Augmentation\\vineyard_data.csv').iloc[:,4:]


# X = all_data[:,:4]
# y = all_data[:,4:].ravel()
# X = all_data.iloc[:, 1:5]
# y = all_data.iloc[:, 5:]

print(f"original y: {y[:3]} ")

# data = pd.DataFrame(X, columns=["NDRE", "CHM", "LAI", "DTM"])
# # data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
# data["Disease"] = y

# Split data into features and target variable
# X = data.iloc[:, 1:5]
# y = data["Disease"].values



# class FTTransformerClassifier(SklearnBaseClassifier):
#     __doc__ = generate_docstring(
#         DefaultFTTransformerConfig,
#         """FTTransformer Classifier. This class extends the SklearnBaseClassifier class
#         and uses the FTTransformer model with the default FTTransformer configuration.""",
#         examples="""
#         >>> from deeptab.models import FTTransformerClassifier
#         >>> model = FTTransformerClassifier(d_model=64, n_layers=8)
#         >>> model.fit(X_train, y_train)
#         >>> preds = model.predict(X_test)
#         >>> model.evaluate(X_test, y_test)
#         """,
#     )

#     def __init__(self, **kwargs):
#         super().__init__(
#             model=FTTransformer, config=DefaultFTTransformerConfig, **kwargs
#         )



table = PrettyTable(["Fold", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean f1", "Mean OOB"]) 
# class_table = PrettyTable(["Fold","Diseased in train", "Healthy in train", "Diseased in test", "Healthy in test"]) 
Confusion_Matrix = PrettyTable(["OOB Fold","True Positive", "True Negative", "False Positive", "False Negative"])
Acc_Confusion_Matrix = PrettyTable(["Accuracy Fold","True Positive", "True Negative", "False Positive", "False Negative"])
# importances_table = PrettyTable(["Fold", "Mean NDRE importance", "Mean CHM importance", "Mean LAI importance", "Mean DTM importance"])


# create four splits for train test split 0.75/0.25 - same as Velez et al.
skf = StratifiedKFold(n_splits=4)
skf.get_n_splits()

print(skf)
StratifiedKFold(n_splits=4, random_state=0.42, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    print(f"Fold {i} in progress...")

    # uncomment below to see how folds are distributed across indexes

    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")


    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


    # X_train = X_train.to_numpy()
    # X_test = X_test.to_numpy()
    y_train = y_train.to_numpy().ravel()
    print(f"y train shape: {y_train.shape} | dimensions: {y_train.ndim}")
    y_test = y_test.to_numpy().ravel()
    print(f"y test shape: {y_test.shape} | dimensions: {y_test.ndim}")

    # set accuracy, precision, count, recall, f1 to zero so metrics mean can be calculated
    accuracy, precision, count, recall, f1, oob = 0, 0, 0, 0, 0, 0

    # initialize to zero for accuracy confusion matrix table later
    acc_true_neg, acc_true_pos, acc_false_neg, acc_false_pos = 0, 0, 0, 0

    # initialize to zero to track feature importances
    NDRE_importance, CHM_importance, LAI_importance, DTM_importance = 0, 0, 0, 0


    for j in range(10):

        ft = FTTransformerClassifier(d_model=64, n_layers=8)
        ft.fit(X_train, y_train)
        preds = ft.predict(X_test)
        ft.evaluate(X_test, y_test)



        # Obtain the OOB error
        # oob_error = 1 - ft.oob_score_
        # oob_pred = np.argmax(ft.oob_decision_function_, axis=1)

        # test RF on test set
        y_pred = ft.predict(X_test)


        #record metrics 
        # uncomment prints to track specific metrics

        accuracy_result = accuracy_score(y_test, y_pred)
        # print(f"accuracy: {accuracy_result}")

        precision_result = precision_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        # print(f"precision: {precision_result}")

        recall_result = recall_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        # print(f"recall result: {recall_result}")

        f1_result = f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        # print(f"f1 score: {f1_result}")


        # confusion matrix based on accuracy
        acc_tn, acc_fp, acc_fn, acc_tp = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None, normalize=None).ravel().tolist()



        # get feature importances for this run and record them for averaging later
        # NDRE, CHM, LAI, DTM = ft.feature_importances_.ravel()
        
        # NDRE_importance += NDRE
        # CHM_importance += CHM
        # LAI_importance += LAI
        # DTM_importance += DTM


        # normalize confusion matrix values so they can be averaged properly

        acc_true_neg += acc_tn/(acc_tn + acc_fp + acc_fn + acc_tp) * 100
        acc_false_pos += acc_fp/(acc_tn + acc_fp + acc_fn + acc_tp) * 100
        acc_false_neg += acc_fn/(acc_tn + acc_fp + acc_fn + acc_tp) * 100
        acc_true_pos += acc_tp/(acc_tn + acc_fp + acc_fn + acc_tp) * 100



        accuracy += accuracy_result
        precision += precision_result
        recall += recall_result
        f1 += f1_result
        count += 1

    


    # add results for each fold to each table
    table.add_row([i, accuracy/count, precision/count, recall/count, f1/count, oob/count, ]) 
    Acc_Confusion_Matrix.add_row([i, acc_true_pos/count, acc_true_neg/count, acc_false_pos/count, acc_false_neg/count])
    # importances_table.add_row([i, NDRE_importance/count, CHM_importance/count, LAI_importance/count, DTM_importance/count])



 
print(table)
# print(class_table)
print(Acc_Confusion_Matrix)
