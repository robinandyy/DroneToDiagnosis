import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from prettytable import PrettyTable 

# retrieve X and y from csv
X = np.genfromtxt('Augmentation\\vineyard_data.csv', skip_header=1, usecols=(0,1,2,3), delimiter=',')
y = np.genfromtxt('Augmentation\\vineyard_data.csv', skip_header=1, usecols=(4), delimiter=',').reshape(150,1)

# create four splits for train test split 0.75/0.25 - same as Velez et al.
skf = StratifiedKFold(n_splits=4)
skf.get_n_splits()

# specify the Column Names while initializing each result table 
table = PrettyTable(["Fold", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean f1", "Mean OOB"]) 
class_table = PrettyTable(["Fold","Diseased in train", "Healthy in train", "Diseased in test", "Healthy in test"]) 
Confusion_Matrix = PrettyTable(["Fold","True Positive", "True Negative", "False Positive", "False Negative"])
importances_table = PrettyTable(["Fold", "Mean NDRE importance", "Mean CHM importance", "Mean LAI importance", "Mean DTM importance"])



print(skf)
StratifiedKFold(n_splits=4, random_state=0.42, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    print(f"Fold {i} in progress...")

    # uncomment below to see how folds are distributed across indexes

    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")


    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    y_train = y_train.ravel()
    y_test = y_test.ravel()


    # set accuracy, precision, count, recall, f1 to zero so metrics mean can be calculated
    accuracy, precision, count, recall, f1, oob = 0, 0, 0, 0, 0, 0

    # initialize to zero for confusion matrix table later
    true_neg, true_pos, false_neg, false_pos = 0, 0, 0, 0

    # initialize to zero to track feature importances
    NDRE_importance, CHM_importance, LAI_importance, DTM_importance = 0, 0, 0, 0


    for j in range(10):

        # 500 estimators and oob_score is True - same as Velez et al.
        rf = RandomForestClassifier(n_estimators=500, oob_score=True)

        # train RF model
        rf.fit(X_train, y_train)

        # Obtain the OOB error
        oob_error = 1 - rf.oob_score_
        oob_pred = np.argmax(rf.oob_decision_function_, axis=1)

        # test RF on test set
        y_pred = rf.predict(X_test)


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

        # confusion matrix based on OOB 
        # not good practice, but the same as Velez et al. for the sake of comparison
        tn, fp, fn, tp = confusion_matrix(y_train, oob_pred, labels=None, sample_weight=None, normalize=None).ravel().tolist()
        
        NDRE, CHM, LAI, DTM = rf.feature_importances_.ravel()
        
        # record importances of each feature for this run
        NDRE_importance += NDRE
        CHM_importance += CHM
        LAI_importance += LAI
        DTM_importance += DTM

        # normalize confusion matrix values so they can be averaged properly
        true_neg += tn/(tn + fp + fn + tp) * 100
        false_pos += fp/(tn + fp + fn + tp) * 100
        false_neg += fn/(tn + fp + fn + tp) * 100
        true_pos += tp/(tn + fp + fn + tp) * 100



        accuracy += accuracy_result
        precision += precision_result
        recall += recall_result
        f1 += f1_result
        oob += oob_error
        count += 1

    


    # add results for each fold to each table
    table.add_row([i, accuracy/count, precision/count, recall/count, f1/count, oob/count, ]) 
    class_table.add_row([i, np.sum(y_train == 1.0), np.sum(y_train == 0.0), np.sum(y_test == 1.0), np.sum(y_test == 0.0)]) 
    Confusion_Matrix.add_row([i, true_pos/count, true_neg/count, false_pos/count, false_neg/count]) 
    importances_table.add_row([i, NDRE_importance/count, CHM_importance/count, LAI_importance/count, DTM_importance/count])

 
print(table)
print(class_table)
print(Confusion_Matrix)
print(importances_table)
