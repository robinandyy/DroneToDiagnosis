import numpy as np
from numpy import concatenate
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


X = np.genfromtxt('Augmentation\\vineyard_data.csv', skip_header=1, usecols=(0,1,2,3), delimiter=',')
y = np.genfromtxt('Augmentation\\vineyard_data.csv', skip_header=1, usecols=(4), delimiter=',').reshape(150,1)

# use four fold validation, testing all parts of data while keeping the 0.75/0.25 train test split
skf = StratifiedKFold(n_splits=4)
skf.get_n_splits()



print(skf)
StratifiedKFold(n_splits=4, random_state=0.42, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    
    print(f"Fold {i}:")
    print(f"train test split {len(train_index)/len(X)}/{len(test_index)/len(X)}")

    # adjust train and test set according to current fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # turn y into vectors
    y_train = y_train.ravel()
    y_test = y_test.ravel()


    # set accuracy, precision, count, recall, f1 to zero so metrics mean can be calculated
    accuracy, precision, count, recall, f1, oob = 0, 0, 0, 0, 0, 0
    true_neg, true_pos, false_neg, false_pos = 0, 0, 0, 0



    for i in range(10):

        # 500 estimators and oob_score is True - same as Velez et al.
        rf = RandomForestClassifier(n_estimators=500, oob_score=True)
        rf.fit(X_train, y_train)

        # Obtain the OOB error - important for comparing to Velez et al. results
        oob_error = 1 - rf.oob_score_
        oob_pred = np.argmax(rf.oob_decision_function_, axis=1)

        # threshold was originally fine tuned - change this back later
        threshold = 0.50  
        y_probs = rf.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= threshold).astype(int)

        #record metrics and print
        accuracy_result = accuracy_score(y_test, y_pred)
        # print(f"accuracy: {accuracy_result}")

        precision_result = precision_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        # print(f"precision: {precision_result}")

        recall_result = recall_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        # print(f"recall result: {recall_result}")

        f1_result = f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        # print(f"f1 score: {f1_result}")
        tn, fp, fn, tp = confusion_matrix(y_train, oob_pred, labels=None, sample_weight=None, normalize=None).ravel().tolist()
        importances = rf.feature_importances_
        
        print(importances)
        # print(f"Matrix: {matrix}")
        true_neg += tn
        false_pos += fp
        false_neg += fn
        true_pos += tp
        accuracy += accuracy_result
        precision += precision_result
        recall += recall_result
        f1 += f1_result
        oob += oob_error
        count += 1

    
    print(f"Mean accuracy: {accuracy/count} | Mean precision: {precision/count} \n | Mean recall: {recall/count} | Mean F1 Score: {f1/count} | \n Mean OOB error: {oob/count}")
    print(f"Diseased in train {np.sum(y_train == 1.0)} | Healthy in train: {np.sum(y_train == 0.0)} \n | Diseased in test: {np.sum(y_test == 1.0)} | Healthy in test: {np.sum(y_test == 0.0)}")
    print(f"True positives: {true_pos/count} | False positives: {false_pos/count} | \n False negatives: {false_neg/count} | True negatives {true_neg/count}")
    
