import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from prettytable import PrettyTable 
from ctgan import CTGAN
from pandas import DataFrame


num_epochs = 300
num_synthetic_samples = 175

# retrieve X and y from csv
X = np.genfromtxt('Augmentation\\vineyard_data.csv', skip_header=1, usecols=(0,1,2,3), delimiter=',')
y = np.genfromtxt('Augmentation\\vineyard_data.csv', skip_header=1, usecols=(4), delimiter=',').reshape(150,1)



all_data = np.genfromtxt('Augmentation\\vineyard_data.csv', skip_header=1, usecols=(0,1,2,3,4), delimiter=',')

discrete_columns = [0,1,2,3,4]


# create four splits for train test split 0.75/0.25 - same as Velez et al.
skf = StratifiedKFold(n_splits=4)
skf.get_n_splits()

# specify the Column Names while initializing each result table 
table = PrettyTable(["Fold", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean f1", "Mean OOB"]) 
class_table = PrettyTable(["Fold","Diseased in train", "Healthy in train", "Diseased in test", "Healthy in test"]) 
Confusion_Matrix = PrettyTable(["OOB Fold","True Positive", "True Negative", "False Positive", "False Negative"])
Acc_Confusion_Matrix = PrettyTable(["Accuracy Fold","True Positive", "True Negative", "False Positive", "False Negative"])
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




    

    
    train_data = np.hstack([X_train, y_train])
    healthy_train_data = train_data[train_data[:,-1]==0.0]
    
    diseased_train_data = train_data[train_data[:,-1]==1.0]




    ctgan_healthy = CTGAN(epochs=num_epochs)
    ctgan_diseased = CTGAN(epochs=num_epochs)


    print("epochs in progress...")


    ctgan_healthy.fit(train_data=healthy_train_data, discrete_columns=discrete_columns)
    ctgan_diseased.fit(train_data=diseased_train_data, discrete_columns=discrete_columns)

    healthy_synthetic_data = ctgan_healthy.sample(num_synthetic_samples)
    diseased_synthetic_data = ctgan_diseased.sample(num_synthetic_samples)


    X_healthy_train_syn = healthy_synthetic_data[:,:4] 
    y_healthy_train_syn = healthy_synthetic_data[:,-1:]

    X_diseased_train_syn = diseased_synthetic_data[:,:4] 
    y_diseased_train_syn = diseased_synthetic_data[:,-1:]

    X_train = np.vstack([X_train, X_healthy_train_syn, X_diseased_train_syn])
    y_train = np.vstack([y_train, y_healthy_train_syn, y_diseased_train_syn])

    
    updated_train = np.hstack([X_train, y_train])
    combined_test = np.hstack([X_test, y_test])

    total_combined = np.vstack([updated_train, combined_test])




    y_train = y_train.ravel()
    print(y_train[:3])
    y_test = y_test.ravel()
    print(y_test[:3])

    # convert array into dataframe
    DF = DataFrame(total_combined)

    # save the dataframe as a csv file
    DF.to_csv(f"Augmentation\\{num_epochs}_epochs_{num_synthetic_samples}_samples_vineyard_data.csv")

    # set accuracy, precision, count, recall, f1 to zero so metrics mean can be calculated
    accuracy, precision, count, recall, f1, oob = 0, 0, 0, 0, 0, 0

    # initialize to zero for oob confusion matrix table later
    oob_true_neg, oob_true_pos, oob_false_neg, oob_false_pos = 0, 0, 0, 0

    # initialize to zero for accuracy confusion matrix table later
    acc_true_neg, acc_true_pos, acc_false_neg, acc_false_pos = 0, 0, 0, 0

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
        oob_tn, oob_fp, oob_fn, oob_tp = confusion_matrix(y_train, oob_pred, labels=None, sample_weight=None, normalize=None).ravel().tolist()
        
        # confusion matrix based on accuracy
        acc_tn, acc_fp, acc_fn, acc_tp = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None, normalize=None).ravel().tolist()



        # get feature importances for this run and record them for averaging later
        NDRE, CHM, LAI, DTM = rf.feature_importances_.ravel()
        
        NDRE_importance += NDRE
        CHM_importance += CHM
        LAI_importance += LAI
        DTM_importance += DTM


        # normalize confusion matrix values so they can be averaged properly
        oob_true_neg += oob_tn/(oob_tn + oob_fp + oob_fn + oob_tp) * 100
        oob_false_pos += oob_fp/(oob_tn + oob_fp + oob_fn + oob_tp) * 100
        oob_false_neg += oob_fn/(oob_tn + oob_fp + oob_fn + oob_tp) * 100
        oob_true_pos += oob_tp/(oob_tn + oob_fp + oob_fn + oob_tp) * 100

        acc_true_neg += acc_tn/(acc_tn + acc_fp + acc_fn + acc_tp) * 100
        acc_false_pos += acc_fp/(acc_tn + acc_fp + acc_fn + acc_tp) * 100
        acc_false_neg += acc_fn/(acc_tn + acc_fp + acc_fn + acc_tp) * 100
        acc_true_pos += acc_tp/(acc_tn + acc_fp + acc_fn + acc_tp) * 100



        accuracy += accuracy_result
        precision += precision_result
        recall += recall_result
        f1 += f1_result
        oob += oob_error
        count += 1

    


    # add results for each fold to each table
    table.add_row([i, accuracy/count, precision/count, recall/count, f1/count, oob/count, ]) 
    class_table.add_row([i, np.sum(y_train == 1.0), np.sum(y_train == 0.0), np.sum(y_test == 1.0), np.sum(y_test == 0.0)]) 
    Confusion_Matrix.add_row([i, oob_true_pos/count, oob_true_neg/count, oob_false_pos/count, oob_false_neg/count]) 
    Acc_Confusion_Matrix.add_row([i, acc_true_pos/count, acc_true_neg/count, acc_false_pos/count, acc_false_neg/count])
    importances_table.add_row([i, NDRE_importance/count, CHM_importance/count, LAI_importance/count, DTM_importance/count])



 
print(table)
print(class_table)
print(Confusion_Matrix)
print(Acc_Confusion_Matrix)
print(importances_table)
