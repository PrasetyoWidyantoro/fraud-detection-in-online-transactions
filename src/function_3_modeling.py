#import all realated libraries
#import libraries for data analysis
import numpy as np
import pandas as pd
import util as util

# import pickle and json file for columns and model file
import pickle
import json
import joblib
import yaml
import scipy.stats as scs

# import warnings for ignore the warnings
import warnings 
warnings.filterwarnings("ignore")

# library for model selection and models
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

# evaluation metrics for classification model
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import GridSearchCV
import json
from datetime import datetime
from sklearn.metrics import classification_report
import uuid

from tqdm import tqdm
import pandas as pd
import os
import copy
import yaml
import joblib

def time_stamp() -> datetime:
    # Return current date and time
    return datetime.now()

###################################################
#fungsi melakukan load untuk X_sm_clean, y_sm, X_valid_clean, y_valid, X_test_clean, y_test
def load_data_scaling(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    X_sm_clean = util.pickle_load(config_data["standar_scaler_sm"][0])
    y_sm = util.pickle_load(config_data["standar_scaler_sm"][1])

    X_test_clean = util.pickle_load(config_data["standar_scaler_test"][0])
    y_test = util.pickle_load(config_data["standar_scaler_test"][1])

    X_valid_clean = util.pickle_load(config_data["standar_scaler_valid"][0])
    y_valid = util.pickle_load(config_data["standar_scaler_valid"][1])

    # Return 3 set of data
    return X_sm_clean, y_sm, X_valid_clean, y_valid, X_test_clean, y_test

# fungsi binary classification untuk model extratrees
def binary_classification_extratrees(x_train, y_train, x_valid, y_valid, x_test, y_test):
    # Instantiate the classifier
    extratrees_clf = ExtraTreesClassifier(random_state=123)
    
    # Train the model
    extratrees_clf.fit(x_train, y_train)
    
    # Evaluate on validation set
    valid_pred = extratrees_clf.predict(x_valid)
    valid_acc = accuracy_score(y_valid, valid_pred)
    print('Validation accuracy:', valid_acc)
    
    # Evaluate on test set
    test_pred = extratrees_clf.predict(x_test)
    test_acc = accuracy_score(y_test, test_pred)
    print('Test accuracy:', test_acc)
    
    return extratrees_clf

#Fungsu untuk save log model yang telah dibuat
def save_model_log(model, model_name, X_test, y_test):
    # generate unique id
    model_uid = uuid.uuid4().hex
    
    # get current time and date
    now = datetime.now()
    training_time = now.strftime("%H:%M:%S")
    training_date = now.strftime("%Y-%m-%d")
    
    # generate classification report
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # create dictionary for log
    log = {"model_name": model_name,
           "model_uid": model_uid,
           "training_time": training_time,
           "training_date": training_date,
           "classification_report": report}
    
    # menyimpan log sebagai file JSON
    with open('training_log/training_log.json', 'w') as f:
        json.dump(log, f)
        
        
if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()
    
    # 2. Load dataset
    X_sm_clean, y_sm, X_valid_clean, y_valid, X_test_clean, y_test = load_data_scaling(config_data)
    
    # 3. Fitting data dan pembuatan model
    extra_trees_awal = binary_classification_extratrees(x_train = X_sm_clean, y_train = y_sm, \
                                                        x_valid = X_valid_clean, y_valid = y_valid, \
                                                        x_test = X_test_clean, y_test = y_test)
    
    # 4. Save log model pada folder training log
    save_model_log(model = extra_trees_awal, model_name = "extra_trees_model", X_test = X_test_clean, y_test=y_test)
    
    # 5. Save Model yang terlah dibuat
    extra_trees_model = config_data["model_final"]
    with open(extra_trees_model, 'wb') as file:
        pickle.dump(extra_trees_awal, file)