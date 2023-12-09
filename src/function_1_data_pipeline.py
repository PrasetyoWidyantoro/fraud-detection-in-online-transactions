from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import joblib
import yaml
from datetime import datetime
import util as util
#Import library untuk data preparation dan visualization
import numpy as np
# import warnings for ignore the warnings
import warnings 
warnings.filterwarnings("ignore")
# import pickle and json file for columns and model file
import pickle
import json
import copy

#Fungsi read data
def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw dataset dir
    raw_dataset_dir_1 = config["raw_dataset_path_1"]
    raw_dataset_dir_2 = config["raw_dataset_path_2"]
    
    train_transaction = pd.read_csv(raw_dataset_dir_1)
    
    # Read train_identity.csv
    train_identity = pd.read_csv(raw_dataset_dir_2)
    
    # Merge train_transaction and train_identity based on 'TransactionID'
    train_set = pd.merge(train_transaction, train_identity, how='left', on='TransactionID')

    # Return raw dataset
    return train_set


if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()
    
    #2. Read all raw Dataset
    train_set = read_raw_data(config_data).drop(['TransactionID', 'TransactionDT'], axis = 1)
    
    #3. Menghitung persentase nilai null pada setiap kolom
    null_percentages = train_set.isnull().mean() * 100

    #4. Mengidentifikasi kolom-kolom dengan persentase nilai null di atas 50%
    columns_to_drop = null_percentages[null_percentages > 50].index

    #5. Menghapus kolom-kolom yang memiliki persentase nilai null di atas 50%
    df_train = train_set.drop(columns=columns_to_drop)
    
    #5. Splitting input output
    # Pemisahan Variabel X dan Y
    X = df_train.drop(columns = "isFraud")
    y = df_train["isFraud"]

    #6. Splitting train test
    #Split Data 70% training 30% testing
    X_train, X_test, \
    y_train, y_test = train_test_split(
        X, y, 
        test_size = 0.3, 
        random_state = 123)
    
    #6. Splitting test valid
    X_valid, X_test, \
    y_valid, y_test = train_test_split(
        X_test, y_test,
        test_size = 0.4,
        random_state = 42,
        stratify = y_test
    )
    
    #Menggabungkan x train dan y train untuk keperluan EDA
    util.pickle_dump(X_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(X_valid, config_data["valid_set_path"][0])
    util.pickle_dump(y_valid, config_data["valid_set_path"][1])

    util.pickle_dump(X_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])
    
    print("Data Pipeline passed successfully.")
