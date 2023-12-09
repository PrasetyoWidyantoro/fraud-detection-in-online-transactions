import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import util as util
import function_1_data_pipeline as function_1_data_pipeline
import function_2_data_processing as function_2_data_processing
import function_3_modeling as function_3_modeling
###################################################
#API
app = FastAPI() 

# kumpulan kolom float
float_columns  = [
    'card1', 'TransactionAmt', 'card2', 'card3', 'card5', 'addr1', 'addr2',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D3', 'D4', 'D10', 'D11', 'D15',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30',
    'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40',
    'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50',
    'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60',
    'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70',
    'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80',
    'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90',
    'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100',
    'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110',
    'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120',
    'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130',
    'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137',
    'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288',
    'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298',
    'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308',
    'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318',
    'V319', 'V320', 'V321'
]

# kumpulan kolom object
object_columns = [
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M6'
]

#Range masing-masing kolom numerikal
range_constraints = {
                    #'isFraud': (0, 1),
                    'TransactionAmt': (0.251, 31937.391),
                    'card1': (1000, 18396),
                    'card2': (100.0, 600.0),
                    'card3': (100.0, 231.0),
                    'card5': (100.0, 237.0),
                    'addr1': (100.0, 540.0),
                    'addr2': (10.0, 102.0),
                    'C1': (0.0, 4685.0),
                    'C2': (0.0, 5691.0),
                    'C3': (0.0, 26.0),
                    'C4': (0.0, 2253.0),
                    'C5': (0.0, 349.0),
                    'C6': (0.0, 2253.0),
                    'C7': (0.0, 2255.0),
                    'C8': (0.0, 3331.0),
                    'C9': (0.0, 210.0),
                    'C10': (0.0, 3257.0),
                    'C11': (0.0, 3188.0),
                    'C12': (0.0, 3188.0),
                    'C13': (0.0, 2918.0),
                    'C14': (0.0, 1429.0),
                    'D1': (0.0, 640.0),
                    'D2': (0.0, 640.0),
                    'D3': (0.0, 819.0),
                    'D4': (-122.0, 869.0),
                    'D10': (0.0, 876.0),
                    'D11': (-53.0, 670.0),
                    'D15': (-83.0, 879.0),
                    'V1': (0.0, 1.0),
                    'V2': (0.0, 8.0),
                    'V3': (0.0, 9.0),
                    'V4': (0.0, 6.0),
                    'V5': (0.0, 6.0),
                    'V6': (0.0, 9.0),
                    'V7': (0.0, 9.0),
                    'V8': (0.0, 8.0),
                    'V9': (0.0, 8.0),
                    'V10': (0.0, 4.0),
                    'V11': (0.0, 5.0),
                    'V12': (0.0, 3.0),
                    'V13': (0.0, 6.0),
                    'V14': (0.0, 1.0),
                    'V15': (0.0, 7.0),
                    'V16': (0.0, 15.0),
                    'V17': (0.0, 15.0),
                    'V18': (0.0, 15.0),
                    'V19': (0.0, 7.0),
                    'V20': (0.0, 15.0),
                    'V21': (0.0, 5.0),
                    'V22': (0.0, 8.0),
                    'V23': (0.0, 13.0),
                    'V24': (0.0, 13.0),
                    'V25': (0.0, 7.0),
                    'V26': (0.0, 13.0),
                    'V27': (0.0, 4.0),
                    'V29': (0.0, 5.0),
                    'V30': (0.0, 9.0),
                    'V31': (0.0, 7.0),
                    'V32': (0.0, 15.0),
                    'V33': (0.0, 7.0),
                    'V34': (0.0, 13.0),
                    'V35': (0.0, 3.0),
                    'V36': (0.0, 5.0),
                    'V37': (0.0, 54.0),
                    'V38': (0.0, 54.0),
                    'V39': (0.0, 15.0),
                    'V40': (0.0, 24.0),
                    'V41': (0.0, 1.0),
                    'V42': (0.0, 8.0),
                    'V43': (0.0, 8.0),
                    'V44': (0.0, 48.0),
                    'V45': (0.0, 48.0),
                    'V46': (0.0, 6.0),
                    'V47': (0.0, 12.0),
                    'V48': (0.0, 5.0),
                    'V49': (0.0, 5.0),
                    'V50': (0.0, 5.0),
                    'V51': (0.0, 6.0),
                    'V52': (0.0, 12.0),
                    'V53': (0.0, 5.0),
                    'V54': (0.0, 6.0),
                    'V55': (0.0, 17.0),
                    'V56': (0.0, 51.0),
                    'V57': (0.0, 6.0),
                    'V58': (0.0, 10.0),
                    'V59': (0.0, 16.0),
                    'V60': (0.0, 16.0),
                    'V61': (0.0, 6.0),
                    'V62': (0.0, 10.0),
                    'V63': (0.0, 7.0),
                    'V64': (0.0, 7.0),
                    'V65': (0.0, 1.0),
                    'V66': (0.0, 7.0),
                    'V67': (0.0, 8.0),
                    'V68': (0.0, 2.0),
                    'V69': (0.0, 5.0),
                    'V70': (0.0, 6.0),
                    'V71': (0.0, 6.0),
                    'V72': (0.0, 10.0),
                    'V73': (0.0, 7.0),
                    'V74': (0.0, 8.0),
                    'V75': (0.0, 4.0),
                    'V76': (0.0, 6.0),
                    'V77': (0.0, 30.0),
                    'V78': (0.0, 31.0),
                    'V79': (0.0, 7.0),
                    'V80': (0.0, 19.0),
                    'V81': (0.0, 19.0),
                    'V82': (0.0, 7.0),
                    'V83': (0.0, 7.0),
                    'V84': (0.0, 7.0),
                    'V85': (0.0, 7.0),
                    'V86': (0.0, 30.0),
                    'V87': (0.0, 30.0),
                    'V88': (0.0, 1.0),
                    'V89': (0.0, 2.0),
                    'V90': (0.0, 5.0),
                    'V91': (0.0, 6.0),
                    'V92': (0.0, 7.0),
                    'V93': (0.0, 7.0),
                    'V94': (0.0, 2.0),
                    'V95': (0.0, 880.0),
                    'V96': (0.0, 1410.0),
                    'V97': (0.0, 976.0),
                    'V98': (0.0, 12.0),
                    'V99': (0.0, 88.0),
                    'V100': (0.0, 28.0),
                    'V101': (0.0, 869.0),
                    'V102': (0.0, 1285.0),
                    'V103': (0.0, 928.0),
                    'V104': (0.0, 15.0),
                    'V105': (0.0, 99.0),
                    'V106': (0.0, 55.0),
                    'V107': (0.0, 1.0),
                    'V108': (0.0, 7.0),
                    'V109': (0.0, 7.0),
                    'V110': (0.0, 7.0),
                    'V111': (0.0, 9.0),
                    'V112': (0.0, 9.0),
                    'V113': (0.0, 9.0),
                    'V114': (0.0, 6.0),
                    'V115': (0.0, 6.0),
                    'V116': (0.0, 6.0),
                    'V117': (0.0, 3.0),
                    'V118': (0.0, 3.0),
                    'V119': (0.0, 3.0),
                    'V120': (0.0, 3.0),
                    'V121': (0.0, 3.0),
                    'V122': (0.0, 3.0),
                    'V123': (0.0, 13.0),
                    'V124': (0.0, 13.0),
                    'V125': (0.0, 13.0),
                    'V126': (0.0, 160000.0),
                    'V127': (0.0, 160000.0),
                    'V128': (0.0, 160000.0),
                    'V129': (0.0, 55125.0),
                    'V130': (0.0, 55125.0),
                    'V131': (0.0, 55125.0),
                    'V132': (0.0, 93736.0),
                    'V133': (0.0, 133915.0),
                    'V134': (0.0, 98476.0),
                    'V135': (0.0, 90750.0),
                    'V136': (0.0, 90750.0),
                    'V137': (0.0, 90750.0),
                    'V279': (0.0, 880.0),
                    'V280': (0.0, 975.0),
                    'V281': (0.0, 22.0),
                    'V282': (0.0, 32.0),
                    'V283': (0.0, 68.0),
                    'V284': (0.0, 12.0),
                    'V285': (0.0, 95.0),
                    'V286': (0.0, 8.0),
                    'V287': (0.0, 31.0),
                    'V288': (0.0, 10.0),
                    'V289': (0.0, 12.0),
                    'V290': (1.0, 67.0),
                    'V291': (1.0, 1055.0),
                    'V292': (1.0, 323.0),
                    'V293': (0.0, 869.0),
                    'V294': (0.0, 1286.0),
                    'V295': (0.0, 928.0),
                    'V296': (0.0, 93.0),
                    'V297': (0.0, 12.0),
                    'V298': (0.0, 93.0),
                    'V299': (0.0, 49.0),
                    'V300': (0.0, 11.0),
                    'V301': (0.0, 13.0),
                    'V302': (0.0, 16.0),
                    'V303': (0.0, 20.0),
                    'V304': (0.0, 16.0),
                    'V305': (1.0, 2.0),
                    'V306': (0.0, 108800.0),
                    'V307': (0.0, 145765.0),
                    'V308': (0.0, 108800.0),
                    'V309': (0.0, 55125.0),
                    'V310': (0.0, 55125.0),
                    'V311': (0.0, 55125.0),
                    'V312': (0.0, 55125.0),
                    'V313': (0.0, 4817.47021484375),
                    'V314': (0.0, 7519.8701171875),
                    'V315': (0.0, 4817.47021484375),
                    'V316': (0.0, 93736.0),
                    'V317': (0.0, 134021.0),
                    'V318': (0.0, 98476.0),
                    'V319': (0.0, 104060.0),
                    'V320': (0.0, 104060.0),
                    'V321': (0.0, 104060.0)
                    }

category_values = {
    'ProductCD': ['W', 'H', 'C', 'S', 'R', 'empty'],
    'card4': ['discover', 'mastercard', 'visa', 'american express', 'empty'],
    'card6': ['credit', 'debit', 'empty', 'debit or credit', 'charge card'],
    'P_emaildomain': ['empty', 'gmail.com', 'outlook.com', 'yahoo.com', 'mail.com', 'anonymous.com', 'hotmail.com',
                      'verizon.net', 'aol.com', 'me.com', 'comcast.net', 'optonline.net', 'cox.net', 'charter.net',
                      'rocketmail.com', 'prodigy.net.mx', 'embarqmail.com', 'icloud.com', 'live.com.mx', 'gmail',
                      'live.com', 'att.net', 'juno.com', 'ymail.com', 'sbcglobal.net', 'bellsouth.net', 'msn.com',
                      'q.com', 'yahoo.com.mx', 'centurylink.net', 'servicios-ta.com', 'earthlink.net', 'hotmail.es',
                      'cfl.rr.com', 'roadrunner.com', 'netzero.net', 'gmx.de', 'suddenlink.net', 'frontiernet.net',
                      'windstream.net', 'frontier.com', 'outlook.es', 'mac.com', 'netzero.com', 'aim.com', 'web.de',
                      'twc.com', 'cableone.net', 'yahoo.fr', 'yahoo.de', 'yahoo.es', 'sc.rr.com', 'ptd.net',
                      'live.fr', 'yahoo.co.uk', 'hotmail.fr', 'hotmail.de', 'hotmail.co.uk', 'protonmail.com',
                      'yahoo.co.jp'],
    'M1': ['T', 'empty', 'F'],
    'M2': ['T', 'empty', 'F'],
    'M3': ['T', 'empty', 'F'],
    'M4': ['M2', 'M0', 'empty', 'M1'],
    'M6': ['T', 'F', 'empty']
}

# Unique Values pada kolom kategorikal
def data_defense(data, float_columns, object_columns, range_constraints, category_values):
    # Check float columns
    float_cols = data.select_dtypes(include=['float64']).columns
    for col in float_cols:
        assert col in float_columns, f"Error: Column '{col}' is not allowed as a float column."

    # Check object columns
    object_cols = data.select_dtypes(include=['object']).columns
    for col in object_cols:
        assert col in object_columns, f"Error: Column '{col}' is not allowed as an object column."
        # Check category values
        assert data[col].isin(category_values[col]).all(), f"Error: Column '{col}' has invalid category values."

    # Check range constraints
    for col, (min_val, max_val) in range_constraints.items():
        assert data[col].dtype == 'float64', f"Error: Column '{col}' should be of type float64."
        assert data[col].between(min_val, max_val).all(), f"Error: Column '{col}' has values outside the allowed range."
    
#Fungsi untuk imputasi Data    
def load_data_impute(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    X_train_impute = util.pickle_load(config_data["impute_data_train"][0])
    y_train = util.pickle_load(config_data["impute_data_train"][1])

    X_test_impute = util.pickle_load(config_data["impute_data_test"][0])
    y_test = util.pickle_load(config_data["impute_data_test"][1])

    X_valid_impute = util.pickle_load(config_data["impute_data_test"][0])
    y_valid = util.pickle_load(config_data["impute_data_test"][1])
    
    return X_train_impute, y_train, X_test_impute, y_test, X_valid_impute, y_valid

config_data = util.load_config()
# Load dataset
X_train, X_valid, X_test, y_train, y_valid, y_test = function_2_data_processing.load_data(config_data)
# Load dataset
dataset, valid_set, test_set = function_2_data_processing.load_dataset(config_data)
# load dataset
X_train_impute, y_train, X_test_impute, y_test, X_valid_impute, y_valid = load_data_impute(config_data)

#Scaler
scaler = function_2_data_processing.load_scaler(config_data["scaler"])
# Load model and make prediction])
model = joblib.load(config_data["model_final"])
#input kolom yang akan melalui proses prediksi
class api_data(BaseModel):
        card1: float
        TransactionAmt: float
        card2: float
        card3: float
        card5: float
        addr1: float
        addr2: float
        C1: float
        C2: float
        C3: float
        C4: float
        C5: float
        C6: float
        C7: float
        C8: float
        C9: float
        C10: float
        C11: float
        C12: float
        C13: float
        C14: float
        D1: float
        D2: float
        D3: float
        D4: float
        D10: float
        D11: float
        D15: float
        V1: float
        V2: float
        V3: float
        V4: float
        V5: float
        V6: float
        V7: float
        V8: float
        V9: float
        V10: float
        V11: float
        V12: float
        V13: float
        V14: float
        V15: float
        V16: float
        V17: float
        V18: float
        V19: float
        V20: float
        V21: float
        V22: float
        V23: float
        V24: float
        V25: float
        V26: float
        V27: float
        V28: float
        V29: float
        V30: float
        V31: float
        V32: float
        V33: float
        V34: float
        V35: float
        V36: float
        V37: float
        V38: float
        V39: float
        V40: float
        V41: float
        V42: float
        V43: float
        V44: float
        V45: float
        V46: float
        V47: float
        V48: float
        V49: float
        V50: float
        V51: float
        V52: float
        V53: float
        V54: float
        V55: float
        V56: float
        V57: float
        V58: float
        V59: float
        V60: float
        V61: float
        V62: float
        V63: float
        V64: float
        V65: float
        V66: float
        V67: float
        V68: float
        V69: float
        V70: float
        V71: float
        V72: float
        V73: float
        V74: float
        V75: float
        V76: float
        V77: float
        V78: float
        V79: float
        V80: float
        V81: float
        V82: float
        V83: float
        V84: float
        V85: float
        V86: float
        V87: float
        V88: float
        V89: float
        V90: float
        V91: float
        V92: float
        V93: float
        V94: float
        V95: float
        V96: float
        V97: float
        V98: float
        V99: float
        V100: float
        V101: float
        V102: float
        V103: float
        V104: float
        V105: float
        V106: float
        V107: float
        V108: float
        V109: float
        V110: float
        V111: float
        V112: float
        V113: float
        V114: float
        V115: float
        V116: float
        V117: float
        V118: float
        V119: float
        V120: float
        V121: float
        V122: float
        V123: float
        V124: float
        V125: float
        V126: float
        V127: float
        V128: float
        V129: float
        V130: float
        V131: float
        V132: float
        V133: float
        V134: float
        V135: float
        V136: float
        V137: float
        V279: float
        V280: float
        V281: float
        V282: float
        V283: float
        V284: float
        V285: float
        V286: float
        V287: float
        V288: float
        V289: float
        V290: float
        V291: float
        V292: float
        V293: float
        V294: float
        V295: float
        V296: float
        V297: float
        V298: float
        V299: float
        V300: float
        V301: float
        V302: float
        V303: float
        V304: float
        V305: float
        V306: float
        V307: float
        V308: float
        V309: float
        V310: float
        V311: float
        V312: float
        V313: float
        V314: float
        V315: float
        V316: float
        V317: float
        V318: float
        V319: float
        V320: float
        V321: float
        ProductCD: str
        card4: str
        card6: str
        P_emaildomain: str
        M1: str
        M2: str
        M3: str
        M4: str
        M6: str
        
@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: api_data):
    #load konfigurasi
    config_data = util.load_config()
    # Convert data api to dataframe
    df = pd.DataFrame(data.dict(), index=[0])
    
    # Data defense untuk memastikan format data sesuai dengan ketentuan
    data_defense(df, float_columns, object_columns, range_constraints, category_values)
    
    # Get Dummies for Categorical Columns
    dataset, df = function_2_data_processing.get_dummies(X_train_impute, df)
    
    # Standart Scaler
    df = function_2_data_processing.transform_data(df, scaler)
    
    #Sort Columns
    df = df[sorted(df.columns)]
    
    # Make prediction
    prediction = model.predict(df)
    
    #if else untuk membatasi ketika output 0 maka "Class 0 = Non Fraud" dan 1 maka "Class 1 = Fraud"
    if prediction[0] == 0:
        return "Class 0 = Non Fraud"
    else:
        return "Class 1 = Fraud"

#melakukan running dengan melakukan ekspansi pada port 8000
if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8000)
    
    
    
    
