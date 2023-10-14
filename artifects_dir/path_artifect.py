import os
import joblib
import xgboost

# Configuration of path: as pycharm not recognizing th root folder
def load_cluster_model():
    try:
        CLUSTER_MODEL_PATH_ROOT = 'model_dir/DBScan.joblib'
        cluster_model = joblib.load(CLUSTER_MODEL_PATH_ROOT)
    except FileNotFoundError:
        CLUSTER_MODEL_PATH_ABS = 'D:/msn/pycharm_projects/deployment/model_dir/DBScan.joblib'
        cluster_model = joblib.load(CLUSTER_MODEL_PATH_ABS)
    return cluster_model


def load_xgboost_model():
    try:
        XGBOOST_MODEL_PATH_ROOT = 'model_dir/model_Xgboost.joblib'
        xgboost_model = joblib.load(XGBOOST_MODEL_PATH_ROOT)
    except FileNotFoundError:
        XGBOOST_MODEL_PATH_ABS = 'D:/msn/pycharm_projects/deployment/model_dir/model_Xgboost.joblib'
        xgboost_model = joblib.load(XGBOOST_MODEL_PATH_ABS)
    return xgboost_model


def load_feature_selection_model():
    try:
        FEATURE_SELECTION_MODEL_PATH_ROOT = 'model_dir/random_forest_feature_selection.joblib'
        feature_selection_model = joblib.load(FEATURE_SELECTION_MODEL_PATH_ROOT)
    except FileNotFoundError:
        FEATURE_SELECTION_MODEL_PATH_ABS = 'D:/msn/pycharm_projects/deployment/model_dir/random_forest_feature_selection.joblib'
        feature_selection_model = joblib.load(FEATURE_SELECTION_MODEL_PATH_ABS)
    return feature_selection_model


# database:
DATABASE_NAME = "Car"
DATABASE_COLLECTION_NAME = "Cars_collection"

# artifact for ingestion file:
CSV_DIR = 'csv_dir'
DATAFRAME_NAME = 'car_dekho.csv'
MODEL_NAME = 'model_Xgboost.joblib'
MODEL_DIR = 'model_dir'
FEATURE_MODEL = 'random_forest_feature_selection.joblib'
DBSCAN_MODEL = 'DBScan.joblib'

# path:
CSV_PATH = os.path.join(CSV_DIR, DATAFRAME_NAME)
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
FEATURE_SELECTION_MODEL_PATH = os.path.join(MODEL_DIR, FEATURE_MODEL)
DBSCAN_MODEL_PATH = os.path.join(MODEL_DIR, DBSCAN_MODEL)
