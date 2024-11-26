import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import datetime
import pandas as pd
from ai_core_sdk.models import Metric
from ai_core_sdk.tracking import Tracking
import pickle
import torch
from preprocessing_utils import process_landmark_embeddings
from model_utils import train

print(f'TORCH VERSION: {torch.__version__}')
print(f'CUDA AVAIL: {torch.cuda.is_available()}')

# Variables
DATA_PATH = os.getenv('DATA_PATH', '/app/data/pose_embeddings.csv')
METADATA_PATH = os.getenv('METADATA_PATH', '/app/data/metadata.csv')
EPOCHS = int(os.getenv('EPOCHS', 1))
OPTIMIZER = os.getenv('OPTIMIZER', 'adam')
LR = float(os.getenv('LR', 0.005))
DROPOUT = float(os.getenv('DROPOUT', 0.5))
ACTIVATION = os.getenv('ACTIVATION', 'relu')
LOG_METRICS = int(os.getenv('LOG_METRICS', 0))
if LOG_METRICS:
    print('Logging metrics!')
DL_BATCH_SIZE = int(os.getenv('DL_BATCH_SIZE', 16))

MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/best_model')
LB_PATH = os.getenv('LB_PATH', '/app/model/lb.pkl')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SAP AI Core connection for logging metrics
if LOG_METRICS:
    aic_connection = Tracking()
else:
    aic_connection = None

# Load datasets
emb_df = pd.read_csv(DATA_PATH)
print(emb_df.columns)
metadata_df = pd.read_csv(METADATA_PATH, index_col=0)
metadata_df['image_path'] = metadata_df['image_path'].map(lambda x: x.split('./data/')[1].strip() if type(x) == str else x)
print(metadata_df.columns)
df = emb_df.merge(metadata_df[['pose_name_l3', 'pose_name_l2', 'pose_name_l1', 'image_path']], right_on='image_path', left_on='blob_name').drop(columns=['blob_name'])

# Preprocess embeddings
X = df[emb_df.drop(columns=['blob_name']).columns].values
X_scaled = process_landmark_embeddings(X)
y_l3 = df['pose_name_l3'].values
y_l2 = df['pose_name_l2'].values
y_l1 = df['pose_name_l1'].values

# Create train-test split
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10
X_train, X_test_no_val, y_train, y_test_no_val, ind_train, ind_test_no_val = train_test_split(X_scaled, y_l3, df.index, test_size=1-train_ratio)
# test is now 10% of the initial data set
# validation is now 15% of the initial data set
X_val, X_test, y_val, y_test, ind_val, ind_test = train_test_split(X_test_no_val, y_test_no_val, ind_test_no_val, test_size=test_ratio/(test_ratio+validation_ratio))

lb = LabelBinarizer()
y_train_dummy = lb.fit_transform(y_train)
y_val_dummy = lb.fit_transform(y_val)
y_test_dummy = lb.transform(y_test)
y_test_no_val_dummy = lb.transform(y_test_no_val)

with open(LB_PATH, 'wb') as f:
  pickle.dump(lb, f)
#
# Metric Logging: Basic 
if LOG_METRICS:
    aic_connection.log_metrics(
        metrics = [
            Metric(name= "n_train_samples", value=float(X_train.shape[0]), timestamp=datetime.datetime.now(datetime.timezone.utc)),
            Metric(name= "device", value=DEVICE, timestamp=datetime.datetime.now(datetime.timezone.utc)),
        ]
    )

 # train
print(f'Training model for {EPOCHS} epochs...')
model = train(X_train, y_train_dummy, X_val, y_val_dummy, DEVICE, EPOCHS, DROPOUT, ACTIVATION, OPTIMIZER, LR, DL_BATCH_SIZE, aic_connection=aic_connection)
print(f'Training complete!')
# Save model
torch.save(model, MODEL_PATH)