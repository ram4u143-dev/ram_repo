import pandas as pd
from imblearn.over_sampling import RamdomOverSampler
from sklearn.model_selection import train_test_split
from skalearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from google.cloud import storage
import json
from google.cloud import bigquery
from datatime import datetime
from sklearn.pipeline import Pipeline,make_pipeline


storage_client = storage.Client()
bucket = storage_client.bucket("ml_bucket_first_01")

def load_data(path):
    return pd.read_csv(path, sep=";")

def encode_categorical(df,categorical_features):
    le = LabelEncoder()
    df[categorical_cols]= df[categorical_cols].apply(lambda col: le.fit_transform(col))
    return df

def preprocess_data(df):
    x = df.drop("y", axis=1)
    y = df["y"].apply(lambda val: 1 if val == "yes" else 0)
    
    sc= StandardScaler()
    pd.DataFrame(sc.fit_transform(x), columns=x.columns)
    return x, y

def bucket_pdays(pdays):
    if pdays == 999:
        return 0
    elif pdays < 30:
        return 1
    elif pdays < 60:
        return 2
    else:
        return 3
    

def apply_bucketing(df):
    df["pdays"] = df["pdays"].apply(bucket_pdays)
    df = df.drop("pdays", axis=1)
    df = df.drop("duration", axis=1)
    return df


def train_model(model_name,x_train,y_train):
    if model_name == "logistic":
        model = LogisticRegression(random_state=42)
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "knn":
        model = KNeighborsClassifier()
    elif model_name == "xgboost":
        model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    else:
        raise ValueError("Unsupported model type")

    Pipeline = Pipeline(model)
    Pipeline.fit(x_train, y_train)
    return Pipeline

def get_classification_report(pipeliney_true, y_pred):
    y_pred = pipeline.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

def save_model_artifact(model_name,pipline):
    artifact_name = model_name +'model.joblib'
    dump(pipline,artifact_name)


def load_model_artifact(file_name):
    blob = bucket.blob("ml-artifacts/" + file_name)
    blob.download_to_filename(file_name)
    return load(file_name)

def write_metrics_to_bigquery(algo_name, training_time, model_metrics):
    client = bigquery.Client()
    table_id = "bustracking-467614.ml_ops.bank_campaign_model_metrics"
    table = bigquery.Table(table_id)

    row = {"algo_name": algo_name, "training_time": training_time.strftime('%Y-%m-%d %H:%M:%S'), "model_metrics": json.dumps(model_metrics)}
    errors = client.insert_rows_json(table, [row])

    if errors == []:
        print("Metrics inserted successfully into BigQuery.")
    else:
        print("Error inserting metrics into BigQuery:", errors)

def write_metrics_to_bigquery(algo_name,training_time,model_metrics):
    client = bigquery.Client()
    table_id = "bustracking-467614.ml_ops.bank_campaign_model_metrics"
    table = bigquery.Table(table_id)

    row = {"algo_name": algo_name, "training_time": training_time.strftime('%Y-%m-%d %H:%M:%S'), "model_metrics": json.dumps(model_metrics)}
    errors = client.insert_rows_json(table, [row])

    if errors == []:
        print("Metrics inserted successfully into BigQuery.")
    else:
        print("Error inserting metrics into BigQuery:", errors)

def main():
    data_path = "gs://ml_bucket_first_01/bank-campaign-training-data.csv"
    model_name = "xgboost"

    df = load_data(data_path)

    categorical_cols = ['jobs','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
    df = encode_categorical(df, categorical_cols)
    df = apply_bucketing(df)

    x, y = preprocess_features(df)

    oversampler= RandomForestClassifier(random_state=42)
    x_resampled,y_resampled = oversampler.fit_resample(x,y)

    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)


    pipeline = train_model(model_name,x_train,y_train)
    accuracy_metrics = get_classification_report(pipeline,x_test,y_test)
    training_time = datetime.now() 
    write_metrics_to_bigquery(model_name,training_time,accuracy_metrics)
    save_model_artifact(model_name,pipeline)

if __name__ == "__main__":
    main()

   

    