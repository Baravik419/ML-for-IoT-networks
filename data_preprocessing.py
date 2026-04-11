import time

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# IoT_network_data = pd.read_csv("IoT_network_data.csv")

# Counting how many null values
# counts = IoT_network_data.count()
# percent = (counts / len(IoT_network_data)) * 100
#
# summary = pd.DataFrame({
#     "non_null_counts": counts,
#     "percent_filled": percent
# }).sort_values("percent_filled", ascending=False)
#
# print(summary)
# print(IoT_network_data.isna().sum())

# Counting how many 0 or -
# summary = pd.DataFrame({
#     "zero_or_dash": ((IoT_network_data == "0") | (IoT_network_data == "-")).sum(),
# })
#
# summary["percent"] = summary["zero_or_dash"] / len(IoT_network_data) * 100
# summary = summary.sort_values("percent", ascending=False)
#
# print(summary)

# Begging data preprocessing
def prepare_data():
    start_total = time.perf_counter()
    IoT_network_data = pd.read_csv("IoT_network_data.csv")

    # Dropping uneccessary data
    IoT_network_data = IoT_network_data.drop(columns=["label", "uid"])

    # Separating features and targets
    X = IoT_network_data.drop(columns=["type"])
    y = IoT_network_data["type"]

    #print(X.shape)
    #print(y.shape)
    #print(X.columns)
    #print(y.head())
    #print(X.dtypes)

    X["src_bytes"] = pd.to_numeric(X["src_bytes"], errors="coerce")

    Categorical_Columns = X.select_dtypes(include=["string"]).columns

    for column in Categorical_Columns:
        X[column] = X[column].astype(str).str.strip().str.lower()

    # Label encoding
    for column in Categorical_Columns:
        X[column] = X[column].astype("category").cat.codes

    # NaN to 0
    X = X.fillna(0)
    X.to_csv("X.csv", index=False)

    Label_Encoder = LabelEncoder()
    y = pd.Series(Label_Encoder.fit_transform(y), name="type")
    y.to_csv("y.csv", index=False)

    end_total = time.perf_counter()

    print(f"Total data preparation: {end_total - start_total:1f} s")

    return X, y, Label_Encoder

def generate_folds(use_smote=True, chosen_fold_number=None):

    start_total = time.perf_counter()

    start_preprocessing = time.perf_counter()
    X, y, Label_Encoder = prepare_data()
    end_preprocessing = time.perf_counter()

    print(f"Total preprocessing: {end_preprocessing - start_preprocessing:1f} s")

    # Splitting the data into train and test sets
    start_folds = time.perf_counter()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    end_folds = time.perf_counter()

    print(f"Folds generation: {end_folds - start_folds:1f} s")

    folds = []

    for fold_number, (train_index, test_index) in enumerate(skf.split(X, y), start=1):

        if chosen_fold_number is not None and fold_number != chosen_fold_number:
            continue

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # Data scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Data balancing
        if use_smote:
            start_smote = time.perf_counter()
            smote = SMOTE(random_state=42)
            X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
            end_smote = time.perf_counter()
            print(pd.Series(y_train).value_counts().sort_index())
            print(pd.Series(y_train_final).value_counts().sort_index())
            print(f"SMOTE: {end_smote - start_smote:1f} s")
        else:
            X_train_final, y_train_final = X_train_scaled, y_train

        folds.append({
            "fold_number": fold_number,
            "X_train": X_train_final,
            "y_train": y_train_final,
            "X_test": X_test_scaled,
            "y_test": y_test,
            "scaler": scaler
        })

    end_total = time.perf_counter()
    print(f"Total folds generation and balancing: {end_total - start_total:1f} s")
    return folds, Label_Encoder, X.columns.tolist()