import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from data_import import IoT_data
#from imblearn.over_sampling import SMOTE
from collections import Counter

# Separating features and targets
X = IoT_data.drop(columns=["type"])
y = IoT_data["type"]

#print(X.shape)
#print(y.shape)
#print(X.columns)
#print(y.head())
#print(X.dtypes)

# Conversion of the data from strings to numerical

Categorical_Columns = ["temp_condition", "device_type", "door_state", "sphone_signal", "light_status"]
for column in Categorical_Columns:
    X[column] = X[column].astype(str).str.strip()

    # Cleaning the sphone_signal feature as it is messy

X["sphone_signal"] = X["sphone_signal"].astype(str).str.strip().str.lower()
X["sphone_signal"] = X["sphone_signal"].replace({
    "0": "false",
    "1": "true"
})

X = pd.get_dummies(X, columns=Categorical_Columns)

bool_columns = X.select_dtypes(include=["bool"]).columns
X[bool_columns] = X[bool_columns].astype(int)

    # NaN to 0

X = X.fillna(0)

#print(X.isnull().sum().sum())

#print(X.shape)
#print(X.columns)
#print(X.head())
#print(X.dtypes)

#X.to_csv("X.csv", index=False)

Label_Encoder = LabelEncoder()
y = Label_Encoder.fit_transform(y)

print(y[:10])
print(Label_Encoder.classes_)

# Splitting the data into train and test sets

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in skf.split(X, y):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    # Data scaling

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Data balancing

    #smote = SMOTE(random_state=42)
    #X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

#print(Counter(y_train))
#print(Counter(y_train_balanced))