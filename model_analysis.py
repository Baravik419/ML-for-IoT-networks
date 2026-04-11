import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from data_manipulation import X_test_scaled, y_test

# Random Forest Model with SMOTE

RFMS = joblib.load("Random_Forest (1 fold)/Random_Forest_Model_SMOTE.pkl")
RFMS_scaler = joblib.load("Random_Forest (1 fold)/RFMS_scaler.pkl")
RFMS_Label_Encoder = joblib.load("Random_Forest (1 fold)/RFMS_Label_Encoder.pkl")
RFMS_X_columns = joblib.load("Random_Forest (1 fold)/RFMS_X_columns.pkl")

y_RFMS_pred = RFMS.predict(X_test_scaled)

print("Accuracy: ", accuracy_score(y_test, y_RFMS_pred))
print("Balanced Accuracy: ", balanced_accuracy_score(y_test, y_RFMS_pred))
print("Classification Report: ", classification_report(y_test, y_RFMS_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_RFMS_pred))

# Random Forest Model

RFM = joblib.load("Random_Forest/Random_Forest_Model.pkl")
RFM_scaler = joblib.load("Random_Forest/RFM_scaler.pkl")
RFM_Label_Encoder = joblib.load("Random_Forest/RFM_Label_Encoder.pkl")
RFM_X_columns = joblib.load("Random_Forest/RFM_X_columns.pkl")

y_RFM_pred = RFM.predict(X_test_scaled)

print("Accuracy: ", accuracy_score(y_test, y_RFM_pred))
print("Balanced Accuracy: ", balanced_accuracy_score(y_test, y_RFM_pred))
print("Classification Report: ", classification_report(y_test, y_RFM_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_RFM_pred))