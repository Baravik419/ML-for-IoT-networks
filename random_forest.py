from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

from data_manipulation import X_train_balanced, y_train_balanced, X_test_scaled, y_test, scaler, Label_Encoder, X

Random_Forest_Model = RandomForestClassifier(random_state=42, verbose=1)
Random_Forest_Model.fit(X_train_balanced, y_train_balanced) # Starting to learn

y_pred = Random_Forest_Model.predict(X_test_scaled) # Creates prognosis
accuracy = accuracy_score(y_test, y_pred)

# Exporting
joblib.dump(Random_Forest_Model, "Random_Forest/Random_Forest_Model.pkl")
joblib.dump(scaler, "Random_Forest/RFM_scaler.pkl")
joblib.dump(Label_Encoder, "Random_Forest/RFM_Label_Encoder.pkl")
joblib.dump(X.columns.tolist(), "Random_Forest/RFM_X_columns.pkl")

print(accuracy)