from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data_manipulation import X_train_balanced, y_train_balanced, X_test_scaled, y_test

Random_Forest_Model = RandomForestClassifier(random_state=42, verbose=1)
Random_Forest_Model.fit(X_train_balanced, y_train_balanced) # Starting to learn

y_pred = Random_Forest_Model.predict(X_test_scaled) # Creates prognosis
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)