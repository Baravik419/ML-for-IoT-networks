import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import joblib
import time

from data_preprocessing import generate_folds

def train_model(use_smote=True, chosen_fold_number=None):

    start_total = time.perf_counter()

    folds, Label_Encoder, X_columns = generate_folds(use_smote=use_smote, chosen_fold_number=chosen_fold_number)

    accuracies = []

    best_accuracy = -1
    best_model = None
    best_scaler = None
    best_fold_number = None
    best_y_test = None
    best_y_pred = None

    if chosen_fold_number is None:
        folds_to_run = folds
    else:
        folds_to_run = [fold for fold in folds if fold["fold_number"] == chosen_fold_number]

        if not folds_to_run:
            raise ValueError(f"Fold number {chosen_fold_number} not found.")

    for fold in folds_to_run:
        fold_number = fold["fold_number"]
        X_train = fold["X_train"]
        y_train = fold["y_train"]
        X_test = fold["X_test"]
        y_test = fold["y_test"]
        scaler = fold["scaler"]

        start_learning = time.perf_counter()
        Light_GBM_Model = LGBMClassifier(random_state=42, verbose=1)
        Light_GBM_Model.fit(X_train, y_train) # Starting to learn
        end_learning = time.perf_counter()
        print(f"Fold {fold_number} learning time: {end_learning - start_learning:1f} s")

        start_prediction = time.perf_counter()
        y_pred = Light_GBM_Model.predict(X_test) # Creates prognosis
        accuracy = accuracy_score(y_test, y_pred)
        end_prediction = time.perf_counter()
        print(f"Fold {fold_number} prediction time: {end_prediction - start_prediction:1f} s")

        accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = Light_GBM_Model
            best_scaler = scaler
            best_fold_number = fold_number
            best_y_test = y_test
            best_y_pred = y_pred

    print("\nAll fold accuracies: ", accuracies)
    print("Mean accuracy: ", np.mean(accuracies))
    print("Best accuracy: ", best_accuracy)
    print("Best fold number: ", best_fold_number)

    end_total = time.perf_counter()
    print(f"Total training time: {end_total - start_total:1f} s")

    return {
        "accuracies": accuracies,
        "mean_accuracy": np.mean(accuracies),
        "best_accuracy": best_accuracy,
        "best_model": best_model,
        "best_scaler": best_scaler,
        "best_fold_number": best_fold_number,
        "best_y_test": best_y_test,
        "best_y_pred": best_y_pred,
        "Label_Encoder": Label_Encoder,
        "X_columns": X_columns
    }

if __name__ == "__main__":

    start_total = time.perf_counter()

    results = train_model(use_smote=True, chosen_fold_number=None)

    # Exporting
    os.makedirs("LightGBM_Model (5 fold)", exist_ok=True)

    joblib.dump(results["best_model"], "LightGBM_Model (5 fold)/LightGBM_Model.pkl")
    joblib.dump(results["best_scaler"], "LightGBM_Model (5 fold)/LGBM_scaler.pkl")
    joblib.dump(results["Label_Encoder"], "LightGBM_Model (5 fold)/LGBM_Label_Encoder.pkl")
    joblib.dump(results["X_columns"], "LightGBM_Model (5 fold)/LGBM_X_columns.pkl")

    print(f"\nBest model from fold {results['best_fold_number']} saved!")

    report_text = []
    report_text.append(f"Best fold: {results['best_fold_number']}")
    report_text.append(f"Best accuracy: {results['best_accuracy']:.6f}")
    report_text.append(f"Mean accuracy: {results['mean_accuracy']:.6f}")
    report_text.append("")

    report_text.append("All fold accuracies:")
    for i, acc in enumerate(results["accuracies"], start=1):
        report_text.append(f"Fold {i}: {acc:.6f}")
    report_text.append("")

    report_text.append(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(results['best_y_test'], results['best_y_pred']):.6f}"
    )
    report_text.append("")

    report_text.append("Classification report:")
    report_text.append(
        classification_report(
            results["best_y_test"],
            results["best_y_pred"],
            target_names=results["Label_Encoder"].classes_
        )
    )
    report_text.append("")

    report_text.append("Confusion matrix:")
    report_text.append(str(confusion_matrix(results["best_y_test"], results["best_y_pred"])))
    report_text.append("")

    report_text.append("Feature importance:")
    feature_importance = pd.DataFrame({
        "feature": results["X_columns"],
        "importance": results["best_model"].feature_importances_
    }).sort_values("importance", ascending=False)
    report_text.append(feature_importance.to_string(index=False))
    report_text.append("")


    with open("LightGBM_Model (5 fold)/LGBM_metrics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_text))

    end_total = time.perf_counter()
    print(f"Total code running time: {end_total - start_total:1f} s")