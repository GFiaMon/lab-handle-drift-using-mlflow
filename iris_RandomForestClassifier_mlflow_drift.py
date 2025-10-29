import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
import pandas as pd
import json

# import evidently
# print(f"Evidently version: {evidently.__version__}")
# print(f"Evidently path: {evidently.__file__}")

# from evidently.report import Report               # <-- Only for version < 0.7.x from version 0.7.x use:
from evidently import Report                        # <-- Only for version >= 0.7.x     

# from evidently.metric_preset import DataDriftPreset       # <-- Only for version < 0.7.x from version 0.7.x use:
from evidently.presets import DataDriftPreset        # <-- Only for version >= 0.7.x 

from datetime import datetime

# Create timestamp for the run name
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

run_name = f"random-forest-v2-drifted-{timestamp}"


# ----------------------------------------------
# Set the tracking URI to connect to your local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Create or set experiment
experiment_name = "iris-classification-drift-handling"
mlflow.set_experiment(experiment_name)

print(f"✓ Experiment '{experiment_name}' is ready")
print(f"View it at: http://127.0.0.1:5000")

# ----------------------------------------------


# ----------------------------------------------

# Load data and prep
iris_data = load_iris(as_frame=True)
df = iris_data.frame
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ----------------------------------------------
# ----------------------------------------------

# Start an MLflow run
with mlflow.start_run(run_name=run_name) as run:
    
    mlflow.sklearn.autolog()
    
    # 1. training code

    # Define hyperparameters
    n_estimators = 100
    max_depth = 5
    random_state = 42
    
    # Log parameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("dataset", "iris")
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    
    # 2. Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # 3. Model logging
        # 1. Create signature (analyzes existing data)
        # Infer model signature and create input example
    signature = infer_signature(X_train, model.predict(X_train))
        # 2. Use first row as example
    input_example = X_train.iloc[:1]
        # Log the model using scikit-learn flavor
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,                # <-- OPTIONAL: Model signature for input/output schema to avoid having warnings.
        input_example=input_example,        # <-- OPTIONAL: Model signature for input/output schema to avoid having warnings.
        registered_model_name=None  # We'll register via UI first
    )
    
    # # Store run_id for later use
    run_id = run.info.run_id
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"Run ID: {run_id}")
    print(f"\nMetrics:")
    print(f"  - Accuracy:  {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1 Score:  {f1:.4f}")

# print('\nLogging drift report to Mlflow...\n')
# with mlflow.start_run(run_id=run_id):

    # 4. Drift detection
    # ----------------------------------------------
    # Introduce data drift by modifying the 'sepal length (cm)' feature

        # drift simulation code
    X_drifted = X_test.copy()
    X_drifted["sepal length (cm)"] += np.random.normal(loc=2.0, scale=0.3, size=len(X_drifted))

        # Create and run drift report (UPDATED for v0.7.15)
    report = Report(metrics=[DataDriftPreset()])

    snapshot = report.run(current_data=X_drifted, reference_data=X_train)  # <-- UPDATED for v0.7.x Only way it works!
    snapshot.save_html("drift_report.html")                                # <-- need to save from the snapshot object      

    # 5. Log drift report to SAME run
    mlflow.log_artifact("drift_report.html", "drift_reports")
    print("✓ Drift report saved to 'drift_report.html' and logged to MLflow")

            # Test model performance on drifted data
    print("\nStep 6: Evaluating model on drifted data...")
    y_pred_drifted = model.predict(X_drifted)
    accuracy_drifted = accuracy_score(y_test, y_pred_drifted)

    print(f"Model accuracy on original data: {accuracy:.4f}")
    print(f"Model accuracy on drifted data:  {accuracy_drifted:.4f}")
    print(f"Performance change: {accuracy_drifted - accuracy:+.4f}")

    # 6. Log drifted performance metrics
    mlflow.log_metric("accuracy_drifted", accuracy_drifted)
    mlflow.log_metric("performance_change", accuracy_drifted - accuracy)

# ------------------------------------------------------------------------------------------    
# Add this CLI output section
print("\n" + "="*50)
print("DRIFT DETECTION RESULTS")
print("="*50)

# Convert snapshot to JSON string and parse
json_str = snapshot.json()
drift_results = json.loads(json_str)
drift_metrics = drift_results.get("metrics", [])

# Parse the metrics
drifted_count = 0
total_features = 0
feature_drifts = []

for metric in drift_metrics:
    metric_id = metric.get("metric_id", "")
    
    if "DriftedColumnsCount" in metric_id:
        # This is the summary metric
        drifted_count = int(metric["value"]["count"])
        drift_share = metric["value"]["share"]
        total_features = int(drifted_count / drift_share) if drift_share > 0 else 0
        
    elif "ValueDrift" in metric_id:
        # This is a feature-level drift metric
        feature_name = metric_id.split("column=")[1].replace(")", "")
        drift_value = metric["value"]
        # Typically, low p-value (< 0.05) means drift detected
        is_drifted = drift_value < 0.05
        feature_drifts.append({
            "feature": feature_name,
            "drift_value": drift_value,
            "drifted": is_drifted
        })

# Print results
print(f"📊 DRIFT SUMMARY:")
print(f"   Drifted Features: {drifted_count}/{total_features} ({drift_share:.1%})")
print(f"   Dataset Drift: {'🚨 DETECTED' if drifted_count > 0 else '✅ No drift'}")

print(f"\n🔍 FEATURE-LEVEL ANALYSIS:")
for feature in feature_drifts:
    status = "🚨 DRIFT" if feature["drifted"] else "✅ No drift"
    print(f"   - {feature['feature']:20} {status} (p-value: {feature['drift_value']:.6f})")

print(f"\n📝 INTERPRETATION:")
print(f"   • p-value < 0.05: Significant drift detected")
print(f"   • p-value >= 0.05: No significant drift")
print(f"   • {drifted_count} feature(s) showing statistical drift")

print("✓ Drift report saved to 'drift_report.html' and logged to MLflow")
print(f"👉 View detailed report: http://127.0.0.1:5000")


    
    
 # ==============================================================================================================================

    # THIS WAS THE OLD CODE FOR EVIDENTLY >= v0.6.x WHICH IS NO LONGER WORKING IN v0.7.x +

    # report.run(current_data=X_drifted, reference_data=X_train)  # <-- WE WILL TRY IT SAVING TO A VARIABLE FIRST

    # # Save to your lab directory and log to MLflow
    # report.save_html("drift_report.html")
    # mlflow.log_artifact("drift_report.html", "drift_reports")

    # # Alternative save method : ALSO NOT WORKING IN v0.7.15
    # with open("drift_report.html", "w") as f:
    #     f.write(report.get_html())



