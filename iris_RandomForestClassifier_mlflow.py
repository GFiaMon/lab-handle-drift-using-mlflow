import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

# Start an MLflow run
with mlflow.start_run(run_name="random-forest-v1") as run:
    
    # mlflow.tensorflow.autolog()
    
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
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Log the model using scikit-learn flavor
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=None  # We'll register via UI first
    )
    
    # Store run_id for later use
    run_id_v1 = run.info.run_id
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"Run ID: {run_id_v1}")
    print(f"\nMetrics:")
    print(f"  - Accuracy:  {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1 Score:  {f1:.4f}")
    print("\n✓ Model logged to MLflow")
    print(f"\n👉 View this run in the UI: http://127.0.0.1:5000")


# ----------------------------------------------
# Note: If you see warnings like the one below, it means the model was logged without a signature and input example.
# The next script addresses this by adding them when logging the model.
# 2025/10/29 18:43:04 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.