
![IronHack_Logo](https://user-images.githubusercontent.com/92721547/180665853-e52e3369-9973-4c1e-8d88-1ecef1eb8e9e.png)

# LAB |  Handling Feature Drift in Production with MLflow & Evidently

## Learning Goals

- Train and log your model using MLflow  
- Simulate feature drift with new incoming data  
- Detect and log drift reports using Evidently  
- Log drift artifacts back to MLflow for traceability  

## Prerequisites

- Familiarity with Python and machine learning  
- MLflow and Evidently installed  
- Basic understanding of model monitoring concepts  

## Step-by-Step Guide

### Step 1: Train (and Log) a Baseline Model Using MLflow  
- Install required packages:  `!pip install mlflow scikit-learn pandas evidently`
- 📦 Train and track your baseline model, `iris_RandomForestClassifier.py`, using MLflow autologging.

### Step 2: Introduce Some Drift — On Purpose  
- Simulate feature drift by modifying incoming data or feature distributions to mimic changes over time or external influences.

    - sample drift:
        ```python
        import numpy as np
        import pandas as pd

        X_drifted = X_test.copy()
        X_drifted["sepal length (cm)"] += np.random.normal(loc=2.0, scale=0.3, size=len(X_drifted))
        ```

### Step 3: Catch Drift Using Evidently  
- Use Evidently to generate drift reports that detect changes in feature distributions between baseline and current data.
    - Sample Report:
        ```python
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=X_train, current_data=X_drifted)

        report.save_html("drift_report.html")
        ```

### Step 4: Log the Drift Report  
- Save the drift report as an artifact in MLflow.  
- Later, view this report in the MLflow UI to trace drift evolution over time.

## Bonus: Automate The Process  
- ⏰ Schedule drift checks daily or weekly using workflow orchestrators like Airflow or Prefect.  
- 🚨 Set alerts if more than 30% of features exhibit significant statistical drift.  
- 🔄 Trigger retraining experiments automatically through MLflow to update your model.  
- 🛠 Update the model registry to roll back or promote new models based on drift insights.

***

## 💡 Remarks  
- Feature drift is a common and costly challenge in production ML systems.  
- MLflow helps you track your models and experiments.  
- Evidently provides powerful drift detection capabilities.  
- Automate detection, alerting, and retraining for proactive model maintenance.  
- Don’t let silent failures erode trust in your AI applications.


## Deliverables 
- Submit your modified `iris_RandomForestClassifier.py` file with all MLflow integrations.
- Provide screenshots or exported views from the MLflow UI showing experiments, metrics, and runs.

## Submission

Upon completion, add your deliverables to git. Then commit git and push your branch to the remote.