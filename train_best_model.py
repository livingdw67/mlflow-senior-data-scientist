
import mlflow
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# Start an MLflow run
with mlflow.start_run():
    # Initialize a DummyClassifier as a placeholder for the model with unspecified architecture
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate F1 score
    f1 = f1_score(y_test, y_pred)
    
    # Log the F1 score to MLflow
    mlflow.log_metric("f1_score", f1)
    
    print(f"F1 Score: {f1}")
