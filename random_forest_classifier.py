import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
df = pd.read_csv('merged_file_4.csv')

# Separate features and target
X = df.drop('face_shape', axis=1)
y = df['face_shape']

# Encode the labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Calculate the square root of the number of features
max_features = int(np.sqrt(X_train.shape[1]))

# Initialize the RandomForestClassifier with initial parameters
model = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=max_features, random_state=42)

# Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'max_features': ['sqrt', None]  # Include 'sqrt' and None to compare
}

# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, cv=kfold)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and highest accuracy
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best accuracy found: {grid_search.best_score_}")

# Use the best estimator to make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {accuracy}")

# Check for underfitting or overfitting
training_accuracy = accuracy_score(y_train, best_model.predict(X_train))
validation_accuracy = grid_search.best_score_

print(f"Training Accuracy: {training_accuracy}")
print(f"Validation Accuracy: {validation_accuracy}")

if training_accuracy > validation_accuracy:
    if (training_accuracy - validation_accuracy) > 0.1: # arbitrary threshold
        print("The model may be overfitting.")
    else:
        print("The model is probably fitting well.")
else:
    if validation_accuracy - training_accuracy > 0.1: # arbitrary threshold
        print("The model may be underfitting.")
    else:
        print("The model is probably fitting well.")

# Save the model for future use
# best_model_filename = 'random_forest_model_with_sqrt_feature.model'
# best_model.save_model(best_model_filename)
# print(f"Model saved as {best_model_filename}")
