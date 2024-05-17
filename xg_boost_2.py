import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
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

# Initialize the XGBClassifier with initial parameters
model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=5,
                      min_child_weight=1, gamma=0, subsample=0.8,
                      colsample_bytree=0.8, objective='multi:softprob',
                      nthread=4, seed=27)

# Fit the model to the training data
model.fit(X_train, y_train)

# Feature selection using model importance
selection = SelectFromModel(model, threshold='median', prefit=True)
select_X_train = selection.transform(X_train)

# Train model
selection_model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=5,
                                min_child_weight=1, gamma=0, subsample=0.8,
                                colsample_bytree=0.8, objective='multi:softprob',
                                nthread=4, seed=27)
selection_model.fit(select_X_train, y_train)

# Evaluate model
select_X_test = selection.transform(X_test)
y_pred = selection_model.predict(select_X_test)
predictions = [round(value) for value in y_pred]

# Evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Compute log loss
y_proba = selection_model.predict_proba(select_X_test)
logloss = log_loss(y_test, y_proba)
print("Log Loss: %.2f" % (logloss))

# Plot learning curves
train_sizes, train_scores, test_scores = learning_curve(selection_model, select_X_train, y_train, cv=5, scoring='neg_log_loss', n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curves
plt.subplots(1, figsize=(10,10))
plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Negative Log Loss"), plt.legend(loc="best")
plt.tight_layout()
plt.show()
