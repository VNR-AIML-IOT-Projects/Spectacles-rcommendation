import xgboost as xgb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pandas as pd

df = pd.read_csv('normalized_file.csv')
X = df.drop('face_shape', axis=1).astype('category')
y = df['face_shape']

# Encode the labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model('face_shape_classifier.json')

# Now you can use xgb_model to make predictions
y_pred = xgb_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Plotting using seaborn
plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(cm, annot=True, fmt='g') # font size

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
