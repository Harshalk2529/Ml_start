import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
data = pd.read_csv(r'C:\Users\Hi\Desktop\Color-Detection-OpenCV-main\colors.csv', names=["color_name", "hex", "R", "G", "B"], header=None)

# Extract features (RGB values) and labels (color names)
X = data[["R", "G", "B"]].values
y = data["color_name"].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize k-NN classifier
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Print accuracy and classification report
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Save the model to a file
joblib.dump(model, 'color_classifier_knn.pkl')
