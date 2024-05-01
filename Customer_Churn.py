# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.combine import SMOTEENN

# Load dataset
df = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\keerthi\\afame\\Churn_Modelling.csv")

# Drop unnecessary columns
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
df = df.drop(columns=columns_to_drop)

# Encode categorical variables
encoded_df = pd.get_dummies(df, columns=['Gender', 'Geography'])

# Split data into features and target
X = encoded_df.drop(columns=['Exited'])
y = encoded_df['Exited']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implement SMOTEENN for combined sampling
smoteenn = SMOTEENN()
X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train_scaled, y_train)

# Initialize and train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions on test set
y_pred_rf = rf_classifier.predict(X_test_scaled)

# Evaluate Random Forest Classifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf}')
print(classification_report(y_test, y_pred_rf))

# Visualize feature importances
feature_importances = rf_classifier.feature_importances_
feature_names = X.columns

# Create DataFrame for feature importances
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(8, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()
