#1. Load the Data import pandas as pd
df=pd.read_csv("C:\\Users\\srike\\OneDrive\\Documents\\Desktop\\reduced_200_rows.csv") df

#2. Data Exploration and preprocessing
# Display the first 5 rows df.head()
# Show dataset information (column types, non-null counts) df.info()
# Check for missing/null values df.isnull().sum()

#3. Statistical Anomaly Detection (Z-Score Method)
import matplotlib.pyplot as plt
# Calculate mean and standard deviation mean = df['Transaction_Amount'].mean() std = df['Transaction_Amount'].std()
# Compute Z-score for each transaction
df['Z_score'] = (df['Transaction_Amount'] - mean) / std # Define anomaly threshold
threshold = 2
# Flag anomalies
df['Anomaly'] = df['Z_score'].abs() > threshold # Extract anomalies
anomalies = df[df['Anomaly'] == True]
 
# Show anomalies
print(f"Total anomalies detected: {len(anomalies)}") display(anomalies[['Transaction_ID', 'Transaction_Amount', 'Z_score']]) # Visualize Z-score based anomalies
plt.figure(figsize=(10, 6))
plt.scatter(df[~df['Anomaly']].index,df[~df['Anomaly']]['Transaction_Amount'], label='Normal', color='blue', alpha=0.6)
plt.scatter(anomalies.index, anomalies['Transaction_Amount'], label='Anomaly', color='red', marker='x', s=100)
plt.title('Z-score Based Anomaly Detection on Transaction Amount') plt.xlabel('Transaction Index')
plt.ylabel('Transaction Amount') plt.legend()
plt.grid(True) plt.show()

#4. Machine Learning Based Detection
5.3	Isolation Forest
from sklearn.ensemble import IsolationForest from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score # Select features
features=['Transaction_Amount','Average_Transaction_Amount','Frequency_of_Transaction’] X = df[features]
# Normalize the data scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Train the Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42) df['Anomaly_IF'] = iso_forest.fit_predict(X_scaled)
 
# Convert prediction values (1=normal, -1=anomaly) df['Anomaly_IF'] = df['Anomaly_IF'].map({1: 0, -1: 1}) # Plot anomalies detected by Isolation Forest
plt.scatter(df['Transaction_Amount'], df['Frequency_of_Transactions'], c=df['Anomaly_IF'], cmap='coolwarm', alpha=0.6)
plt.xlabel('Transaction Amount') plt.ylabel('Frequency of Transactions') plt.title('Anomalies Detected by Isolation Forest') plt.show()
# Evaluation (optional if 'Anomaly' column exists) if 'Anomaly' in df.columns:
print("Isolation Forest Evaluation:") print(classification_report(df['Anomaly'], df['Anomaly_IF'])) print("ROC-AUC:", roc_auc_score(df['Anomaly'], df['Anomaly_IF']))

5.4	Random Forest Classifier
from sklearn.model_selection import train_test_split from sklearn.ensemble import RandomForestClassifier import matplotlib.pyplot as plt
# Define target
y = df['Anomaly']
# Scale features again scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Train-test split
X_train, X_test, y_train, y_test = train_test_split( X_scaled, y, test_size=0.2, random_state=42, stratify=y)
# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42) rf_model.fit(X_train, y_train)
 
# Predictions
y_pred = rf_model.predict(X_test) # Evaluation
print("Classification Report:") print(classification_report(y_test, y_pred)) # ROC-AUC
if len(np.unique(y_test)) == 2:
y_probs = rf_model.predict_proba(X_test)[:, 1] print("ROC-AUC Score:", roc_auc_score(y_test, y_probs))
else:
print("ROC-AUC cannot be calculated: only one class present in y_test") # Feature Importances
importances = rf_model.feature_importances_ feature_names = features
# Plot Feature Importances plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='teal') plt.xlabel('Importance')
plt.title('Random Forest Feature Importances') plt.grid(True)
plt.tight_layout() plt.show()

#5. Model Evaluation
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc # Train Isolation Forest on train set
iso_forest = IsolationForest(contamination=0.05, random_state=42) iso_forest.fit(X_train)
# Get anomaly scores
iso_scores = -iso_forest.decision_function(X_test)
 
iso_preds = iso_forest.predict(X_test) iso_preds = np.where(iso_preds == -1, 1, 0) # Metrics
roc_auc = roc_auc_score(y_test, iso_scores)
precision, recall, _ = precision_recall_curve(y_test, iso_scores) pr_auc = auc(recall, precision)
f1 = f1_score(y_test, iso_preds) # Print results
print("Isolation Forest Evaluation:") print(f"ROC-AUC Score: {roc_auc:.4f}") print(f"PR-AUC Score: {pr_auc:.4f}") print(f"F1 Score: {f1:.4f}")
# Plot Precision-Recall Curve
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})') plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Isolation Forest') plt.legend()
plt.grid() plt.show()

#6. User Interaction (Manual CLI Prediction)
# Load dataset again
data=pd.read_csv(r"C:\Users\srike\OneDrive\Pictures\Documents\MINI PROJECT\reduced_200_rows.csv")
features=['Transaction_Amount','Average_Transaction_Amount','Frequency_of_Transaction’] data = data[features].dropna()
# Train Isolation Forest contamination_ratio = 4 / len(data)
model = IsolationForest(contamination=contamination_ratio, random_state=42) model.fit(data[features])
 
# Predict anomalies
data['Prediction'] = model.predict(data[features]) print("=== Anomaly Detection System ===")
print(f"Model detected {len(data[data['Prediction'] == -1])} anomalies out of {len(data)} total transactions.\n")
print("Detected Anomalous Transactions:\n") print(data[data['Prediction'] == -1][features])
print("\nYou can test values manually below. Type 'exit' at any time to quit.\n") # CLI-based manual prediction
while True: try:
ta = input("Transaction Amount: ") if ta.lower() == 'exit':
break
ata = input("Average Transaction Amount: ") if ata.lower() == 'exit':
break
freq = input("Frequency of Transactions: ") if freq.lower() == 'exit':
break
input_data = pd.DataFrame([[float(ta), float(ata), float(freq)]], columns=features) prediction = model.predict(input_data)[0]

if prediction == -1:
print("⚠️ Anomalous Transaction Detected!\n") else:
print("✅ Transaction is Normal.\n") except ValueError:
print("❌ Invalid input. Please enter numeric values only.\n")
