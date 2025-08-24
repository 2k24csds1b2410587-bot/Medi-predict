import pandas as pd

# Load dataset
data = pd.read_csv(r"C:\Users\ISHA\Downloads\projecthealth\diabetes.csv")

# Show first 5 rows
print(data.head())

# Show info about columns
print(data.info())

# Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data preparation done!")
print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))

# ------------------ MODEL TRAINING ------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Create Random Forest model with more trees for better accuracy
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(r"C:\Users\ISHA\Desktop\confusion_matrix.png")
plt.close()

# Save trained model
joblib.dump(model, r"C:\Users\ISHA\Desktop\health_model.pkl")
print(" Model saved on Desktop as health_model.pkl")

# ------------------ USER INPUT PREDICTION ------------------
print("\nEnter your health details to check risk:")

preg = int(input("Number of pregnancies: "))
glucose = int(input("Glucose level: "))
bp = int(input("Blood pressure: "))
skin = int(input("Skin thickness: "))
insulin = int(input("Insulin level: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = int(input("Age: "))

# Prepare input
user_input = [[preg, glucose, bp, skin, insulin, bmi, dpf, age]]

# Predict
prediction = model.predict(user_input)
probability = model.predict_proba(user_input)[0][1]  # chance of diabetes

# Show result + suggestions
if prediction[0] == 1:
    print(f"\n High Risk! (Probability: {probability:.2f})")
    print("👉 Suggestions:")
    if glucose > 140:
        print("- Control your blood sugar with diet & medication.")
    if bmi > 30:
        print("- Focus on weight management through exercise.")
    if bp > 130:
        print("- Keep an eye on your blood pressure, reduce salt intake.")
    if age > 45:
        print("- Regular health checkups are strongly recommended.")
    print("- Please consult a doctor for proper guidance.")
else:
    print(f"\n✅ Low Risk (Probability: {probability:.2f})")
    print("👉 Suggestions: Maintain a healthy lifestyle with exercise, balanced diet, and regular checkups.")
