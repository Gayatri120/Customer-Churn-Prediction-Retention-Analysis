import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import seaborn as sns

df = None  # global DataFrame
log_model, rf_model = None, None  # ML models
label_encoders = {}  # store encoders for categorical columns

# Load Excel
def load_file():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    if file_path:
        try:
            df = pd.read_excel(file_path)
            messagebox.showinfo("Success", "Customer Churn File Loaded Successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot read file\n{e}")

# Clean data
def clean_data():
    global df
    if df is None:
        messagebox.showerror("Error", "Load a file first!")
        return
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    after = df.shape[0]
    messagebox.showinfo("Cleaning Done", f"Removed {before - after} rows.")

# Analyze data
def analyze_data():
    global df
    if df is None:
        messagebox.showerror("Error", "Load a file first!")
        return
    try:
        total_customers = df['Customer_ID'].nunique()
        churned = df['Churned'].sum()
        active = total_customers - churned
        avg_usage = df['Monthly_Usage'].mean()
        avg_satisfaction = df['Customer_Satisfaction_Score'].mean()

        result.set(f"Total Customers: {total_customers}\n"
                   f"Active Customers: {active}\n"
                   f"Churned Customers: {churned}\n"
                   f"Avg Monthly Usage: {avg_usage:.2f}\n"
                   f"Avg Satisfaction Score: {avg_satisfaction:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Analysis failed\n{e}")

# Create charts
def create_charts():
    global df
    if df is None:
        messagebox.showerror("Error", "Load a file first!")
        return
    try:
        # Churn distribution
        df['Churned'].value_counts().plot(kind='bar', title="Churn Distribution")
        plt.xlabel("Churn (0=Active, 1=Churned)")
        plt.ylabel("Count")
        plt.show()

        # Subscription type breakdown
        df['Type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title("Subscription Type Distribution")
        plt.ylabel("")
        plt.show()

        # Region-wise churn
        churn_by_region = df.groupby('Region')['Churned'].mean()
        churn_by_region.plot(kind='bar', title="Churn Rate by Region")
        plt.ylabel("Churn Rate")
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Charts failed\n{e}")

# Train ML Models
def train_models():
    global df, log_model, rf_model, label_encoders
    if df is None:
        messagebox.showerror("Error", "Load a file first!")
        return
    try:
        data = df.copy()

        # Encode categorical columns
        for col in ['Type', 'Region']:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

        X = data[['Monthly_Usage', 'Total_Payments', 'Complaint_Count',
                  'Customer_Satisfaction_Score', 'Type', 'Region']]
        y = data['Churned']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Logistic Regression
        log_model = LogisticRegression(max_iter=1000)
        log_model.fit(X_train, y_train)
        log_preds = log_model.predict(X_test)

        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)

        # Evaluation
        acc_log = accuracy_score(y_test, log_preds)
        acc_rf = accuracy_score(y_test, rf_preds)

        cm = confusion_matrix(y_test, rf_preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Random Forest Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        result.set(f"Models Trained!\n"
                   f"Logistic Regression Accuracy: {acc_log:.2f}\n"
                   f"Random Forest Accuracy: {acc_rf:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Model training failed\n{e}")

# ---------------- GUI ----------------
root = Tk()
root.title(" Customer Churn Prediction System")
root.geometry("550x450")

Button(root, text=" Load File", command=load_file, width=30).pack(pady=5)
Button(root, text=" Clean Data", command=clean_data, width=30).pack(pady=5)
Button(root, text=" Analyze Data", command=analyze_data, width=30).pack(pady=5)
Button(root, text=" Create Charts", command=create_charts, width=30).pack(pady=5)
Button(root, text=" Train ML Models", command=train_models, width=30, bg="lightblue").pack(pady=5)

result = StringVar()
Label(root, textvariable=result, bg="white", width=70, height=8, anchor="w", justify=LEFT).pack(pady=10)

root.mainloop()

