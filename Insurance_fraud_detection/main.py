import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import time
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Set the page layout to wide
st.set_page_config(layout="wide")

# Streamlit App
st.title("Insurance Fraud Detection")

# Load Dataset or Sample Data
use_sample_data = st.checkbox("Use Sample Data", value=True)

if use_sample_data:
    sample_excel_path = "Worksheet in Case Study question 2.xlsx"  # Replace with your actual sample data path
    with st.spinner('🔄 Loading sample data...'):
        time.sleep(2)  # Simulate loading time for effect
        data = pd.read_excel(sample_excel_path)
        st.success('Sample data loaded successfully!')

    # Show preview of sample data
    st.write("Sample Data Preview:")
    st.dataframe(data)

    # Download button for sample data
    st.download_button(
        label="Download Sample Data",
        data=data.to_csv(index=False),
        file_name="sample_data.csv",
        mime="text/csv"
    )
else:
    uploaded_file = st.file_uploader("Upload your CSV file for fraud detection", type=["csv", "xlsx"])
    if uploaded_file is not None:
        with st.spinner('🔄 Loading your data...'):
            time.sleep(3)  # Simulate loading time for effect
            data = pd.read_excel(uploaded_file, engine="openpyxl")
        st.success('Your data has been loaded successfully!')

target_variable = "fraud_reported"

# Convert target column 'fraud_reported' to binary
data["fraud_reported"] = data["fraud_reported"].apply(lambda x: 1 if x == "Y" else 0)

# Exploratory Data Analysis (EDA) Section
categorical_columns = [
    "policy_csl", "insured_sex", "incident_type", 
    "collision_type", "incident_severity", "incident_state", 
    "incident_city", "property_damage", "police_report_available"
]

# Separate numerical and categorical columns
numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()

with st.spinner("🔄 Exploratory Data Analysis..."):
    time.sleep(2)

    st.write("## Exploratory Data Analysis")
    
    # Encode categorical variables if any
    if len(categorical_columns) > 0:
        label_encoded_data = data[categorical_columns].apply(LabelEncoder().fit_transform)
        combined_data = pd.concat([data[numerical_columns], label_encoded_data], axis=1)
    else:
        combined_data = data[numerical_columns]

    # Compute correlation matrix for the entire dataset
    st.write("### Correlation Heatmap")
    correlation_matrix = combined_data.corr()

    # Plot the heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    st.pyplot(plt)

# Data Preprocessing
with st.spinner("🔄 Data preprocessing..."):
    time.sleep(5)

    # Selected features for preprocessing
    features = [
        "months_as_customer", "age", "policy_number", "policy_deductable",
        "policy_annual_premium", "umbrella_limit", "insured_zip", "incident_hour_of_the_day", "number_of_vehicles_involved",
        "bodily_injuries", "total_claim_amount", "property_claim",
        "injury_claim", "vehicle_claim", "auto_year" ,"incident_type", "collision_type", "incident_severity"
    ]

    # Handle missing values
    imputer = SimpleImputer(strategy="most_frequent")
    data_selected = data[features]  # Select only the required features
    df = pd.DataFrame(imputer.fit_transform(data_selected), columns=features)

    # Encode categorical features (adjust columns as needed)
    categorical_columns = [
        "policy_number", "insured_zip", "auto_year" ,
        "incident_type", "collision_type", "incident_severity" # Ensure these are actual categorical columns
    ]
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_columns = [
        "months_as_customer", "age", "policy_deductable",
        "policy_annual_premium", "umbrella_limit", "incident_hour_of_the_day",
        "number_of_vehicles_involved", "bodily_injuries", "total_claim_amount"
    ]
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Display the preprocessed data
    st.write("#### Preprocessed Data")
    st.dataframe(df)

    # Prepare features (X) and target (y)
    X = df
    y = data["fraud_reported"]

    # Split the data into training and test sets
    with st.spinner('🔄 Splitting data into training and testing sets...'):
            time.sleep(3) 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Assuming X_train, X_test, y_train, y_test are already defined

def train_and_evaluate_model(model, model_name):
    with st.spinner(f'🔄 Training {model_name}...'):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, target_names=["Non-Fraud", "Fraud"], output_dict=True)
        fraud_metrics = {
            "Metric": ["Precision", "Recall", "F1-Score", "Accuracy"],
            "Value": [
                report_dict["Fraud"]["precision"],
                report_dict["Fraud"]["recall"],
                report_dict["Fraud"]["f1-score"],
                accuracy
            ]
        }
        cm = confusion_matrix(y_test, y_pred)
        return pd.DataFrame(fraud_metrics), cm, accuracy, y_pred

# Model selection dropdown
model_option = st.selectbox(
    "Select the model", ["Logistic Regression", "Random Forest"],
    format_func=lambda x: 'Select the model' if x == "" else x
)

# Initialize models based on user selection
if model_option == "Logistic Regression":
    model = LogisticRegression(random_state=42)
    model_name = "Logistic Regression"
elif model_option == "Random Forest":
    model = RandomForestClassifier(random_state=42)
    model_name = "Random Forest"

# Train the selected model
metrics, cm, accuracy, y_pred = train_and_evaluate_model(model, model_name)

# Display results for the selected model
st.write(f"### Classification Report for {model_name}")
# Convert to DataFrame and display better
st.dataframe(metrics)

st.write(f"### Confusion Matrix for {model_name}")
# Display confusion matrix as a table
st.table(pd.DataFrame(cm, columns=["Predicted Non-Fraud", "Predicted Fraud"], 
                      index=["Actual Non-Fraud", "Actual Fraud"]))


# Train both models and get predictions for Fraud vs Non-Fraud graph
model_rf = RandomForestClassifier(random_state=42)
model_lr = LogisticRegression(random_state=42)

metrics_rf, cm_rf, accuracy_rf, y_pred_rf = train_and_evaluate_model(model_rf, "Random Forest")
metrics_lr, cm_lr, accuracy_lr, y_pred_lr = train_and_evaluate_model(model_lr, "Logistic Regression")

# Create a DataFrame to show Fraud vs Non-Fraud predictions
predictions = pd.DataFrame({
    'Model': ['Logistic Regression'] * len(y_pred_lr) + ['Random Forest'] * len(y_pred_rf),
    'Prediction': list(y_pred_lr) + list(y_pred_rf)
})

# Plot Fraud vs Non-Fraud Predictions for both models
st.write("### Fraud vs Non-Fraud Predictions")
fig, ax = plt.subplots(figsize=(10, 3))  # Smaller graph size

# Count the number of Fraud and Non-Fraud predictions for each model
sns.countplot(data=predictions, x='Model', hue='Prediction', palette='Blues', ax=ax)
ax.set_title('Fraud vs Non-Fraud Predictions for Both Models')
ax.set_xlabel('Model')
ax.set_ylabel('Count')
ax.legend(title='Prediction', labels=['Non-Fraud', 'Fraud'])

st.pyplot(fig)
