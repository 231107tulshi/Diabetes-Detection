import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset (use double slashes or forward slashes in the path)
data = pd.read_csv('C:/Users/User/VS code/diabetes_detection/diabetes.csv')

# Clean the dataset by converting all columns to numeric and dropping rows with missing values
data = data.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric values to NaN
data = data.dropna()  # Drop rows with missing values

# Split the data into features (X) and target (y)
X = data.drop(columns='Outcome')  # 'Outcome' is the target variable
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Function to predict diabetes based on input
def predict_diabetes(input_data):
    # Ensure input data matches the format of X_train
    input_data = pd.DataFrame([input_data], columns=X.columns)
    prediction = model.predict(input_data)
    return 'Diabetes Detected' if prediction[0] == 1 else 'No Diabetes Detected'


# Example prediction (you can remove or modify this part as needed)
input_data = [6, 148, 72, 35, 0, 33.6, 0.627, 50]  # Sample input
print(predict_diabetes(input_data))
