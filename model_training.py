import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data = pd.read_csv('salaryData.csv')

# Inspect the first few rows
print("First few rows of the dataset:")
print(data.head())

# Check the shape of the dataset
print("\nShape of the dataset:")
print(data.shape)

# Descriptive statistics of the dataset
print("\nDescriptive statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Drop rows with missing values
data = data.dropna()

# Confirm no missing values remain
print("\nMissing values after cleaning:")
print(data.isnull().sum())

# Example preprocessing: Assuming 'Gender', 'Education Level', and 'Job Title' are categorical
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Education Level'] = data['Education Level'].astype('category').cat.codes
data['Job Title'] = data['Job Title'].astype('category').cat.codes

# Features and target variable
X = data[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = data['Salary']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training complete and saved to model.pkl")
