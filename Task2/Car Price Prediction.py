# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('CarPrice_Assignment.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values and data types
print(data.info())

# Statistical summary of numerical columns
print(data.describe())

# Visualize correlations between numerical features
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Select features and target variable
# You might need to preprocess categorical variables (e.g., one-hot encoding)
# For simplicity, let's assume 'price' is the target variable and other columns are features
features = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']
target = 'price'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Now, you can use this trained model to predict car prices for new data
# For example, to predict the price of a single car:
new_data = pd.DataFrame({
    'symboling': [0],
    'wheelbase': [100],
    'carlength': [180],
    'carwidth': [70],
    'carheight': [50],
    'curbweight': [2500],
    'enginesize': [150],
    'boreratio': [3.5],
    'stroke': [3],
    'compressionratio': [9],
    'horsepower': [120],
    'peakrpm': [6000],
    'citympg': [25],
    'highwaympg': [30]
})

# Make predictions on the new data
predicted_price = model.predict(new_data)
print(f"Predicted price for the new car: {predicted_price[0]}")
