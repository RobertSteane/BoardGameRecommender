import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV file into a DataFrame
file_path = 'collection.csv'
data = pd.read_csv(file_path)

# Split the data into items with non-zero ratings and items with zero ratings whilst excluding any expansions
non_zero_data = data[(data['rating'] != 0) & (data['itemtype'] == 'standalone')]
zero_data = data[(data['rating'] == 0) & (data['itemtype'] == 'standalone')].copy()  # Explicit copy

# Define cap values for each variable
caps = {
    'maxplaytime': 300,
    'maxplayers': 8,
}

# Apply cap function
def cap_application_function(df, variable, variable_cap):
    df.loc[:, variable] = df[variable].clip(upper=variable_cap)
    return df

# Apply caps to both non-zero data and zero data
for variable, cap in caps.items():
    non_zero_data = cap_application_function(non_zero_data, variable, cap)
    zero_data = cap_application_function(zero_data, variable, cap)

# Define feature variables (X) and the target variable (y) for training
x_variables = ['baverage', 'avgweight', 'maxplayers', 'maxplaytime']
X = non_zero_data[x_variables]
y = non_zero_data['rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Print model coefficients and intercept
print("""Model Coefficients and Intercept:
""")
for feature, coef in zip(x_variables, model.coef_):
    print(f'Coefficient for {feature}: {coef}')
print("Model Intercept:", model.intercept_)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print()
print(f'Mean Squared Error: {mse}')
print("""f'R^2 Score: {r2}'
""")

# Prepare the zero_data for prediction
X_zero = zero_data[x_variables]

# Make predictions for the items with a 'rating' of 0
predictions = model.predict(X_zero)

# Set predictions in the copied DataFrame
zero_data.loc[:, 'predicted_rating'] = predictions

# Display the predictions for currently unrated games
print("The predictions for the unrated games in your wishlist are:")
print(zero_data[['objectname', 'predicted_rating']])
