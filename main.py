import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException
import time

# Load the CSV file into a DataFrame
file_path = 'collection.csv'
df = pd.read_csv(file_path)

# Initial list of variables to assign weights to
x_variables = ['baverage', 'avgweight', 'maxplayers', 'maxplaytime']


# function to initialise the chrome web driver
def initialize_driver():
    """Initialize and return a headless Chrome driver."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return chrome_driver


# function to retrieve multiple elements from a website
def get_elements_text(chrome_driver, by, value, timeout=10):
    """Get text from multiple elements, returning an empty list if none found."""
    try:
        elements = WebDriverWait(chrome_driver, timeout).until(
            ec.presence_of_all_elements_located((by, value))
        )
        return [element.text.strip() for element in elements]
    except NoSuchElementException:
        return []


# function to retrieve data about which categories a board game has on boardgamegeek.com
def get_game_category_data(game_id, web_driver):
    url = f'https://boardgamegeek.com/boardgame/{game_id}'
    web_driver.get(url)

    categories = web_driver.execute_script("""
                    const popupLists = Array.from(document.querySelectorAll('popup-list'));
                    const categoryList = popupLists
                        .filter(list => list.getAttribute('sref').includes('boardgamecategory'))
                        .flatMap(list => Array.from(list.querySelectorAll('.text-block a.ng-binding')))
                        .map(a => a.textContent.trim())
                        .filter(text => text.length > 0);
                    return categoryList;
                """)
    return categories


# function to retrieve data about which mechanics a board game has on boardgamegeek.com
def get_game_mechanic_data(game_id, web_driver):
    url = f'https://boardgamegeek.com/boardgame/{game_id}'
    web_driver.get(url)
    current_url = web_driver.current_url
    new_url = f'{current_url}/credits'
    web_driver.get(new_url)

    mechanics = web_driver.execute_script("""
        const parentElement = document.querySelector('#mainbody > div.global-body-content-container.container-fluid > div > div.content.ng-isolate-scope > div:nth-child(2) > ng-include > div > div > ui-view > ui-view > div > div > div.panel-body > credits-module > ul > li:nth-child(15) > div.outline-item-description > div');
        if (!parentElement) {
            return []; // Return an empty array if the parent element is not found
        }
        const links = Array.from(parentElement.querySelectorAll('a'));
        return links.map(link => link.textContent.trim()).filter(text => text.length > 0);
    """)
    return mechanics


# Initialize the WebDriver
driver = initialize_driver()

# Filter the DataFrame to only include rows where 'rating' != 0
non_zero_df = df[df['rating'] != 0]

print("Do you wish to model with categories of each game, mechanics of each game, or neither?")
choice = ()
while choice != "categories" and choice != "mechanics" and choice != "neither":
    choice = input().lower()


def process_data(web_driver, get_game_data_func):

    # Iterate over each `objectid` in the filtered DataFrame
    for game_id in non_zero_df['objectid'].unique():
        data = get_game_data_func(game_id, web_driver)
        if data:
            for item in data:
                column_name = item
                if column_name not in df.columns:
                    df[column_name] = 0
                    x_variables.append(column_name)
                df.loc[df['objectid'] == game_id, column_name] = 1
        # Introduce a delay between requests
        time.sleep(2)


if choice == "categories":
    process_data(driver, get_game_category_data)
if choice == "mechanics":
    process_data(driver, get_game_mechanic_data)

# Close the WebDriver
driver.quit()

essential_columns = ['objectid', 'rating', 'itemtype', 'average' 'baverage', 'avgweight', 'maxplayers', 'maxplaytime']
# Initialize a list to store columns to drop

for column in df.columns:
    # Ensure essential columns are not removed
    if column not in essential_columns:
        # Count non-zero entries in the column
        one_count = (df[column] == 1).sum()
        if 0 < one_count < 7:
            # Remove any variables that have only not many data points
            if column in x_variables:
                x_variables.remove(column)

# Save the modified DataFrame back to a CSV file
df.to_csv('modified_file.csv', index=False)

print("CSV file has been updated with new columns for each unique category.")

# Reload the modified CSV file
df = pd.read_csv('modified_file.csv')

# Split the data into items with non-zero ratings and items with zero ratings whilst excluding any expansions
non_zero_data = df[(df['rating'] != 0) & (df['itemtype'] == 'standalone')]
zero_data = df[(df['rating'] == 0) & (df['itemtype'] == 'standalone')].copy()  # Explicit copy

# Define cap values for each variable
caps = {
    'maxplaytime': 300,
    'maxplayers': 8,
}


# Apply cap function
def cap_application_function(dataframe, caps_variable, variable_cap):
    dataframe.loc[:, caps_variable] = dataframe[caps_variable].clip(upper=variable_cap)
    return dataframe


# Apply caps to both non-zero data and zero data
for variable, cap in caps.items():
    non_zero_data = cap_application_function(non_zero_data, variable, cap)
    zero_data = cap_application_function(zero_data, variable, cap)

# Define feature variables (X) and the target variable (y) for training
X = non_zero_data[x_variables]
y = non_zero_data['rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Print model coefficients and intercept
print("Model Coefficients and Intercept:")
for feature, coef in zip(x_variables, model.coef_):
    print(f'Coefficient for {feature}: {coef}')
print("Model Intercept:", model.intercept_)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Prepare the zero_data for prediction
X_zero = zero_data[x_variables]

# Make predictions for the items with a 'rating' of 0
predictions = model.predict(X_zero)

# Set predictions in the copied DataFrame
zero_data['predicted_rating'] = predictions

# Display the predictions for currently unrated games
print("The predictions for the unrated games in your wishlist are:")
print(zero_data[['objectname', 'predicted_rating']])
