#Sudam Sanjula

#Importing Libraries

#pandas: Used for data manipulation and analysis
import pandas as pd  
#train_test_split: Splits the data into training and testing sets
from sklearn.model_selection import train_test_split
#RandomForestClassifier: A machine learning algorithm used for classification.
from sklearn.ensemble import RandomForestClassifier
#accuracy_score, classification_report: Metrics to evaluate the model's performance
from sklearn.metrics import accuracy_score, classification_report

#Loading the Dataset

# Prompt for the dataset path
#input: Prompts the user to enter the path to the dataset file
file_path = input('Please enter the path to your Titanic dataset file: ')
#pd.read_csv: Reads the dataset into a DataFrame.
titanic_df = pd.read_csv(file_path)

#Data Cleaning and Preprocessing

# Fill missing Age values with the median age
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
# Fill missing Embarked values with the most common port of embarkation
titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])
# Create a new feature indicating whether a cabin number is known
titanic_df['CabinKnown'] = titanic_df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
# Drop the original Cabin column
titanic_df = titanic_df.drop(columns=['Cabin'])
# Convert categorical features to numerical values
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
titanic_df['Embarked'] = titanic_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Drop columns that won't be used in the model
titanic_df = titanic_df.drop(columns=['Name', 'Ticket'])

#Splitting the Data
# Split the data into features and target label
X = titanic_df.drop(columns=['Survived'])
y = titanic_df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training the Model
# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Making Predictions and Evaluating the Model
# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Print classification report
print(classification_report(y_test, y_pred))

# If you have new data and want to make predictions
# Assuming new_data is a DataFrame with the same structure as X
# new_data = pd.read_csv('path_to_new_data.csv')
# predictions = model.predict(new_data)
# print(predictions)
