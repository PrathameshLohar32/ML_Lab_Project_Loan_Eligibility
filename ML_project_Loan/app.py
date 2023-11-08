from flask import Flask, render_template, request
import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__,static_folder="static")

# Load the training data
train_data = pd.read_csv('Train_Data.csv')

# Encode categorical features
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    train_data[feature] = le.fit_transform(train_data[feature])
    label_encoders[feature] = le

# Define the features and target variable
features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History'] + ['Property_Area']
target = 'Loan_Status'

a = train_data[features]
b = train_data[target]

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
a = pd.DataFrame(imputer.fit_transform(a), columns=a.columns)

# Create and fit the Random forest model
Random_forest = RandomForestClassifier(n_estimators = 100)
Random_forest.fit(a, b)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Get user inputs from the form
        user_data = request.form.to_dict()

        # Prepare user input data
        input_data = [float(user_data['ApplicantIncome']), float(user_data['CoapplicantIncome']),
                      float(user_data['LoanAmount']), float(user_data['Loan_Amount_Term']),
                      float(user_data['Credit_History'])]

        # Encode 'Property_Area' using the label encoder
        input_data.append(label_encoders['Property_Area'].transform([user_data['Property_Area']])[0])

        # Handle missing values in user input
        input_data = imputer.transform([input_data])

        # Make a prediction using the KNN model
        prediction = Random_forest.predict(input_data)[0]
    if prediction=='Y':
        prediction="Congratulations You are eligible"
    else:
        prediction="Sorry You are Not Eligible for loan"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
