import streamlit as st
import pandas as pd
#import requests
#from io import BytesIO

# Load the pre-trained Random Forest model (replace 'model.pkl' with your actual path)
model = pd.read_pickle(r'titanic_model.pkl')

# Title and description for your app
st.title("TITANIC SIRVIVAL PREDICTOR")
st.write("Predict a passenger's survival chance based on their information.")

#features = ['Pclass', 'Sex', 'Embarked', 'famsize','AgeCategory','SibSp','Parch'] 
# Input fields for passenger characteristics
pclass = st.selectbox("Passenger Class", options=[1, 2, 3])
sex = st.selectbox("Sex", options=["Male", "Female"])
age = st.slider('Age', 0, 120, 1)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0)
embarked = st.selectbox("Embarked Location", options=["C", "S", "Q"])

#derived_variables
famsize = parch + sibsp

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
age_category = pd.cut([age], bins=bins, labels=labels, right=True)[0]

embarked = {'S': 1, 'C': 2, 'Q': 3}.get(embarked)

sex = {'male': 1, 'female': 0}.get(sex)


# Prepare user input data (consider data cleaning and feature engineering)
data = {
    "Pclass": pclass,
    "Sex": sex,
    "Embarked": embarked,
    "famsize":famsize,
    "AgeCategory": age_category,
    "SibSp": sibsp,
    "Parch": parch
}

# Convert data to a DataFrame
df = pd.DataFrame(data, index=[0])

#Image Reader function 


# Prediction button
if st.button("Predict Survival"):

    # Make prediction using the model
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)
    survival_percentage = round(probabilities[0][1]*100,2)
    no_survival_percentage = round(probabilities[0][1]*100,2)

    if survival_percentage>50:
        #image_base64 = get_base64_of_bin_file(r"data:image/jpeg;base64,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")
        st.image(r"survived.JPG", caption='Hurray...!', use_column_width=True)
        predicted_label = "**Your survival is statistically significant.**"
        # Display prediction results
        st.write(predicted_label)
        st.write(f"**Probability of Survival**: {survival_percentage}%")


    else:
        #image_base64 = get_base64_of_bin_file(r"C:\Users\rasa1012\Documents\Personal\ML Projects\Titanic Prediction\drowning.jpg")
        #st.image(r"drowning.jpg", caption='Sorry...', use_column_width=True)
        predicted_label = "**You went down with the ship, but your data lives on!.Maybe next time, choose a lifeboat!**"
        # Display prediction results
        st.write(predicted_label)
        st.write(f"**Probability of Survival*: {abs(1-survival_percentage)}%")


