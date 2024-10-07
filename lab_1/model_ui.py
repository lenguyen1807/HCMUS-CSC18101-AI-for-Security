import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the pretrained model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Preprocess the input text
def preprocess_text(text, vectorizer):
    return vectorizer.transform([text])

# Predict function for a single email
def predict_email(model, vectorizer, subject, message):
    email_text = subject + " " + message
    email_vectorized = preprocess_text(email_text, vectorizer)
    prediction = model.predict(email_vectorized)
    return "Spam" if prediction[0] == 'spam' else "Ham"

# Evaluate function for CSV file input
def evaluate_on_csv(model, vectorizer, file):
    data = pd.read_csv(file)
    data['Subject'] = data['Subject'].fillna('')
    data['Message'] = data['Message'].fillna('')
    data['Text'] = data['Subject'] + " " + data['Message']
    X = vectorizer.transform(data['Text'])
    y_true = data['Spam/Ham']
    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='spam')
    recall = recall_score(y_true, y_pred, pos_label='spam')
    f1 = f1_score(y_true, y_pred, pos_label='spam')

    return accuracy, precision, recall, f1

# Streamlit app
st.title('Spam Classifier')

# Input section for subject and message
subject = st.text_input("Email Subject")
message = st.text_area("Email Message")

if st.button("Predict"):
    if subject and message:
        prediction = predict_email(model, tfidf, subject, message)
        st.write(f"The email is classified as: **{prediction}**")
    else:
        st.write("Please enter both subject and message to classify.")

# File upload for CSV evaluation
st.subheader('Evaluate on CSV')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    accuracy, precision, recall, f1 = evaluate_on_csv(model, tfidf, uploaded_file)
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1 Score: {f1}")

