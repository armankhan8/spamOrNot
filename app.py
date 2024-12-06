from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize Flask App
app = Flask(__name__)

# Load and preprocess dataset
df = pd.read_csv('spam_or_not_spam.csv')
df = df.drop_duplicates().dropna()
df['email'] = df['email'].str.lower()

# Balance the dataset if imbalanced
spam_count = df[df['label'] == 1].shape[0]
non_spam_count = df[df['label'] == 0].shape[0]
if spam_count > non_spam_count:
    df = pd.concat([df[df['label'] == 0], df[df['label'] == 1].sample(non_spam_count, random_state=42)])
else:
    df = pd.concat([df[df['label'] == 1], df[df['label'] == 0].sample(spam_count, random_state=42)])

# Feature extraction using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
x = vectorizer.fit_transform(df['email'])
y = df['label']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Evaluate accuracy
y_pred = model.predict(x_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")


# Flask Routes
@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict whether the input email is Spam or Not Spam."""
    new_email = request.form['email']
    if not new_email.strip():
        result = "Invalid Input! Please enter valid email content."
        return render_template('result.html', result=result, email=new_email)

    # Preprocess and predict
    processed_email = vectorizer.transform([new_email.lower()])
    prediction = model.predict(processed_email)[0]
    result = "Spam" if prediction == 1 else "Not Spam"

    return render_template('result.html', result=result, email=new_email)


if __name__ == "__main__":
    app.run(debug=True)
