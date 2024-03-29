from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import string
from nltk.corpus import stopwords
import nltk
from sklearn.tree import DecisionTreeClassifier





# Download NLTK resources
nltk.download('stopwords')

app = Flask(__name__)

# Function to preprocess the input text
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load the trained model and vectorizer
model = DecisionTreeClassifier(random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Load the data and prepare it
fake = pd.read_csv('Fake.csv')  # Replace 'fake_data.csv' with your fake data file
true = pd.read_csv('True.csv')  # Replace 'true_data.csv' with your true data file

fake['target'] = 'fake'
true['target'] = 'true'
data = pd.concat([fake, true]).reset_index(drop=True)
data.drop(["date", "title"], axis=1, inplace=True)

# Preprocess the text data
data['text'] = data['text'].apply(preprocess_text)

# Vectorize the text data
X = vectorizer.fit_transform(data['text'])
y = data['target']

# Train the model
model.fit(X, y)

# Flask route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)
        # Vectorize the preprocessed text
        text_vectorized = vectorizer.transform([preprocessed_text])
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        return render_template('index.html', prediction=prediction, text=text)

if __name__ == '__main__':
    app.run(debug=True)
