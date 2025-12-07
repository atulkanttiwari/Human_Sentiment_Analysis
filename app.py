from flask import Flask, render_template, request
import os
import re
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# ensure nltk stopwords are available
try:
	stopwords.words('english')
except Exception:
	nltk.download('stopwords')

port_stem = PorterStemmer()

def stemming(content: str) -> str:
	content = re.sub('[^a-zA-Z]', ' ', content)
	content = content.lower()
	content = content.split()
	content = [port_stem.stem(word) for word in content if not word in stopwords.words('english')]
	content = ' '.join(content)
	return content


MODEL_PATH = 'trained_model.pkl'
VECT_PATH = 'vectorizer.pkl'
DATA_CSV = 'twitterdata.csv'


def load_model():
	if not os.path.exists(MODEL_PATH):
		raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
	model = joblib.load(MODEL_PATH)
	return model


def load_or_build_vectorizer():
	if os.path.exists(VECT_PATH):
		vect = joblib.load(VECT_PATH)
		return vect

	# build vectorizer using the same preprocessing as the notebook
	if not os.path.exists(DATA_CSV):
		raise FileNotFoundError(f"No vectorizer saved and training CSV not found: {DATA_CSV}")

	df = pd.read_csv(DATA_CSV)
	# expect the text column to be named 'text' as in the notebook
	if 'text' in df.columns:
		text_col = 'text'
	elif 'tweet' in df.columns:
		text_col = 'tweet'
	else:
		# choose the most likely text column: object dtype and largest average length
		obj_cols = [c for c in df.columns if df[c].dtype == object]
		if not obj_cols:
			text_col = df.columns[0]
		else:
			def avg_len(col):
				return df[col].astype(str).map(len).mean()
			text_col = max(obj_cols, key=avg_len)

	df['stemmed_content'] = df[text_col].astype(str).apply(stemming)
	vectorizer = TfidfVectorizer()
	vectorizer.fit(df['stemmed_content'].values)
	joblib.dump(vectorizer, VECT_PATH)
	return vectorizer


@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
	text = request.form.get('text', '')
	if not text:
		return render_template('result.html', error='No text provided')

	try:
		model = load_model()
	except Exception as e:
		return render_template('result.html', error=f'Error loading model: {e}')

	try:
		vectorizer = load_or_build_vectorizer()
	except Exception as e:
		return render_template('result.html', error=f'Error building vectorizer: {e}')

	processed = stemming(text)
	X = vectorizer.transform([processed])

	try:
		pred = model.predict(X)[0]
	except Exception as e:
		return render_template('result.html', error=f'Prediction error: {e}')

	label = 'Positive' if int(pred) == 1 else 'Negative'
	prob = None
	try:
		if hasattr(model, 'predict_proba'):
			prob = float(model.predict_proba(X)[0][1])
	except Exception:
		prob = None

	return render_template('result.html', text=text, prediction=label, probability=prob)


if __name__ == '__main__':
	app.run(debug=True)
