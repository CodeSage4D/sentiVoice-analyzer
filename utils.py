# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# import joblib
# import os
# from tqdm import tqdm
# import time

# # Function to load and filter data
# def load_data(file_path):
#     try:
#         df = pd.read_csv(file_path)
#         df = df[df['label'].isin([0, 1, 2])]
#         return df
#     except FileNotFoundError:
#         print(f"Error: File '{file_path}' not found.")
#     except pd.errors.ParserError as pe:
#         print(f"Error: Unable to parse CSV file '{file_path}'. Details: {pe}")
#     except Exception as e:
#         print(f"Error loading data: {e}")
#     return None

# # Function to preprocess data and return vectorized X and y
# def preprocess_data(df, vectorizer=None, max_features=5000):
#     if vectorizer is None:
#         vectorizer = TfidfVectorizer(max_features=max_features)
#     corpus = df['text']
#     X = vectorizer.transform(tqdm(corpus, desc="Vectorizing Data"))
#     y = df['label']
#     return X, y, vectorizer

# # Function to save model and vectorizer
# def save_model(model, vectorizer, model_dir='model'):
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#     joblib.dump(model, os.path.join(model_dir, 'best_model.pkl'))
#     joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
#     print("Model and vectorizer saved successfully.")

# # Function to load model and vectorizer
# def load_model(model_dir='model'):
#     model_path = os.path.join(model_dir, 'best_model.pkl')
#     vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
#     if os.path.exists(model_path) and os.path.exists(vectorizer_path):
#         model = joblib.load(model_path)
#         vectorizer = joblib.load(vectorizer_path)
#         return model, vectorizer
#     else:
#         raise FileNotFoundError("Model files not found. Train the model first.")

# # Function to preprocess new data using an existing vectorizer
# def preprocess_new_data(new_data, vectorizer):
#     X = vectorizer.transform(new_data)
#     return X
# ---------------------------------------------------------------------------------------------

import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def store_feedback(feedbacks, predictions):
    feedback_data = pd.DataFrame({'Feedback': feedbacks, 'Predicted_Label': predictions})
    feedback_dir = 'feedback_results'
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_file = os.path.join(feedback_dir, 'feedback_data.csv')
    
    if os.path.exists(feedback_file):
        feedback_data.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        feedback_data.to_csv(feedback_file, index=False)

def load_feedback():
    feedback_file = os.path.join('feedback_results', 'feedback_data.csv')
    if os.path.exists(feedback_file):
        return pd.read_csv(feedback_file)
    else:
        return None

# Function to load and filter data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = df[df['label'].isin([0, 1, 2])]
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except pd.errors.ParserError as pe:
        print(f"Error: Unable to parse CSV file '{file_path}'. Details: {pe}")
    except Exception as e:
        print(f"Error loading data: {e}")
    return None

# Function to preprocess data and return vectorized X and y
def preprocess_data(df, max_data=None):
    if max_data:
        df = df.sample(n=max_data, random_state=42)
    corpus = df['text']
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(tqdm(corpus, desc="Vectorizing Data"))
    y = df['label']
    return X, y, vectorizer

# Function to save model and vectorizer
def save_model(model, vectorizer, model_dir='model'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(model, os.path.join(model_dir, 'best_model.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
    print("Model and vectorizer saved successfully.")

# Function to load model and vectorizer
def load_model(model_dir='model'):
    model_path = os.path.join(model_dir, 'best_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    else:
        raise FileNotFoundError("Model files not found. Train the model first.")

# # Function to store feedback and their predicted labels
# def store_feedback(feedback, predictions, file_path='feedback_storage.csv'):
#     feedback_df = pd.DataFrame({'Feedback': feedback, 'Predicted_Label': predictions})
#     feedback_df.to_csv(file_path, index=False)
#     print(f"Feedback stored successfully in '{file_path}'.")

# # Function to load stored feedback
# def load_feedback(file_path='feedback_storage.csv'):
#     if os.path.exists(file_path):
#         df = pd.read_csv(file_path)
#         return df
#     else:
#         print(f"File '{file_path}' not found.")
#         return None
