# import pandas as pd
# import os
# import joblib
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# import time
# import threading

# def load_data(file_path):
#     try:
#         # Explicitly define the column names and skip the first row if it's a header
#         df = pd.read_csv(file_path, encoding='utf-8', names=['Unnamed: 0', 'text', 'label'], skiprows=1)
        
#         # Check if the expected columns exist
#         expected_columns = ['Unnamed: 0', 'text', 'label']
#         if not all(col in df.columns for col in expected_columns):
#             raise ValueError(f"CSV file '{file_path}' does not contain all expected columns: {expected_columns}")
        
#         # Filter out rows with labels other than 0, 1, 2
#         df = df[df['label'].isin([0, 1, 2])]
        
#         return df
#     except FileNotFoundError:
#         print(f"Error: File '{file_path}' not found.")
#     except pd.errors.ParserError as pe:
#         print(f"Error: Unable to parse CSV file '{file_path}'. Details: {pe}")
#     except ValueError as ve:
#         print(f"Error: {ve}")
#     except Exception as e:
#         print(f"Error loading data: {e}")
#     return None

# # Function to preprocess data and return vectorized X and y
# def preprocess_data(df, max_data=None):
#     if df is None:
#         return None, None, None
    
#     start_time = time.time()
#     if max_data:
#         df = df.sample(n=max_data, random_state=42)
#     corpus = df['text']
#     vectorizer = TfidfVectorizer(max_features=5000)
#     X = vectorizer.fit_transform(tqdm(corpus, desc="Vectorizing Data"))
#     y = df['label']
#     end_time = time.time()
#     preprocess_time = end_time - start_time
#     print("Preprocessing time:", preprocess_time, "seconds")
#     return X, y, vectorizer

# # Function to train the model
# def train_model(X, y):
#     start_time = time.time()
#     model = MultinomialNB()
#     model.fit(X, y)
#     end_time = time.time()
#     train_time = end_time - start_time
#     print("Training time:", train_time, "seconds")
#     return model

# # Function to evaluate the model
# def evaluate_model(model, X_test, y_test):
#     start_time = time.time()
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     class_report = classification_report(y_test, y_pred)
#     end_time = time.time()
#     evaluation_time = end_time - start_time
#     print("Evaluation time:", evaluation_time, "seconds")
#     return accuracy, conf_matrix, class_report

# # Function to plot confusion matrix
# def plot_confusion_matrix(conf_matrix):
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title('Confusion Matrix')
#     plt.show()

# # Function to save the model and vectorizer
# def save_model(model, vectorizer):
#     if not os.path.exists('model'):
#         os.makedirs('model')
#     joblib.dump(model, 'model/best_model.pkl')
#     joblib.dump(vectorizer, 'model/vectorizer.pkl')
#     print("Model and vectorizer saved successfully.")

# # Function to simulate training progress
# def simulate_training_progress():
#     for i in tqdm(range(100), desc="Training Progress"):
#         time.sleep(0.1)  # Simulating training time

# def main():
#     # Load data
#     file_path = 'data/Emotions_filtered.csv'  # Corrected file path
#     df = load_data(file_path)

#     if df is None:
#         print("Failed to load data. Exiting.")
#         return

#     # Specify the maximum data size
#     max_data = None  # Set this to the desired maximum data size, or None to use all data

#     # Preprocess data
#     X, y, vectorizer = preprocess_data(df, max_data=max_data)

#     if X is None or y is None or vectorizer is None:
#         print("Failed to preprocess data. Exiting.")
#         return

#     # Split data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#     # Train the model
#     print("Training the model...")
#     training_thread = threading.Thread(target=simulate_training_progress)
#     training_thread.start()
#     best_model = train_model(X_train, y_train)
#     training_thread.join()

#     # Evaluate the model
#     print("\nEvaluating the model...")
#     accuracy, conf_matrix, class_report = evaluate_model(best_model, X_test, y_test)
#     print("Accuracy:", accuracy)
#     print("Confusion Matrix:")
#     print(conf_matrix)
#     print("Classification Report:")
#     print(class_report)

#     # Plot confusion matrix
#     plot_confusion_matrix(conf_matrix)

#     # Save the model and vectorizer
#     save_model(best_model, vectorizer)

# if __name__ == "__main__":
#     main()
# ------------------------------------------------------------

import pandas as pd

def load_data(file_path):
    try:
        # Read the CSV file, explicitly specifying dtype for 'label' as object (str)
        df = pd.read_csv(file_path, dtype={'label': str})
        
        # Convert 'label' column to numeric, coercing errors to NaN (not a number)
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        
        # Drop rows where 'label' could not be converted to numeric
        df.dropna(subset=['label'], inplace=True)
        
        # Convert 'label' column to integer type
        df['label'] = df['label'].astype(int)
        
        # Filter out rows with labels other than 0, 1, 2
        df = df[df['label'].isin([0, 1, 2])]
        
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"Error loading data: {e}")
    return None

def main():
    file_path = 'data/Emotions_filtered.csv'
    df = load_data(file_path)

    if df is None or df.empty:
        print("Failed to load or preprocess data. Exiting.")
        return

    # Continue with your preprocessing and model training as before
    # Example:
    # X, vectorizer = preprocess_data(df['text'])
    # y = df['label']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # model = train_model(X_train, y_train)
    # Save or use the model...

if __name__ == "__main__":
    main()
