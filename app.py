# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from collections import Counter
# import numpy as np
# from wordcloud import WordCloud
# import plotly.express as px
# import os
# import speech_recognition as sr

# # Load the pre-trained model and vectorizer
# model = joblib.load('model/best_model.pkl')
# vectorizer = joblib.load('model/vectorizer.pkl')

# # Function to predict sentiment and return results
# def predict_sentiment(text):
#     text_vectorized = vectorizer.transform([text])
#     prediction = model.predict(text_vectorized)[0]
#     prediction_proba = model.predict_proba(text_vectorized)
#     return prediction, prediction_proba

# # Function to calculate sentiment statistics
# def calculate_statistics(predictions):
#     count = Counter(predictions)
#     total = len(predictions)
#     positive_percentage = (count[1] / total) * 100 if total > 0 else 0
#     negative_percentage = (count[-1] / total) * 100 if total > 0 else 0
#     neutral_percentage = (count[0] / total) * 100 if total > 0 else 0
#     return count, positive_percentage, negative_percentage, neutral_percentage

# # Function to plot sentiment distribution
# def plot_sentiment_distribution(predictions):
#     labels, counts = zip(*Counter(predictions).items())
#     fig, ax = plt.subplots()
#     ax.bar(labels, counts, align='center', alpha=0.5)
#     ax.set_xticks([-1, 0, 1])
#     ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
#     ax.set_ylabel('Count')
#     ax.set_title('Sentiment Distribution')
#     st.pyplot(fig)

# # Function to display word cloud
# def display_word_cloud(texts):
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(texts))
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     st.pyplot()

# # Function to plot line chart
# def plot_line_chart(data):
#     df = pd.DataFrame(data, columns=['Feedback', 'Score'])
#     fig = px.line(df, x='Feedback', y='Score', title='Sentiment Scores Over Time')
#     st.plotly_chart(fig)

# # Function to perform voice recognition and execute commands
# def process_voice_command():
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()
    
#     with microphone as source:
#         st.warning('Listening...')
#         recognizer.adjust_for_ambient_noise(source)
#         audio = recognizer.listen(source)
    
#     st.warning('Processing...')
    
#     try:
#         command = recognizer.recognize_google(audio).lower()
#         st.info(f'Command recognized: {command}')
        
#         if 'show last 5 feedback' in command:
#             # Implement logic to fetch and display last 5 feedback
#             st.success('Fetching last 5 feedback...')
#             # Replace with actual logic to fetch and display data
#             st.info('Data not implemented yet.')
        
#         elif 'count positive feedback' in command:
#             # Implement logic to count positive feedback
#             st.success('Counting positive feedback...')
#             # Replace with actual logic to count positive feedback
#             st.info('Data not implemented yet.')
        
#         elif 'count negative feedback' in command:
#             # Implement logic to count negative feedback
#             st.success('Counting negative feedback...')
#             # Replace with actual logic to count negative feedback
#             st.info('Data not implemented yet.')
        
#         elif 'count neutral feedback' in command:
#             # Implement logic to count neutral feedback
#             st.success('Counting neutral feedback...')
#             # Replace with actual logic to count neutral feedback
#             st.info('Data not implemented yet.')
        
#         else:
#             st.warning('Command not recognized. Please try again.')
    
#     except sr.UnknownValueError:
#         st.error("Could not understand audio.")
#     except sr.RequestError as e:
#         st.error(f"Could not request results from Google Speech Recognition service; {e}")

# # Main Streamlit application
# def main():
#     st.title('Sentiment Analysis Application')

#     # Collect user feedback
#     user_input = st.text_area('Enter your feedback:')
    
#     if st.button('Analyze'):
#         if user_input:
#             # Predict sentiment and get probabilities
#             prediction, prediction_proba = predict_sentiment(user_input)
            
#             # Extract probabilities for each class
#             negative_proba = prediction_proba[0][-1]
#             neutral_proba = prediction_proba[0][0]
#             positive_proba = prediction_proba[0][1]

#             # Display sentiment result
#             st.subheader('Sentiment Analysis Result:')
#             st.write(f'Feedback: {user_input}')
#             st.write(f'Predicted Sentiment: {"Positive" if prediction == 1 else "Negative" if prediction == -1 else "Neutral"}')

#             # Display sentiment probabilities
#             st.subheader('Sentiment Probabilities:')
#             st.write('Negative:', negative_proba)
#             st.write('Neutral:', neutral_proba)
#             st.write('Positive:', positive_proba)

#             # Plot sentiment distribution
#             predictions = np.argmax(prediction_proba, axis=1) - 1  # Convert probabilities to classes
#             plot_sentiment_distribution(predictions)

#             # Word cloud and line plot
#             feedback_data = [('Feedback', user_input)]
#             plot_line_chart(feedback_data)
            
#             display_word_cloud(user_input.split())
        
#         else:
#             st.warning('Please enter some feedback.')
    
#     # Voice command processing
#     st.sidebar.subheader('Voice Command Processing')
#     if st.sidebar.button('Activate Voice Command'):
#         process_voice_command()

# if __name__ == "__main__":
#     main()
#---------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from collections import Counter
# import numpy as np
# from wordcloud import WordCloud
# import plotly.express as px
# import os
# import speech_recognition as sr
# from utils import preprocess_data, save_model, load_model, store_feedback, load_feedback

# # Load the pre-trained model and vectorizer
# model, vectorizer = load_model()

# # Function to predict sentiment and return results
# def predict_sentiment(text):
#     text_vectorized = vectorizer.transform([text])
#     prediction = model.predict(text_vectorized)[0]
#     prediction_proba = model.predict_proba(text_vectorized)
#     return prediction, prediction_proba

# # Function to calculate sentiment statistics
# def calculate_statistics(predictions):
#     count = Counter(predictions)
#     total = len(predictions)
#     positive_percentage = (count[1] / total) * 100 if total > 0 else 0
#     negative_percentage = (count[0] / total) * 100 if total > 0 else 0
#     neutral_percentage = (count[2] / total) * 100 if total > 0 else 0
#     return count, positive_percentage, negative_percentage, neutral_percentage

# # Function to plot sentiment distribution
# def plot_sentiment_distribution(predictions):
#     labels, counts = zip(*Counter(predictions).items())
#     fig, ax = plt.subplots()
#     ax.bar(labels, counts, align='center', alpha=0.5)
#     ax.set_xticks([0, 1, 2])
#     ax.set_xticklabels(['Negative', 'Positive', 'Neutral'])
#     ax.set_ylabel('Count')
#     ax.set_title('Sentiment Distribution')
#     st.pyplot(fig)

# # Function to display word cloud
# def display_word_cloud(texts):
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(texts))
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     st.pyplot(plt)

# # Function to plot line chart
# def plot_line_chart(data):
#     df = pd.DataFrame(data, columns=['Feedback', 'Score'])
#     fig = px.line(df, x='Feedback', y='Score', title='Sentiment Scores Over Time')
#     st.plotly_chart(fig)

# # Function to perform voice recognition and execute commands
# def process_voice_command():
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()
    
#     with microphone as source:
#         st.warning('Listening...')
#         recognizer.adjust_for_ambient_noise(source)
#         audio = recognizer.listen(source)
    
#     st.warning('Processing...')
    
#     try:
#         command = recognizer.recognize_google(audio).lower()
#         st.info(f'Command recognized: {command}')
        
#         if 'show last 5 feedback' in command:
#             # Implement logic to fetch and display last 5 feedback
#             st.success('Fetching last 5 feedback...')
#             feedback_df = load_feedback()
#             if feedback_df is not None and not feedback_df.empty:
#                 st.dataframe(feedback_df.tail(5))  # Display last 5 feedback
#             else:
#                 st.info('No feedback stored yet.')
        
#         elif 'count positive feedback' in command:
#             # Implement logic to count positive feedback
#             st.success('Counting positive feedback...')
#             feedback_df = load_feedback()
#             if feedback_df is not None and not feedback_df.empty:
#                 count_positive = (feedback_df['Predicted_Label'] == 1).sum()
#                 st.write(f'Number of positive feedback: {count_positive}')
#             else:
#                 st.info('No feedback stored yet.')
        
#         elif 'count negative feedback' in command:
#             # Implement logic to count negative feedback
#             st.success('Counting negative feedback...')
#             feedback_df = load_feedback()
#             if feedback_df is not None and not feedback_df.empty:
#                 count_negative = (feedback_df['Predicted_Label'] == 0).sum()
#                 st.write(f'Number of negative feedback: {count_negative}')
#             else:
#                 st.info('No feedback stored yet.')
        
#         elif 'count neutral feedback' in command:
#             # Implement logic to count neutral feedback
#             st.success('Counting neutral feedback...')
#             feedback_df = load_feedback()
#             if feedback_df is not None and not feedback_df.empty:
#                 count_neutral = (feedback_df['Predicted_Label'] == 2).sum()
#                 st.write(f'Number of neutral feedback: {count_neutral}')
#             else:
#                 st.info('No feedback stored yet.')
        
#         else:
#             st.warning('Command not recognized. Please try again.')
    
#     except sr.UnknownValueError:
#         st.error("Could not understand audio.")
#     except sr.RequestError as e:
#         st.error(f"Could not request results from Google Speech Recognition service; {e}")

# # Main Streamlit application
# def main():
#     st.title('Sentiment Analysis Application')

#     # Collect user feedback
#     user_input = st.text_area('Enter your feedback:')
    
#     if st.button('Analyze'):
#         if user_input:
#             # Predict sentiment and get probabilities
#             prediction, prediction_proba = predict_sentiment(user_input)
            
#             # Extract probabilities for each class
#             negative_proba = prediction_proba[0][0]
#             positive_proba = prediction_proba[0][1]
#             neutral_proba = prediction_proba[0][2]

#             # Display sentiment result
#             st.subheader('Sentiment Analysis Result:')
#             st.write(f'Feedback: {user_input}')
            
#             # Determine sentiment based on prediction
#             if prediction == 1:
#                 sentiment = 'Positive'
#             elif prediction == 0:
#                 sentiment = 'Negative'
#             else:
#                 sentiment = 'Neutral'

#             st.write(f'Predicted Sentiment: {sentiment}')

#             # Display sentiment probabilities
#             st.subheader('Sentiment Probabilities:')
#             st.write('Negative:', negative_proba)
#             st.write('Positive:', positive_proba)
#             st.write('Neutral:', neutral_proba)

#             # Plot sentiment distribution
#             predictions = np.argmax(prediction_proba, axis=1)  # Convert probabilities to classes
#             plot_sentiment_distribution(predictions)

#             # Word cloud and line plot
#             feedback_data = [('Feedback', user_input)]
#             plot_line_chart(feedback_data)
            
#             display_word_cloud(user_input.split())

#             # Store feedback along with predicted label
#             store_feedback([user_input], [prediction])
            
#             # Calculate and display sentiment statistics
#             all_feedback = load_feedback()
#             if all_feedback is not None and not all_feedback.empty:
#                 _, positive_percentage, negative_percentage, neutral_percentage = calculate_statistics(all_feedback['Predicted_Label'])
#                 st.subheader('Sentiment Statistics:')
#                 st.write(f"Positive Feedback: {positive_percentage:.2f}%")
#                 st.write(f"Negative Feedback: {negative_percentage:.2f}%")
#                 st.write(f"Neutral Feedback: {neutral_percentage:.2f}%")
#             else:
#                 st.info('No feedback stored yet.')
        
#         else:
#             st.warning('Please enter some feedback.')
    
#     # Voice command processing
#     st.sidebar.subheader('Voice Command Processing')
#     if st.sidebar.button('Activate Voice Command'):
#         process_voice_command()

# if __name__ == "__main__":
#     main()

# # This code also work well
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from collections import Counter
# import numpy as np
# from wordcloud import WordCloud
# import plotly.express as px
# import os
# import speech_recognition as sr
# from utils import preprocess_data, save_model, load_model, store_feedback, load_feedback

# # Load the pre-trained model and vectorizer
# model, vectorizer = load_model()

# # Function to predict sentiment and return results
# def predict_sentiment(text):
#     text_vectorized = vectorizer.transform([text])
#     prediction = model.predict(text_vectorized)[0]
#     prediction_proba = model.predict_proba(text_vectorized)
#     return prediction, prediction_proba

# # Function to calculate sentiment statistics
# def calculate_statistics(predictions):
#     count = Counter(predictions)
#     total = len(predictions)
#     positive_percentage = (count[1] / total) * 100 if total > 0 else 0
#     negative_percentage = (count[-1] / total) * 100 if total > 0 else 0
#     neutral_percentage = (count[0] / total) * 100 if total > 0 else 0
#     return count, positive_percentage, negative_percentage, neutral_percentage

# # Function to plot sentiment distribution
# def plot_sentiment_distribution(predictions):
#     labels, counts = zip(*Counter(predictions).items())
#     fig, ax = plt.subplots()
#     ax.bar(labels, counts, align='center', alpha=0.5)
#     ax.set_xticks([-1, 0, 1])
#     ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
#     ax.set_ylabel('Count')
#     ax.set_title('Sentiment Distribution')
#     st.pyplot(fig)

# # Function to display word cloud
# def display_word_cloud(texts):
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(texts))
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     st.pyplot()

# # Function to plot line chart
# def plot_line_chart(data):
#     df = pd.DataFrame(data, columns=['Feedback', 'Score'])
#     fig = px.line(df, x='Feedback', y='Score', title='Sentiment Scores Over Time')
#     st.plotly_chart(fig)

# # Function to perform voice recognition and execute commands
# def process_voice_command():
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()
    
#     with microphone as source:
#         st.warning('Listening...')
#         recognizer.adjust_for_ambient_noise(source)
#         audio = recognizer.listen(source)
    
#     st.warning('Processing...')
    
#     try:
#         command = recognizer.recognize_google(audio).lower()
#         st.info(f'Command recognized: {command}')
        
#         if 'show last 5 feedback' in command:
#             # Implement logic to fetch and display last 5 feedback
#             st.success('Fetching last 5 feedback...')
#             feedback_df = load_feedback()
#             if feedback_df is not None:
#                 st.dataframe(feedback_df.tail(5))  # Display last 5 feedback
#             else:
#                 st.info('No feedback stored yet.')
        
#         elif 'count positive feedback' in command:
#             # Implement logic to count positive feedback
#             st.success('Counting positive feedback...')
#             feedback_df = load_feedback()
#             if feedback_df is not None:
#                 count_positive = (feedback_df['Predicted_Label'] == 1).sum()
#                 st.write(f'Number of positive feedback: {count_positive}')
#             else:
#                 st.info('No feedback stored yet.')
        
#         elif 'count negative feedback' in command:
#             # Implement logic to count negative feedback
#             st.success('Counting negative feedback...')
#             feedback_df = load_feedback()
#             if feedback_df is not None:
#                 count_negative = (feedback_df['Predicted_Label'] == -1).sum()
#                 st.write(f'Number of negative feedback: {count_negative}')
#             else:
#                 st.info('No feedback stored yet.')
        
#         elif 'count neutral feedback' in command:
#             # Implement logic to count neutral feedback
#             st.success('Counting neutral feedback...')
#             feedback_df = load_feedback()
#             if feedback_df is not None:
#                 count_neutral = (feedback_df['Predicted_Label'] == 0).sum()
#                 st.write(f'Number of neutral feedback: {count_neutral}')
#             else:
#                 st.info('No feedback stored yet.')
        
#         else:
#             st.warning('Command not recognized. Please try again.')
    
#     except sr.UnknownValueError:
#         st.error("Could not understand audio.")
#     except sr.RequestError as e:
#         st.error(f"Could not request results from Google Speech Recognition service; {e}")

# # Main Streamlit application
# def main():
#     st.title('Sentiment Analysis Application')

#     # Collect user feedback
#     user_input = st.text_area('Enter your feedback:')
    
#     if st.button('Analyze'):
#         if user_input:
#             # Predict sentiment and get probabilities
#             prediction, prediction_proba = predict_sentiment(user_input)
            
#             # Extract probabilities for each class
#             negative_proba = prediction_proba[0][-1]
#             neutral_proba = prediction_proba[0][0]
#             positive_proba = prediction_proba[0][1]

#             # Display sentiment result
#             st.subheader('Sentiment Analysis Result:')
#             st.write(f'Feedback: {user_input}')
#             st.write(f'Predicted Sentiment: {"Positive" if prediction == 1 else "Negative" if prediction == -1 else "Neutral"}')

#             # Display sentiment probabilities
#             st.subheader('Sentiment Probabilities:')
#             st.write('Negative:', negative_proba)
#             st.write('Neutral:', neutral_proba)
#             st.write('Positive:', positive_proba)

#             # Plot sentiment distribution
#             predictions = np.argmax(prediction_proba, axis=1) - 1  # Convert probabilities to classes
#             plot_sentiment_distribution(predictions)

#             # Word cloud and line plot
#             feedback_data = [('Feedback', user_input)]
#             plot_line_chart(feedback_data)
            
#             display_word_cloud(user_input.split())

#             # Store feedback along with predicted label
#             store_feedback([user_input], [prediction])
        
#         else:
#             st.warning('Please enter some feedback.')
    
#     # Voice command processing
#     st.sidebar.subheader('Voice Command Processing')
#     if st.sidebar.button('Activate Voice Command'):
#         process_voice_command()

# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
import os
import speech_recognition as sr
from utils import preprocess_data, save_model, load_model, store_feedback, load_feedback

# Load the pre-trained model and vectorizer
model, vectorizer = load_model()

# Function to predict sentiment and return results
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    prediction_proba = model.predict_proba(text_vectorized)
    return prediction, prediction_proba

# Function to calculate sentiment statistics
def calculate_statistics(predictions):
    count = Counter(predictions)
    total = len(predictions)
    positive_percentage = (count[1] / total) * 100 if total > 0 else 0
    negative_percentage = (count[-1] / total) * 100 if total > 0 else 0
    neutral_percentage = (count[0] / total) * 100 if total > 0 else 0
    return count, positive_percentage, negative_percentage, neutral_percentage

# Function to plot sentiment distribution
def plot_sentiment_distribution(predictions):
    labels, counts = zip(*Counter(predictions).items())
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    counts_dict = {label: 0 for label in sentiment_labels}
    
    for label, count in zip(labels, counts):
        if label == -1:
            counts_dict['Negative'] += count
        elif label == 0:
            counts_dict['Neutral'] += count
        elif label == 1:
            counts_dict['Positive'] += count
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(sentiment_labels, [counts_dict[label] for label in sentiment_labels], align='center', alpha=0.5)
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')

    st.pyplot(fig)

# Function to display word cloud
def display_word_cloud(texts):
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(' '.join(texts))
    plt.figure(figsize=(6, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt.gcf())  # Use plt.gcf() to get the current figure

# Function to plot line chart using Plotly
def plot_line_chart(data):
    df = pd.DataFrame(data, columns=['Feedback', 'Score'])
    fig = px.line(df, x='Feedback', y='Score', title='Sentiment Scores Over Time')
    st.plotly_chart(fig)

# Function to perform voice recognition and execute commands
def process_voice_command():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    with microphone as source:
        st.warning('Listening...')
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    st.warning('Processing...')
    
    try:
        command = recognizer.recognize_google(audio).lower()
        st.info(f'Command recognized: {command}')
        
        if 'show me last five feedback' in command or 'show last five feedback' in command:
            # Implement logic to fetch and display last 5 feedback
            st.success('Fetching last 5 feedback...')
            feedback_df = load_feedback()
            if feedback_df is not None and len(feedback_df) > 0:
                st.dataframe(feedback_df.tail(5))  # Display last 5 feedback
            else:
                st.info('No feedback stored yet.')
        
        elif 'count positive feedback' in command:
            # Implement logic to count positive feedback
            st.success('Counting positive feedback...')
            feedback_df = load_feedback()
            if feedback_df is not None:
                count_positive = (feedback_df['Predicted_Label'] == 1).sum()
                st.write(f'Number of positive feedback: {count_positive}')
            else:
                st.info('No feedback stored yet.')
        
        elif 'count negative feedback' in command:
            # Implement logic to count negative feedback
            st.success('Counting negative feedback...')
            feedback_df = load_feedback()
            if feedback_df is not None:
                count_negative = (feedback_df['Predicted_Label'] == -1).sum()
                st.write(f'Number of negative feedback: {count_negative}')
            else:
                st.info('No feedback stored yet.')
        
        elif 'count neutral feedback' in command:
            # Implement logic to count neutral feedback
            st.success('Counting neutral feedback...')
            feedback_df = load_feedback()
            if feedback_df is not None:
                count_neutral = (feedback_df['Predicted_Label'] == 0).sum()
                st.write(f'Number of neutral feedback: {count_neutral}')
            else:
                st.info('No feedback stored yet.')
        
        else:
            st.warning('Command not recognized. Please try again.')
    
    except sr.UnknownValueError:
        st.error("Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")

# Main Streamlit application
def main():
    st.title('Sentiment Analysis Application')

    # Collect user feedback
    user_input = st.text_area('Enter your feedback:')
    
    if st.button('Analyze'):
        if user_input:
            # Predict sentiment and get probabilities
            prediction, prediction_proba = predict_sentiment(user_input)
            
            # Extract probabilities for each class
            negative_proba = prediction_proba[0][-1]
            neutral_proba = prediction_proba[0][0]
            positive_proba = prediction_proba[0][1]

            # Display sentiment result
            st.subheader('Sentiment Analysis Result:')
            st.write(f'Feedback: {user_input}')
            st.write(f'Predicted Sentiment: {"Positive" if prediction == 1 else "Negative" if prediction == -1 else "Neutral"}')

            # Display sentiment probabilities (in percentage)
            total_proba = negative_proba + neutral_proba + positive_proba
            if total_proba > 0:
                st.subheader('Sentiment Probabilities (in %):')
                st.write(f'Negative: {negative_proba * 100 / total_proba:.2f}%')
                st.write(f'Neutral: {neutral_proba * 100 / total_proba:.2f}%')
                st.write(f'Positive: {positive_proba * 100 / total_proba:.2f}%')

            # Plot sentiment distribution
            predictions = np.argmax(prediction_proba, axis=1) - 1  # Convert probabilities to classes
            plot_sentiment_distribution(predictions)

            # Word cloud and line plot
            feedback_data = [('Feedback', user_input)]
            st.subheader('Additional Visualizations:')
            col1, col2 = st.beta_columns(2)
            with col1:
                display_word_cloud(user_input.split())
            with col2:
                plot_line_chart(feedback_data)
            
            # Store feedback along with predicted label
            store_feedback([user_input], [prediction])
        
        else:
            st.warning('Please enter some feedback.')
    
    # Voice command processing
    st.sidebar.subheader('Voice Command Processing')
    if st.sidebar.button('Activate Voice Command'):
        process_voice_command()

if __name__ == "__main__":
    main()
