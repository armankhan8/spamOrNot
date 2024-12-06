# **Spam Email Detection System**

This is a machine learning-based web application built to classify emails as **Spam** or **Not Spam**. The project uses **Flask**, **scikit-learn**, and **Natural Language Processing (NLP)** techniques to predict whether an email is spam based on its content.

## **Features**
- Users can input an email through the web interface.
- The system predicts whether the email is **Spam** or **Not Spam** using a trained model.
- The app uses **TF-IDF Vectorization** for text feature extraction and **Logistic Regression** for classification.

## **Technologies Used**
- **Python**: The main programming language for the backend.
- **Flask**: For building the web application.
- **scikit-learn**: For machine learning and model training.
- **Pandas**: For data handling and preprocessing.
- **HTML/CSS**: For front-end design.
- **Natural Language Processing (NLP)**: Using **TF-IDF Vectorization** to process email text data.

## **How It Works**
1. **Data Preprocessing**: The dataset is cleaned by removing duplicates and handling missing values. Emails are converted to lowercase for consistency.
2. **Feature Extraction**: The text of the email is transformed into numerical vectors using **TF-IDF** (Term Frequency-Inverse Document Frequency) to capture important words.
3. **Model Training**: A **Logistic Regression** model is trained using the processed data to classify emails into **Spam** or **Not Spam**.
4. **Prediction**: The model takes the email input, processes it, and predicts whether it's spam or not.

## **Setup & Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/spam-email-detection.git
   cd spam-email-detection

