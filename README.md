# Project title: 
    Amazon product review using RNNs,LSTM,GRU


# Intro:
   Amazon product reviews are user-generated feedback that reflect customer experiences with products purchased on Amazon.
   These reviews typically include a star rating (1 to 5) and a written comment. They serve as a valuable source of information,
   for both consumers and businesses. Analyzing these reviews helps in understanding customer satisfaction, improving product quality
   and enhancing marketing strategies. With the power of Natural Language Processing (NLP), we can classify these reviews as,
   positive or negative to gain actionable insights


## 🚀 Features:
     - Cleaned and preprocessed text data
     - Word2Vec Embedding integration
     - RNN, LSTM, and GRU model implementations
     - Evaluation of model performance
     - Sentiment prediction function
     - Web deployment using Streamlit/FastAPI


 ## 🧰 Tools & Libraries Used
      Python
      TensorFlow / Keras - for deep learning models
      Spacy - for text preprocessing
      NumPy / Pandas - for data handling
      Matplotlib for visulization


 ## 🧹 Text Preprocessing Steps
       Lowercasing text
       Removing punctuation, stopwords, and special characters
       Tokenizing and padding sequences
       Creating a vocabulary and embedding matrix using pretrained word vectors (optional)


## 🧠 Model Architectures:

## 1️⃣ RNN
     Embedding Layer (non-trainable or pretrained)
     Simple RNN Layer (with dropout)
     Dense Layer with ReLU
     Output Layer with Sigmoid

## 2️⃣ LSTM
     Embedding Layer 
     LSTM Layer(s) with return_sequences
     Dense + Dropout
     Output Layer (Sigmoid for binary classification)
     You can also experiment with GRU and Bidirectional layers for better performance.

## 3️⃣ GRU
    Embedding Layer
    GRU Layer(s) (with or without return_sequences=True depending on stacking)
    Dense Layer with ReLU activation
    Dropout Layer to prevent overfitting
    Output Layer with Sigmoid activation (for binary classification)




## 🧠 Models Used

- ✅ **Simple RNN**
- ✅ **LSTM**
- ✅ **GRU**

# Exmaple:

predict_sentiment("This product is amazing and exceeded my expectations!")
   Output: Positive


## 🚀 How It Works
   Load and clean the Amazon reviews dataset
   Preprocess the text and prepare sequences
   Build and train deep learning models
   Evaluate accuracy, loss, and F1-score
   Deploy model using Streamlit to predict sentiment of new reviews

# Clone the repository:
   git clone https://github.com/MrHasnain2522/amazon-product-reviews-nlp.git
