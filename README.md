# Twitter Sentiment Analysis and Text Preprocessing

## Overview
This project analyzes sentiment in tweets using the TextBlob library and applies various text preprocessing techniques. Additionally, it explores different text vectorization methods such as Bag-of-Words (BoW), TF-IDF, Word2Vec, and GloVe embeddings. The similarity between original and processed text representations is also calculated using cosine similarity.

## Dataset
- The dataset used is `training.1600000.processed.noemoticon.csv` which consists of **50,000** reviews categorized into positive and negative classes. You can access and download the dataset from the following link:

[training.1600000.processed.noemoticon.csv Dataset](https://drive.google.com/file/d/13u7afZIUzeTo2RaL6SkVApdR09zRGws-/view?usp=sharing)
- Columns: `id`, `timestamp`, `query`, `user`, `tweet`.
- Sentiment is inferred using TextBlob instead of being explicitly provided.

## Project Components
### 1. **Sentiment Classification**
- The `classify_sentiment` function assigns a sentiment label to each tweet based on polarity:
  - Positive (>0)
  - Negative (<0)
  - Neutral (0)
- Sentiment distribution is visualized using a bar chart.

### 2. **Text Preprocessing**
The `preprocess_text` function includes:
- Contraction expansion
- Lowercasing
- URL removal
- Mention removal
- Unicode normalization
- Punctuation removal
- Tokenization
- Word distribution analysis before and after preprocessing

### 3. **Feature Extraction**
#### **Bag-of-Words (BoW)**
- The `create_bow_matrix` function uses `CountVectorizer` (unigram & bigram) to generate a matrix representation.
- The top 10 frequent words are displayed.

#### **TF-IDF (Term Frequency-Inverse Document Frequency)**
- The `create_tfidf_matrix` function applies `TfidfVectorizer` to extract term importance.
- The top 10 important words are displayed.

#### **Word2Vec Embeddings**
- The `create_word2vec` function trains a Word2Vec model on tokenized tweets.
- Generates a 10-dimensional embedding vector for each tweet.

#### **GloVe Embeddings (Simulated using Word2Vec Model)**
- The `create_glove` function approximates GloVe-like embeddings using Word2Vec.
- Generates a 10-dimensional embedding vector for each tweet.

### 4. **Cosine Similarity Analysis**
- The similarity between original and preprocessed texts is measured using cosine similarity for both BoW and TF-IDF representations.
- Higher similarity indicates minimal data loss during preprocessing.

## Dependencies
```bash
pip install pandas numpy matplotlib textblob contractions unidecode gensim sklearn
```

## Running the Project
1. Load dataset: `df = pd.read_csv("training.1600000.processed.noemoticon.csv", names=column_names, encoding='latin1')`
2. Apply sentiment classification: `df['sentiment'] = df['tweet'].apply(classify_sentiment)`
3. Perform text preprocessing: `df['processed_tweet'] = df['tweet'].apply(preprocess_text)`
4. Generate feature vectors (BoW, TF-IDF, Word2Vec, GloVe)
5. Compute and compare cosine similarity between original and preprocessed texts.
6. Visualize results with matplotlib.

## Results & Insights
- Sentiment distribution of tweets is visualized in a bar chart.
- Preprocessing reduces noise and improves text consistency.
- Word frequency distributions show the most common words before and after preprocessing.
- Cosine similarity scores help assess how preprocessing affects text representation.

## Future Improvements
- Implement deep learning-based sentiment classification.
- Use pre-trained embeddings (e.g., GloVe, FastText) for improved feature extraction.
- Experiment with different similarity metrics beyond cosine similarity.



