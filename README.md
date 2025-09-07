# 🍿 Netflix Movie Recommendation System  

This project implements a **content-based movie recommendation system** using **K-Means clustering** on the **Netflix Dataset (2021)**. The system clusters movies based on their features (such as genre, language, and view ratings) and recommends similar movies to the one selected by the user.  

The application is deployed using **Streamlit** for an interactive web-based interface.  

[![Demo Preview](assets/demo.gif)](https://https://github.com/Netflix-Movie-Recommendation-System/Movie_Recommendation_System/raw/main/assets/Demo.gif)
---

## 🚀 Features  
- 🎬 Recommend movies similar to a user-selected title.  
- 🔍 Sidebar controls to choose movies and number of recommendations.  
- 🎲 "Surprise Me" feature for random movie suggestions.  
- 📌 Personalized Watchlist functionality.  
- 🔥 Display of trending movies.  
- 🎨 Modern and clean UI with hover effects.  

---

## 🧑‍💻 Tech Stack  
- **Python** (core programming)  
- **Pandas** (data handling)  
- **Scikit-learn** (machine learning & clustering)  
- **Streamlit** (web app deployment)  
- **Pickle** (model & preprocessor saving/loading)  

---

## 📊 Machine Learning Approach  
- **Model Used**: K-Means Clustering  
- **Dimensionality Reduction**: PCA  
- **Feature Engineering**:  
  - TF-IDF Vectorization
  - Genre Encoding  
  - Language Encoding  
  - Standard Scaling
  -**Cosine Similarity**  

## K-Means Clustering
- We used K-Means to group movies into clusters of similar characteristics. This helps reduce the search space when recommending movies: instead of comparing a movie with the entire dataset, we only compare it with movies in the same cluster. This makes the system faster and more relevant.

## PCA (Principal Component Analysis)
- Our feature space was very high-dimensional (due to text, genres, encodings, etc.). PCA was used to reduce dimensionality while keeping most of the important variance in the data.

## TF-IDF Vectorization
- Movies have textual information such as titles and summaries. Instead of treating text as plain words, we applied TF-IDF Vectorization to give more weight to unique and meaningful words while reducing the importance of very common words (like "the", "movie", etc.). This way, text features contribute effectively to similarity.

## Genre Encoding
- Genres are categorical (e.g., "Action", "Drama"). We can’t feed text labels directly to a model, so we used One-Hot Encoding to convert each genre into numerical format. This allows clustering to treat genres as distinct features in the similarity calculation.

## Language Encoding
- Like genres, movie languages are categorical (e.g., English, Spanish, Hindi). We used Encoding to transform languages into numbers so that the model can recognize language as an important feature. This improves recommendation diversity (e.g., suggesting a Spanish movie if the user prefers Spanish content).

## Cosine Similarity
- Finally, to measure how close two movies are within the same cluster, we used Cosine Similarity. Unlike Euclidean distance, cosine similarity focuses on the direction of feature vectors instead of magnitude. This is crucial when comparing text and high-dimensional data, making similarity scores more meaningful.


### ✅ Model Evaluation  
We used **Silhouette Score** to evaluate the quality of clustering:  

- **Silhouette Score:** `0.08095`

## 🏆 Contributors  
- **Omar Ahmed**
- **Amr Awad**   
- **Marwan Amir**   
- **Rokia Hassan**   
- **Khaled Tarek**   
