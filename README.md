# 🍿 Netflix Movie Recommendation System  

This project implements a **content-based movie recommendation system** using **K-Means clustering** on the **Netflix Dataset (2021)**. The system clusters movies based on their features (such as genre, language, and view ratings) and recommends similar movies to the one selected by the user.  

The application is deployed using **Streamlit** for an interactive web-based interface.  

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
  - TF-IDF Vectorization (for text features like movie titles/descriptions)  
  - Genre Encoding  
  - Language Encoding  
  - Standard Scaling  

### ✅ Model Evaluation  
We used **Silhouette Score** to evaluate the quality of clustering:  

- **Silhouette Score:** `0.08095`

## 🏆 Contributors  
- **Omar Ahmed**
- **Amr Awad**   
- **Marwan Amir**   
- **Rokia Hassan**   
- **Khaled Tarek**   
