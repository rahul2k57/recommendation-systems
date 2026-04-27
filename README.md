Hybrid Movie Recommendation System

A dual-engine machine learning application designed to provide highly personalized movie suggestions. By fusing Collaborative Filtering (Matrix Factorization) and Content-Based Filtering (TF-IDF), this system effectively mitigates the "Cold Start" problem and provides mathematically grounded recommendations.

Key Performance Metrics:

Model Accuracy (RMSE): 0.8491

Dataset: MovieLens 1M (1 Million Ratings)

Strategy Coverage: 100% (Handles both known users and new/niche content)

Technical Methodology:

Collaborative Filtering (Truncated SVD):

The "Brain" of the system identifies latent patterns in user behavior:

Mean Centering: A critical preprocessing step where user rating biases are removed. This ensures '0' represents a user's average taste rather than a missing value, significantly lowering the RMSE.

Matrix Factorization: Uses Truncated SVD to compress the sparse User-Movie matrix into 50 latent features, capturing the "hidden vibes" of movies and users.

Content-Based Filtering (TF-IDF & Cosine Similarity):

The "Thematic" engine analyzes item metadata:

Genre Vectorization: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert movie genres into a mathematical vector space.

Similarity Scoring: Employs Cosine Similarity to calculate the angular distance between movies, finding thematically identical films regardless of user rating history.

The Hybrid Strategy:

The system operates as a Weighted Hybrid, allowing it to:

Use SVD to suggest movies based on "People who liked this also liked..."

Use Content-Based logic to find movies with similar genres, ensuring the system remains reliable even when user data is sparse.


Tech Stack and Tools:

Python: (Version 3.9 or higher)

Scikit-Learn: SVD implementation, TF-IDF Vectorization, and Mean Squared Error calculation.

Streamlit: Real-time, interactive Web Interface.

Pandas & NumPy: Matrix manipulation and linear algebra.

Matplotlib & Seaborn: Exploratory Data Analysis (EDA) of movie popularity and genre distributions.


### How to Run Locally:

**Clone the repository:**

```
git clone https://github.com/rahul2k57/financial-fraud-detection-system.git
```
cd financial-fraud-detection-system

**Install dependencies:**

```
pip install -r requirements.txt
```
**Download the Dataset:**

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection) and and place it in the project's root directory.

Launch the Streamlit App:

```
streamlit run app.py
```
