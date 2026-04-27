import streamlit as st
import pandas as pd
# Import the logic and variables from your recommender.py file
try:
    from recommendation import get_hybrid_dashboard, rmse
except ImportError:
    st.error("Could not find 'recommender.py'. Please ensure both files are in the same folder.")

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Hybrid Recommendation Engine", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #3498db;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title(" Hybrid Movie Recommendation System")
st.markdown("""
This system utilizes a **Weighted Hybrid Approach** to solve the 'Cold Start' and sparsity problems. 
It fuses **Collaborative Filtering (SVD)** with **Content-Based Filtering (TF-IDF)** for optimal results.
""")

# --- SIDEBAR: SYSTEM RELIABILITY & METRICS ---
st.sidebar.header("📊 System Performance")
st.sidebar.metric(label="Model Accuracy (RMSE)", value=f"{rmse:.4f}")
st.sidebar.write("---")

# --- MAIN INTERFACE ---
st.subheader("🔍 Discover Personalized Recommendations")
movie_input = st.text_input("Enter a movie title (e.g., 'Toy Story' or 'Jurassic Park'):", "Toy Story")

if st.button("Generate Hybrid Recommendations"):
    with st.spinner('Running multi-engine analysis...'):
        # Calling the recommendation functions from recommender.py
        results = get_hybrid_dashboard(movie_input)
        
        # Checking if the function returned an error string or the dataframes
        if isinstance(results, str):
            st.error(results)
        else:
            collab_df, content_df = results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("💡 Collaborative Filtering (User Behavior)")
                st.markdown("*People who liked this also liked:*")
                st.dataframe(collab_df, use_container_width=True)
                
            with col2:
                st.info("🎨 Content Based Filtering (Genre Metadata)")
                st.markdown("*Thematically similar movies:*")
                st.dataframe(content_df, use_container_width=True)

# --- FOOTER ---
st.divider()