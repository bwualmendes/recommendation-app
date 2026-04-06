import streamlit as st
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# =========================================
# CONFIG
# =========================================
st.set_page_config(page_title="Product Recommender", layout="centered")

# =========================================
# LOAD DATA + MODEL
# =========================================
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv('final_recommendation_dataset.csv')
    user_item_matrix = df.pivot_table(
        index='userId',
        columns='productId',
        values='rating'
    ).fillna(0)
    matrix_sparse = csr_matrix(user_item_matrix.values)
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    model.fit(matrix_sparse)
    return df, user_item_matrix, model

df, user_item_matrix, model = load_model_and_data()

# =========================================
# HEADER
# =========================================
st.title("Product Recommendation System")
st.caption("KNN-based collaborative filtering")
st.divider()

# =========================================
# INPUT
# =========================================
col1, col2 = st.columns([2, 1])
with col1:
    user_id = st.text_input("User ID")
with col2:
    n = st.number_input("Results", min_value=5, max_value=20, value=10)

st.divider()

# =========================================
# RECOMMENDATION FUNCTION (SAFE)
# =========================================
def get_recommendations(user_id, n=10):
    try:
        user_index = user_item_matrix.index.get_loc(user_id)
    except KeyError:
        return None

    user_vector = csr_matrix(user_item_matrix.iloc[user_index].values)
    distances, indices = model.kneighbors(user_vector, n_neighbors=11)
    similar_users = user_item_matrix.index[indices.flatten()[1:]]

    rated_by_user = set(df[df['userId'] == user_id]['productId'])
    sim_data = df[df['userId'].isin(similar_users)]
    sim_data = sim_data[~sim_data['productId'].isin(rated_by_user)]

    if sim_data.empty:
        return None

    avg_ratings = (
        sim_data.groupby('productId')['rating']
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    avg_ratings.columns = ['productId', 'Predicted Rating']

    product_info = df[['productId', 'product_name', 'brand', 'category', 'price']].drop_duplicates()
    result_df = avg_ratings.merge(product_info, on='productId', how='left')
    return result_df

# =========================================
# BUTTON ACTION
# =========================================
if st.button("Get Recommendations"):
    user_id = user_id.strip()
    if user_id == "":
        st.warning("⚠️ Enter a valid User ID")
    elif user_id not in user_item_matrix.index:
        st.error("❌ User not found")
    else:
        with st.spinner("Finding recommendations..."):
            result_df = get_recommendations(user_id, n)
        if result_df is None:
            st.info("No recommendations available")
        else:
            st.subheader("Recommended Products")
            for i, row in result_df.iterrows():
                st.markdown(f"""
**{i+1}. {row['product_name']}**  
Brand: {row['brand']}  
Category: {row['category']}  
Price: ₹{row['price']}  
Rating: {row['Predicted Rating']}

---
""")

# =========================================
# FOOTER
# =========================================
st.caption("Built using KNN Collaborative Filtering")