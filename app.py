import streamlit as st
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# =========================================
# LOAD DATA + TRAIN MODEL (NO .pkl)
# =========================================
@st.cache_resource
def load_model_and_data():
    # Load dataset
    df = pd.read_csv('final_recommendation_dataset.csv')

    # Create user-item matrix
    user_item_matrix = df.pivot_table(
        index='userId',
        columns='productId',
        values='rating'
    ).fillna(0)

    # Train model inside app (🔥 FIXED)
    matrix_sparse = csr_matrix(user_item_matrix.values)

    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    model.fit(matrix_sparse)

    return df, user_item_matrix, model


df, user_item_matrix, model = load_model_and_data()

# =========================================
# PAGE SETTINGS
# =========================================
st.set_page_config(page_title="Product Recommender", page_icon="🛍️")

st.title("🛍️ Product Recommendation System")
st.write("Get personalized product recommendations using Machine Learning.")

# =========================================
# INPUT SECTION
# =========================================
user_id = st.text_input("Enter User ID")
n = st.slider("Number of Recommendations", 5, 20, 10)

# =========================================
# RECOMMENDATION FUNCTION
# =========================================
def get_recommendations(user_id, n=10):

    user_index = user_item_matrix.index.get_loc(user_id)
    user_vector = csr_matrix(user_item_matrix.iloc[user_index].values)

    # Find similar users
    distances, indices = model.kneighbors(user_vector, n_neighbors=11)
    similar_users = user_item_matrix.index[indices.flatten()[1:]]

    # Products already rated
    rated_by_user = set(df[df['userId'] == user_id]['productId'])

    # Get similar users' data
    sim_data = df[df['userId'].isin(similar_users)]

    # Remove already rated products
    sim_data = sim_data[~sim_data['productId'].isin(rated_by_user)]

    if sim_data.empty:
        return None

    # Average ratings
    avg_ratings = (
        sim_data.groupby('productId')['rating']
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )

    avg_ratings.columns = ['productId', 'Predicted Rating']

    # Product details
    product_info = df[['productId', 'product_name', 'brand', 'category', 'price']].drop_duplicates()

    # Merge
    result_df = avg_ratings.merge(product_info, on='productId', how='left')

    return result_df


# =========================================
# BUTTON ACTION
# =========================================
if st.button("Get Recommendations 🚀"):

    if user_id.strip() == "":
        st.warning("⚠️ Please enter a User ID.")

    elif user_id not in user_item_matrix.index:
        st.error("❌ User ID not found. Try a different one.")

    else:
        with st.spinner("🔍 Finding best products for you..."):

            result_df = get_recommendations(user_id, n)

            if result_df is None:
                st.warning("No recommendations found for this user.")
            else:
                st.success(f"🎯 Top {n} Recommended Products:")

                # Display nicely
                for i, row in result_df.iterrows():
                    st.markdown(f"""
                    ### {i+1}. {row['product_name']}
                    - **Brand:** {row['brand']}
                    - **Category:** {row['category']}
                    - **Price:** ₹{row['price']}
                    - ⭐ **Predicted Rating:** {row['Predicted Rating']}
                    """)

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.caption("📌 Built with KNN Collaborative Filtering | Machine Learning Project")