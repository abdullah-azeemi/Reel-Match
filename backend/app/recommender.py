import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

def get_top_k_recommendations(movie_id: int, embeddings: np.ndarray, movie_id_to_index: dict, index_to_movie_id: dict, movie_df, k: int = 5) -> List[dict]:
    if movie_id not in movie_id_to_index:
        raise ValueError("Movie ID not found in database")

    idx = movie_id_to_index[movie_id]
    movie_vector = embeddings[idx].reshape(1, -1)

    similarities = cosine_similarity(movie_vector, embeddings).flatten()
    top_indices = similarities.argsort()[-(k+1):][::-1]  

    recommendations = []
    for i in top_indices:
        if i == idx:  
            continue
        recommended_id = index_to_movie_id[i]
        row = movie_df[movie_df['id'] == recommended_id].iloc[0]
        recommendations.append({
            "movie_id": int(row["id"]),
            "title": row["title"],
            "genres": row.get("genres", []),
            "vote_average": float(row.get("vote_average", 0.0))
        })

        if len(recommendations) == k:
            break

    return recommendations
