import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def fetch_movie_details(movie_ids):
    results = []
    for movie_id in movie_ids:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        res = requests.get(url)
        if res.status_code == 200:
            results.append(res.json())
    return results
def get_recommendations(rated_movies, embeddings, movie_ids, k=5):
    if not rated_movies:
        return []
    
    user_vector = np.zeros(embeddings.shape[0])
    total_rating = 0
    
    for entry in rated_movies:
        movie_id, rating = entry["id"], entry["rating"]
        if movie_id in movie_ids:
            idx = movie_ids.index(movie_id)
            user_vector += embeddings[idx] * rating
            total_rating += rating
            
    if total_rating == 0:
        return []
        
    user_vector /= total_rating
    similarites = embeddings @ user_vector
    top_indicies = np.argsort(similarites)[::-1]
    rated_ids = set(x["id"] for x in rated_movies)
    recommended = []
    for idx in top_indicies:
        mid = movie_ids[idx]
        if mid not in rated_ids:
            recommended.append(mid)
        if len(recommended) >= k:
            break
        
    return fetch_movie_details(recommended)
        
        

def get_top_k_recommendations(
    movie_id: int,
    embeddings: np.ndarray,
    movie_id_to_index: dict,
    index_to_movie_id: dict,
    movie_df,
    k: int = 5
) -> List[dict]:
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
        poster_path = row.get("poster_path", None)

        recommendations.append({
            "movie_id": int(row["id"]),
            "title": row["title"],
            "genres": row.get("genres", []),
            "vote_average": float(row.get("vote_average", 0.0)),
            "poster_path": poster_path
        })

        if len(recommendations) >= k:
            break

    return recommendations

class Recommender:
    def __init__(self, movie_embeddings, movie_id_to_index, index_movie_to_id):
        self.movie_embeddings = movie_embeddings
        self.movie_id_to_index = movie_id_to_index
        self.index_movie_to_id = index_movie_to_id
    
    def recommend_by_ratings(self, rated_movies, k=10):
        vectors = []
        weights = []
        for item in rated_movies:
            movie_id = item["movie_id"]
            rating = item["rating"]
            if movie_id in self.movie_id_to_index:
                idx = self.movie_id_to_index[movie_id]
                vectors.append(self.movie_embeddings[idx])
                weights.append(rating)
                
        if not vectors:
            return []
        
        weighted_vector = np.average(vectors, axis=0, weights=weights).reshape(1,-1)
        similarities = cosine_similarity(weighted_vector, self.movie_embeddings).flatten()
        top_indicies = similarities.argsort()[::-1]
        
        recommended_ids = []
        for i in top_indicies:
            candidate_id = self.index_movie_to_id[i]
            if candidate_id not in [item["movie_id"] for item in rated_movies]:
                recommended_ids.append(candidate_id)
            if len(recommended_ids) == k:
                break
        
        return recommended_ids
            
            