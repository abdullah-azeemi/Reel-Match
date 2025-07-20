import pickle

def load_model(path: str = "models/movie_embeddings.pkl"):
    with open(path, "rb") as f:
        model_data = pickle.load(f)

    return model_data["embeddings"], model_data["movie_df"], model_data["movie_id_to_index"], model_data["index_to_movie_id"]
