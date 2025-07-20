from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model_loader import load_model
from app.recommender import get_top_k_recommendations

app = FastAPI()

embeddings, movie_df, movie_id_to_index, index_to_movie_id = load_model()

class RecommendationRequest(BaseModel):
    movie_id: int
    k: int = 5

@app.get("/")
def root():
    return {"message": "ReelMatch Recommender API is live!"}

@app.post("/recommend")
def recommend(req: RecommendationRequest):
    try:
        recs = get_top_k_recommendations(
            movie_id=req.movie_id,
            embeddings=embeddings,
            movie_id_to_index=movie_id_to_index,
            index_to_movie_id=index_to_movie_id,
            movie_df=movie_df,
            k=req.k
        )
        return {"recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/movies")
def get_movies():
    return movie_df[['id', 'title']].to_dict(orient="records")
