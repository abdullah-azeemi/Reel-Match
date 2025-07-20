from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.model_loader import load_model
from app.recommender import get_top_k_recommendations, Recommender

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings, movie_df, movie_id_to_index, index_to_movie_id = load_model()
recommender = Recommender(embeddings, movie_id_to_index, index_to_movie_id)

class RecommendationRequest(BaseModel):
    movie_id: int
    k: int = 5

class RatedMovie(BaseModel):
    movie_id: int
    rating: float

class RatingRequest(BaseModel):
    rated_movies: list[RatedMovie]
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

@app.post("/recommend-by-ratings")
def recommend_movies(data: RatingRequest):
    try:
        recommendations = recommender.recommend_by_ratings(
            [rm.dict() for rm in data.rated_movies], data.k
        )
        return {"recommended_movies": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/movies")
def get_movies():
    return movie_df[['id', 'title']].to_dict(orient="records")
