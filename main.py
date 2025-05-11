# main.py

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
from uuid import UUID
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.cluster import KMeans
from sqlalchemy import create_engine
from datetime import datetime, date

# Load environment variables
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

# Create SQLAlchemy engine for MySQL
DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
engine = create_engine(DATABASE_URL, echo=False)

app = FastAPI(title="AquaFocus AI Stats API")

# --- Pydantic Models ---

class WeeklyProgress(BaseModel):
    M: float
    T: float
    W: float
    T2: float
    F: float
    S: float
    S2: float

class StatsResponse(BaseModel):
    user_id: UUID
    today_focus_time: str
    completed_pomodoros: int
    focus_rate: str
    weekly_progress: WeeklyProgress
    avg_session_duration: float
    avg_break_duration: float
    avg_productivity_score: float
    avg_focus_level: float
    total_tasks_completed: int
    study_report: List[str]

class Session(BaseModel):
    session_id: UUID
    reef_id: UUID
    session_duration: int
    break_duration: int
    productivity_score: float
    focus_level: float
    session_start: datetime
    session_end: datetime

# --- Utility Functions ---

def load_data() -> pd.DataFrame:
    """Fetch all pomodoro_sessions from the database and parse dates."""
    df = pd.read_sql("SELECT * FROM pomodoro_sessions", engine)
    df['session_start'] = pd.to_datetime(df['session_start'])
    df['session_end']   = pd.to_datetime(df['session_end'])
    return df
    
def load_tasks() -> pd.DataFrame:
    """Fetch all user_tasks."""
    df = pd.read_sql("SELECT * FROM user_tasks", engine)
    # assume status field exists; parse dates if needed
    return df
    
def compute_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-user averages and counts, then cluster all users into 3 groups
    (or skip clustering if fewer than 3 users exist).
    """
    stats = (
        df.groupby("user_id")
          .agg(
              avg_session_duration=("session_duration", "mean"),
              avg_break_duration=("break_duration", "mean"),
              avg_productivity_score=("productivity_score", "mean"),
              avg_focus_level=("focus_level", "mean"),
              completed_pomodoros=("session_id", "count")
          )
          .reset_index()
    )
    if len(stats) >= 3:
        km = KMeans(n_clusters=3, random_state=42)
        features = stats[[
            "avg_session_duration",
            "avg_break_duration",
            "avg_productivity_score",
            "avg_focus_level",
            "completed_pomodoros"
        ]]
        stats["cluster"] = km.fit_predict(features)
    else:
        stats["cluster"] = 0
    return stats

def generate_recommendation(user_row: pd.Series) -> List[str]:
    """Produce human-friendly recommendations based on the user's cluster and stats."""
    recs: List[str] = []
    c = user_row["cluster"]
    # Cluster-based top-level advice
    if c == 2:
        recs.append("Excellent performance! Maintain your current routine and consistency.")
    elif c == 1:
        recs.append("Your performance is average; try optimizing your environment to reduce distraction.")
    else:
        recs.append("Your overall performance is low; consider reducing session duration and increasing breaks.")
    # Session duration advice
    if user_row["avg_session_duration"] < 10:
        recs.append("Your work sessions are too short, try longer Pomodoro cycles")
    elif user_row["avg_session_duration"] > 60:
        recs.append("Your work sessions are long, try shorter Pomodoro cycles")
    else:
        recs.append("Session length looks good.")
    # Break duration advice
    if user_row["avg_break_duration"] < 5:
        recs.append("Consider extending your break duration to improve focus.")
    else:
        recs.append("Your break durations are well balanced.")
    
    if user_row["avg_productivity_score"] < 60:
        recs.append("Your productivity seems low; experiment with new techniques.")
    else:
        recs.append("Your productivity scores are good.")
    if user_row["avg_focus_level"] < 6:
        recs.append("Your focus level is below average; consider concentration improvement techniques.")
    else:
        recs.append("Your focus level is impressive!")
    return recs

def get_weekly_progress(df: pd.DataFrame, user_id: UUID) -> dict:
    """
    Compute average session duration per weekday, normalize by 90 minutes,
    and return a dict of single-letter day keys to float percentages.
    """
    user_df = df[df.user_id == str(user_id)]
    grouped = user_df.groupby(user_df['session_start'].dt.strftime('%A'))['session_duration'].mean().to_dict()
    day_map = {
        'Monday': 'M', 'Tuesday': 'T', 'Wednesday': 'W',
        'Thursday': 'T2', 'Friday': 'F', 'Saturday': 'S', 'Sunday': 'S2'
    }
    output = {v: 0.0 for v in day_map.values()}
    for day_name, avg_dur in grouped.items():
        key = day_map.get(day_name)
        if key:
            output[key] = round(avg_dur / 90, 2)
    return output

# --- Endpoints ---

@app.get("/stats", response_model=StatsResponse, summary="Get user statistics & AI recommendations")
def stats(user_id: UUID = Query(..., description="The UUID of the user")):
    """
    Returns today's focus time, total sessions, focus rate,
    weekly progress, aggregate stats, and AI-generated study recommendations.
    """
    df = load_data()
    tasks_df    = load_tasks()
    
    stats_df = compute_user_stats(df)
    user_row = stats_df[stats_df.user_id == str(user_id)]
    if user_row.empty:
        raise HTTPException(status_code=404, detail="User not found")
    user = user_row.iloc[0]
    
    today = date.today()
    today_sessions = df[
        (df.user_id == str(user_id)) &
        (df.session_start.dt.date == today)
    ]
    today_focus_total = today_sessions["session_duration"].sum()
    
    # count completed tasks
    completed_tasks = int(
        tasks_df[
            (tasks_df.user_id == str(user_id)) &
            (tasks_df.status == "completed")
        ].shape[0]
    )
    weekly = get_weekly_progress(df, user_id)
    return {
        "user_id": user_id,
        "today_focus_time": f"{int(today_focus_total)} min",
        "completed_pomodoros": int(user.completed_pomodoros),
        "focus_rate": f"{round(user.avg_focus_level * 10)}%",
        "weekly_progress": weekly,
        "avg_session_duration": round(user.avg_session_duration, 2),
        "avg_break_duration": round(user.avg_break_duration, 2),
        "avg_productivity_score": round(user.avg_productivity_score, 2),
        "avg_focus_level": round(user.avg_focus_level, 2),
        "total_tasks_completed": completed_tasks,
        "study_report": generate_recommendation(user)
    }

@app.get("/sessions", response_model=List[Session], summary="List raw pomodoro sessions for a user")
def list_sessions(user_id: UUID = Query(..., description="The UUID of the user")):
    """
    Returns the list of all pomodoro session records for the given user.
    """
    df = load_data()
    user_df = df[df.user_id == str(user_id)]
    if user_df.empty:
        raise HTTPException(status_code=404, detail="No sessions found for this user")
    return user_df.to_dict(orient="records")

# To run:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
