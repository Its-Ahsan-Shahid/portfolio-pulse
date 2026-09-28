"""Pydantic schemas for the public API."""
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=12)
    company_name: str = Field(min_length=1, max_length=120)
    query: str = Field(default="", max_length=500)
    days: int = Field(default=25, ge=1, le=90)


class JobInfo(BaseModel):
    job_id: str
    status: str  # queued|fetching|preprocessing|embedding|agents|done|error
    progress: int = 0
    stage: str = ""
    message: str = ""
    error: str = ""
