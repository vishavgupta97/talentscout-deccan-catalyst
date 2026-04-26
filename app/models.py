from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class JobInput(BaseModel):
    """Raw recruiter input captured from the form or API."""

    title: str = Field(..., min_length=2, max_length=120)
    company: str = Field(default="Deccan Hiring Team", max_length=120)
    location: str = Field(default="Remote - India", max_length=120)
    employment_type: str = Field(default="Full-time", max_length=40)
    description: str = Field(..., min_length=40)
    top_k: int = Field(default=5, ge=3, le=10)


class ParsedJob(BaseModel):
    """Normalized job data used by the ranking pipeline."""

    title: str
    seniority: str
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    domain_keywords: list[str]
    location: str
    work_mode: str
    min_years_experience: int
    summary: str


class Candidate(BaseModel):
    """Candidate profile stored in the sourced talent pool."""

    candidate_id: str
    name: str
    title: str
    location: str
    preferred_work_mode: str
    years_experience: int
    expected_comp_lpa: int
    notice_period_days: int
    current_company: str
    domain_experience: list[str]
    skills: list[str]
    achievements: list[str]
    summary: str


class MatchExplanation(BaseModel):
    """Human-readable reasons attached to the score."""

    highlights: list[str]
    gaps: list[str]
    rationale: str


class ConversationTurn(BaseModel):
    """One message in the outreach simulation."""

    speaker: Literal["agent", "candidate"]
    message: str


class CandidateEvaluation(BaseModel):
    """Final scoring bundle shown to the recruiter."""

    candidate: Candidate
    match_score: float
    interest_score: float
    final_score: float
    explanation: MatchExplanation
    outreach_summary: str
    transcript: list[ConversationTurn]
    status: Literal["hot", "warm", "cold"]


class PipelineResult(BaseModel):
    """Full response returned by the LangGraph pipeline."""

    job: ParsedJob
    ranked_candidates: list[CandidateEvaluation]
    recruiter_summary: str


class PipelineState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    job_input: JobInput
    parsed_job: ParsedJob
    candidates: list[Candidate]
    ranked_candidates: list[CandidateEvaluation]
    recruiter_summary: str
