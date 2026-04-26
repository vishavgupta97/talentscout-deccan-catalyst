from __future__ import annotations

import math
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import Candidate, MatchExplanation, ParsedJob


TOKEN_RE = re.compile(r"[A-Za-z0-9+#.]+")


def normalize_list(values: list[str]) -> list[str]:
    """Normalize free-form string lists before matching."""

    return [value.strip().lower() for value in values if value.strip()]


def tokenize(text: str) -> list[str]:
    """Tokenize text with a lightweight regex tokenizer."""

    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def semantic_similarity(job: ParsedJob, candidate: Candidate) -> float:
    """Approximate semantic fit using TF-IDF over job and profile text.

    This is intentionally simple for a hackathon submission: it is cheap,
    deterministic, and easy to explain to judges.
    """

    docs = [
        " ".join(
            [
                job.summary,
                " ".join(job.must_have_skills),
                " ".join(job.nice_to_have_skills),
                " ".join(job.domain_keywords),
            ]
        ),
        " ".join(
            [
                candidate.summary,
                " ".join(candidate.skills),
                " ".join(candidate.domain_experience),
                " ".join(candidate.achievements),
            ]
        ),
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(docs)
    similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return float(similarity)


def overlap_score(required: list[str], candidate_values: list[str]) -> float:
    """Exact-match overlap for must-have skills."""

    if not required:
        return 1.0
    candidate_set = set(normalize_list(candidate_values))
    hits = sum(1 for skill in normalize_list(required) if skill in candidate_set)
    return hits / len(required)


def partial_overlap_score(required: list[str], candidate_values: list[str]) -> float:
    """Substring overlap for looser domain and nice-to-have matching."""

    if not required:
        return 1.0
    candidate_text = " ".join(normalize_list(candidate_values))
    hits = 0
    for item in normalize_list(required):
        if item in candidate_text:
            hits += 1
    return hits / len(required)


def location_score(job: ParsedJob, candidate: Candidate) -> float:
    """Score how closely the candidate's preferred mode matches the role."""

    job_location = job.location.lower()
    candidate_location = candidate.location.lower()
    if "remote" in job.work_mode.lower():
        return 1.0 if "remote" in candidate.preferred_work_mode.lower() else 0.8
    if job_location == candidate_location:
        return 1.0
    if job_location.split(",")[0] in candidate_location:
        return 0.8
    return 0.45


def experience_score(job: ParsedJob, candidate: Candidate) -> float:
    """Reward candidates who meet or slightly exceed the stated baseline."""

    delta = candidate.years_experience - job.min_years_experience
    if delta >= 0:
        return min(1.0, 0.8 + min(delta, 5) * 0.04)
    return max(0.25, 1.0 + delta * 0.15)


def notice_period_score(candidate: Candidate) -> float:
    """Shorter notice periods are easier for recruiters to act on quickly."""

    return max(0.2, 1 - (candidate.notice_period_days / 180))


def compensation_signal(candidate: Candidate) -> float:
    """Keep a simple compensation heuristic available for future weighting."""

    if candidate.expected_comp_lpa <= 18:
        return 0.9
    if candidate.expected_comp_lpa <= 28:
        return 0.75
    if candidate.expected_comp_lpa <= 38:
        return 0.55
    return 0.35


def score_candidate(job: ParsedJob, candidate: Candidate) -> tuple[float, MatchExplanation]:
    """Calculate the match score and generate recruiter-facing explanations."""

    hard_skill = overlap_score(job.must_have_skills, candidate.skills)
    nice_skill = partial_overlap_score(job.nice_to_have_skills, candidate.skills)
    domain_fit = partial_overlap_score(job.domain_keywords, candidate.domain_experience)
    experience_fit = experience_score(job, candidate)
    logistics_fit = (location_score(job, candidate) + notice_period_score(candidate)) / 2
    semantic_fit = semantic_similarity(job, candidate)

    # The weights mirror the logic described in the submission docs so the
    # implementation and explanation stay aligned during the demo.
    match_score = (
        hard_skill * 35
        + experience_fit * 20
        + domain_fit * 15
        + logistics_fit * 15
        + semantic_fit * 15
    )

    highlights = []
    gaps = []
    candidate_skills = Counter(normalize_list(candidate.skills))
    for skill in normalize_list(job.must_have_skills):
        if skill in candidate_skills:
            highlights.append(f"Strong skill match on {skill}")
        else:
            gaps.append(f"Missing direct evidence for {skill}")

    if candidate.years_experience >= job.min_years_experience:
        highlights.append(f"{candidate.years_experience} years experience exceeds baseline")
    else:
        gaps.append(f"Experience is {job.min_years_experience - candidate.years_experience} years below target")

    if any(domain in " ".join(normalize_list(candidate.domain_experience)) for domain in normalize_list(job.domain_keywords)):
        highlights.append("Relevant domain exposure present")
    else:
        gaps.append("Limited adjacent domain evidence")

    rationale = (
        f"{candidate.name} shows strong alignment on {max(1, math.ceil(hard_skill * len(job.must_have_skills)))} "
        f"must-have skills with semantic profile similarity of {semantic_fit:.2f}."
    )

    return round(match_score, 1), MatchExplanation(highlights=highlights[:4], gaps=gaps[:3], rationale=rationale)
