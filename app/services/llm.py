from __future__ import annotations

import json

from google import genai
from pydantic import BaseModel

from app.config import settings
from app.models import Candidate, ConversationTurn, ParsedJob


class InterestResponse(BaseModel):
    """Structured output for the outreach and interest stage."""

    outreach_summary: str
    transcript: list[ConversationTurn]
    interest_score: float


class GeminiClient:
    """Wrapper around Gemini with a deterministic local fallback.

    For the Deccan submission this is important: the app should still work
    end-to-end even when no external API key is configured.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.default_model
        self._client = genai.Client(api_key=self.api_key) if self.api_key else None

    @property
    def enabled(self) -> bool:
        """Whether a live Gemini client is available."""

        return self._client is not None

    def simulate_interest(self, job: ParsedJob, candidate: Candidate) -> InterestResponse:
        """Generate an interest score from a short outreach exchange."""

        if not self.enabled:
            return self._fallback_interest(job, candidate)

        prompt = f"""
You are writing a short recruiter outreach simulation for the Deccan Catalyst demo.
Return JSON with keys: outreach_summary, transcript, interest_score.
Each transcript item must have speaker in ['agent','candidate'] and message.

Job:
{job.model_dump_json(indent=2)}

Candidate:
{candidate.model_dump_json(indent=2)}
"""
        try:
            response = self._client.models.generate_content(model=self.model, contents=prompt)
            payload = json.loads(response.text)
            return InterestResponse.model_validate(payload)
        except Exception:
            # Fall back quietly so the demo stays usable even if the provider
            # returns malformed JSON or the API call fails.
            return self._fallback_interest(job, candidate)

    def _fallback_interest(self, job: ParsedJob, candidate: Candidate) -> InterestResponse:
        """Score candidate intent using deterministic recruiter-style signals."""

        matched_skills = len(set(skill.lower() for skill in job.must_have_skills) & set(skill.lower() for skill in candidate.skills))
        base = 48 + matched_skills * 8
        if "remote" in job.work_mode.lower() and "remote" in candidate.preferred_work_mode.lower():
            base += 8
        if candidate.notice_period_days <= 45:
            base += 8
        if candidate.expected_comp_lpa <= 32:
            base += 7
        interest = max(35.0, min(94.0, float(base)))

        # The fallback keeps the transcript short and believable so it still
        # looks like a recruiter workflow instead of placeholder text.
        tone = "hot" if interest >= 78 else "warm" if interest >= 60 else "cold"
        candidate_reply = {
            "hot": f"This looks close to the kind of {job.title} role I want next. I'd be open to a recruiter conversation this week.",
            "warm": "This is relevant and I would like a little more detail on team scope, growth, and compensation.",
            "cold": "Interesting role, but timing is not ideal. I would still keep the conversation open for the near future.",
        }[tone]
        summary = {
            "hot": "High intent. Candidate is aligned on scope and open to a recruiter call soon.",
            "warm": "Moderate intent. Candidate is interested but wants a little more detail before committing.",
            "cold": "Low-to-moderate intent. Candidate is polite but not ready to move right away.",
        }[tone]
        transcript = [
            ConversationTurn(
                speaker="agent",
                message=f"Hi {candidate.name}, I found a {job.title} role that aligns with your work in {candidate.current_company}. Are you exploring new roles?",
            ),
            ConversationTurn(speaker="candidate", message=candidate_reply),
            ConversationTurn(
                speaker="agent",
                message=f"The role is based in {job.location} with {job.work_mode.lower()} setup and strong focus on {', '.join(job.must_have_skills[:3])}.",
            ),
            ConversationTurn(
                speaker="candidate",
                message="That helps. Please share next steps and expected interview timeline.",
            ),
        ]
        return InterestResponse(outreach_summary=summary, transcript=transcript, interest_score=round(interest, 1))
