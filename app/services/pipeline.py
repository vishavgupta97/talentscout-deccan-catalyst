from __future__ import annotations

import re
from typing import Iterable

from langgraph.graph import END, StateGraph

from app.models import CandidateEvaluation, JobInput, ParsedJob, PipelineResult, PipelineState
from app.services.llm import GeminiClient
from app.services.repository import CandidateRepository
from app.services.scoring import score_candidate


SKILL_VOCAB = {
    "python",
    "fastapi",
    "django",
    "flask",
    "langchain",
    "langgraph",
    "rag",
    "llm",
    "vector databases",
    "postgres",
    "sql",
    "react",
    "next.js",
    "typescript",
    "aws",
    "gcp",
    "docker",
    "kubernetes",
    "machine learning",
    "nlp",
    "recruiting",
    "ats",
    "graphql",
    "redis",
}

WORK_MODE_WORDS = {
    "remote": "Remote",
    "hybrid": "Hybrid",
    "onsite": "Onsite",
    "on-site": "Onsite",
}

SENIORITY_WORDS = [
    ("staff", "Staff"),
    ("principal", "Principal"),
    ("lead", "Lead"),
    ("senior", "Senior"),
    ("mid", "Mid-level"),
    ("junior", "Junior"),
]


class TalentScoutPipeline:
    """LangGraph workflow for the Deccan Catalyst talent scouting flow."""

    def __init__(self, repository: CandidateRepository, llm_client: GeminiClient):
        self.repository = repository
        self.llm_client = llm_client
        self.graph = self._build_graph()

    def _build_graph(self):
        """Wire the four-stage pipeline into a compiled LangGraph graph."""

        workflow = StateGraph(PipelineState)
        workflow.add_node("parse_job", self.parse_job)
        workflow.add_node("retrieve_candidates", self.retrieve_candidates)
        workflow.add_node("rank_candidates", self.rank_candidates)
        workflow.add_node("summarize_shortlist", self.summarize_shortlist)
        workflow.set_entry_point("parse_job")
        workflow.add_edge("parse_job", "retrieve_candidates")
        workflow.add_edge("retrieve_candidates", "rank_candidates")
        workflow.add_edge("rank_candidates", "summarize_shortlist")
        workflow.add_edge("summarize_shortlist", END)
        return workflow.compile()

    def run(self, job_input: JobInput) -> PipelineResult:
        """Execute the graph and shape the final API/UI response."""

        state = self.graph.invoke({"job_input": job_input})
        return PipelineResult(
            job=state["parsed_job"],
            ranked_candidates=state["ranked_candidates"],
            recruiter_summary=state["recruiter_summary"],
        )

    def parse_job(self, state: PipelineState) -> PipelineState:
        """Convert free-form JD text into structured ranking criteria."""

        job_input = state["job_input"]
        text = f"{job_input.title}\n{job_input.location}\n{job_input.description}"
        lowered = text.lower()

        seniority = next((label for key, label in SENIORITY_WORDS if key in lowered), "Mid-level")
        work_mode = next((label for key, label in WORK_MODE_WORDS.items() if key in lowered), "Remote")
        years_match = re.search(r"(\d+)\+?\s+years", lowered)
        min_years = int(years_match.group(1)) if years_match else 3

        discovered = [skill for skill in SKILL_VOCAB if skill in lowered]
        must_have = discovered[:6] or ["python", "fastapi", "llm"]
        nice_to_have = discovered[6:10] or ["langgraph", "postgres", "docker"]
        domain = [word for word in ["talent", "recruiting", "ats", "hiring", "staffing", "saas", "ai"] if word in lowered]

        parsed = ParsedJob(
            title=job_input.title,
            seniority=seniority,
            must_have_skills=sorted(dict.fromkeys(must_have)),
            nice_to_have_skills=sorted(dict.fromkeys(nice_to_have)),
            domain_keywords=domain or ["ai", "saas"],
            location=job_input.location,
            work_mode=work_mode,
            min_years_experience=min_years,
            summary=job_input.description.strip(),
        )
        return {"parsed_job": parsed}

    def retrieve_candidates(self, state: PipelineState) -> PipelineState:
        """Do a first-pass retrieval before the heavier scoring stage.

        This node intentionally uses a lightweight lexical signal so we do not
        waste time fully scoring weak candidates from the local pool.
        """

        job = state["parsed_job"]
        candidates = self.repository.list_candidates()
        scored = []
        for candidate in candidates:
            text_blob = " ".join(
                [
                    candidate.summary.lower(),
                    " ".join(skill.lower() for skill in candidate.skills),
                    " ".join(domain.lower() for domain in candidate.domain_experience),
                ]
            )
            signal = sum(1 for skill in job.must_have_skills if skill.lower() in text_blob)
            signal += sum(1 for word in job.domain_keywords if word.lower() in text_blob)
            scored.append((signal, candidate))
        top_candidates = [candidate for _, candidate in sorted(scored, key=lambda item: item[0], reverse=True)[: max(10, state["job_input"].top_k * 2)]]
        return {"candidates": top_candidates}

    def rank_candidates(self, state: PipelineState) -> PipelineState:
        """Compute match and interest scores for shortlisted candidates."""

        job = state["parsed_job"]
        evaluations: list[CandidateEvaluation] = []
        for candidate in state["candidates"]:
            match_score, explanation = score_candidate(job, candidate)
            interest = self.llm_client.simulate_interest(job, candidate)
            final_score = round((match_score * 0.7) + (interest.interest_score * 0.3), 1)
            status = "hot" if interest.interest_score >= 78 else "warm" if interest.interest_score >= 60 else "cold"
            evaluations.append(
                CandidateEvaluation(
                    candidate=candidate,
                    match_score=match_score,
                    interest_score=interest.interest_score,
                    final_score=final_score,
                    explanation=explanation,
                    outreach_summary=interest.outreach_summary,
                    transcript=interest.transcript,
                    status=status,
                )
            )
        ranked = sorted(evaluations, key=lambda item: item.final_score, reverse=True)[: state["job_input"].top_k]
        return {"ranked_candidates": ranked}

    def summarize_shortlist(self, state: PipelineState) -> PipelineState:
        """Write the summary line shown above the shortlist."""

        ranked = state["ranked_candidates"]
        if not ranked:
            summary = "No suitable candidates found in the current sourced talent pool."
        else:
            hottest = ranked[0]
            warm_count = sum(1 for item in ranked if item.status == "warm")
            summary = (
                f"{hottest.candidate.name} leads the shortlist with a final score of {hottest.final_score}. "
                f"The current batch has {sum(1 for item in ranked if item.status == 'hot')} high-intent candidates and {warm_count} warm leads."
            )
        return {"recruiter_summary": summary}
