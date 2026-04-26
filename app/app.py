from __future__ import annotations

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR, settings
from app.models import JobInput
from app.services.llm import GeminiClient
from app.services.pipeline import TalentScoutPipeline
from app.services.repository import CandidateRepository


app = FastAPI(title=settings.app_name)
app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

repository = CandidateRepository()
repository.initialize()
pipeline = TalentScoutPipeline(repository=repository, llm_client=GeminiClient())

SAMPLE_JD = """Deccan is hiring a Senior AI Engineer to build talent scouting workflows for recruiters.
You will own Python and FastAPI services, design LLM and RAG pipelines, work with LangChain or LangGraph,
ship production APIs, and collaborate closely with product and design. Candidates should have 4+ years of
experience, strong Postgres and cloud fundamentals, and comfort with remote collaboration across India."""


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main Deccan Catalyst demo page."""

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "sample_jd": SAMPLE_JD,
            "result": None,
        },
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    title: str = Form(...),
    company: str = Form(...),
    location: str = Form(...),
    employment_type: str = Form(...),
    description: str = Form(...),
    top_k: int = Form(5),
):
    """Handle the recruiter form submission and render shortlist results."""

    job_input = JobInput(
        title=title,
        company=company,
        location=location,
        employment_type=employment_type,
        description=description,
        top_k=top_k,
    )
    result = pipeline.run(job_input)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "sample_jd": SAMPLE_JD,
            "result": result.model_dump(),
            "form": job_input.model_dump(),
        },
    )


@app.post("/api/analyze")
async def analyze_api(job_input: JobInput):
    """Expose the same pipeline through a JSON API."""

    result = pipeline.run(job_input)
    return JSONResponse(result.model_dump())


@app.get("/health")
async def health():
    """Basic health endpoint for deployment checks."""

    return {"status": "ok"}
