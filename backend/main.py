import json
import os
from typing import List, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from .ml_model import ImprovedViolationMLModel
from .pipeline import ViolationPipeline
from .schemas import (
    PredictFileResponse,
    ReportResult,
    CaseOutput,
    MetricsResponse,
    AddressModel,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_PATH = os.path.join(BASE_DIR, "ptvr_reports.json")

app = FastAPI(title="PTVR Violation Agentic Pipeline (IMPROVED)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


ml_model = ImprovedViolationMLModel(data_path=DATA_PATH, use_ensemble=False)
ml_model.train()
pipeline = ViolationPipeline(ml_model)


@app.get("/health")
def health():
    return {"status": "ok", "model": "improved", "version": "2.0.0"}


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    m = ml_model.metrics
    return MetricsResponse(
        accuracy=m.get("accuracy", 0.0),
        micro_f1=m.get("micro_f1", 0.0),
        macro_f1=m.get("macro_f1", 0.0),
        classification_report=m.get("classification_report", {}),
    )


@app.post("/predict_file", response_model=PredictFileResponse)
async def predict_file(file: UploadFile = File(...)):
    """
    Accepts a JSON file with:
      { "reports": [ { ... }, ... ] }
    or a single report object.
    Returns multi-case outputs per report, including submitter + violator addresses
    and a flag for human review if confidence < threshold.
    """
    raw = await file.read()
    data = json.loads(raw.decode("utf-8"))

    # Normalise to list of reports
    if isinstance(data, dict) and "reports" in data:
        reports = data["reports"]
    elif isinstance(data, dict):
        reports = [data]
    elif isinstance(data, list):
        reports = data
    else:
        reports = []

    results: List[Any] = []

    for rep in reports:
        state = pipeline.process_report(rep)

        submitter = rep.get("Submitter", {})
        sub_addr = (submitter.get("SubmitterAddress") or {}) or {}
        submitter_address = AddressModel(
            zip=sub_addr.get("zip", ""),
            city=sub_addr.get("city", ""),
            State=sub_addr.get("State", {}),
            address1=sub_addr.get("address1", ""),
            address2=sub_addr.get("address2", ""),
        )

        cases: List[CaseOutput] = []
        for c in state.get("cases", []):
            viol_addr = c.get("violator_address", {})
            cases.append(
                CaseOutput(
                    case_id=c.get("case_id"),
                    guid=c.get("guid"),
                    violator_role=c.get("violator_role"),
                    violator_name=c.get("violator_name"),
                    violator_address=AddressModel(
                        zip=viol_addr.get("zip", ""),
                        city=viol_addr.get("city", ""),
                        State=viol_addr.get("State", {}),
                        address1=viol_addr.get("address1", ""),
                        address2=viol_addr.get("address2", ""),
                    ),
                    submitter_address=submitter_address,
                    confidence=float(c.get("confidence", 0.0)),
                    violation_description=c.get("violation_description", ""),
                )
            )

        results.append(
            ReportResult(
                guid=rep.get("guid", ""),
                submitter_address=submitter_address,
                cases=cases,
                flag_for_review=bool(state.get("flag_for_review", False)),
                max_confidence=float(state.get("overall_confidence", 0.0)),
            )
        )

    return PredictFileResponse(results=results)
