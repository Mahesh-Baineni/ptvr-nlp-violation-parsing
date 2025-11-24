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

app = FastAPI(title="PTVR Violation Agentic Pipeline (IMPROVED)", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Train model and build pipeline
ml_model = ImprovedViolationMLModel(data_path=DATA_PATH, use_ensemble=False)
ml_model.train()
pipeline = ViolationPipeline(ml_model)


# Health & Metrics Endpoints
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "improved",
        "version": "2.1.0",
        "thresholds": ml_model.role_thresholds,
    }


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    m = ml_model.metrics or {}
    return MetricsResponse(
        accuracy=m.get("accuracy", 0.0),
        micro_f1=m.get("micro_f1", 0.0),
        macro_f1=m.get("macro_f1", 0.0),
        classification_report=m.get("classification_report", {}),
    )


@app.get("/thresholds")
def get_thresholds():
    """Convenience endpoint to see learned per-role thresholds."""
    return {"role_thresholds": ml_model.role_thresholds}



# Inference Route
@app.post("/predict_file", response_model=PredictFileResponse)
async def predict_file(file: UploadFile = File(...)):
    """
    Accepts:
      - A JSON object: { "reports": [ { ... }, ... ] }
      - A single report object { ... }
      - Or a raw list [ { ... }, ... ]
    Returns multi-case outputs per report with per-case thresholding and flags.
    """
    raw = await file.read()
    data = json.loads(raw.decode("utf-8"))

    # Normalize to list of reports
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

        # Submitter address (passthrough)
        submitter = rep.get("Submitter", {}) or {}
        sub_addr = (submitter.get("SubmitterAddress") or {}) or {}
        submitter_address = AddressModel(
            zip=sub_addr.get("zip", ""),
            city=sub_addr.get("city", ""),
            State=sub_addr.get("State", {}),
            address1=sub_addr.get("address1", ""),
            address2=sub_addr.get("address2", ""),
        )

        # Build case outputs with per-case fields
        cases: List[CaseOutput] = []
        for c in state.get("cases", []) or []:
            viol_addr = c.get("violator_address", {}) or {}
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
                    threshold=float(c.get("threshold", 0.0)),
                    case_needs_review=bool(c.get("case_needs_review", False)),
                    violation_description=c.get("violation_description", "") or "",
                )
            )

        # Report-level summaries
        min_conf = state.get("report_min_conf", None)
        avg_conf = state.get("report_avg_conf", None)
        overall_conf = float(state.get("overall_confidence", 0.0))  # kept for back-compat

        results.append(
            ReportResult(
                guid=rep.get("guid", ""),
                submitter_address=submitter_address,
                cases=cases,
                flag_for_review=bool(state.get("flag_for_review", False)),
                max_confidence=overall_conf,
                min_confidence=None if min_conf is None else float(min_conf),
                avg_confidence=None if avg_conf is None else float(avg_conf),
            )
        )

    return PredictFileResponse(results=results)
