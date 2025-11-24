from typing import Dict, Any

from .ml_model import ImprovedViolationMLModel
from .agents import (
    ParserAgent,
    ClassifierAgent,
    ValidatorAgent,
    CaseCreatorAgent,
    ReviewAgent,
    RecordAgent,
)


class ViolationPipeline:
    """
    Agent Workflow:
      1. Ingest: Receive report + narrative.
      2. Parse: ParserAgent extracts violator mentions (LLM / fallback).
      3. Classify: ClassifierAgent predicts violator roles (NLP model).
      4. Validate: ValidatorAgent derives confidence.
      5. Create: CaseCreatorAgent generates multi-cases.
      6. Review: ReviewAgent flags for human approval if confidence < threshold.
      7. Record: RecordAgent updates total cases metric.
    """

    def __init__(self, model: ImprovedViolationMLModel):
        self.parser = ParserAgent()
        self.classifier = ClassifierAgent(model)
        self.validator = ValidatorAgent(min_threshold=0.8)
        self.case_creator = CaseCreatorAgent()
        self.review = ReviewAgent()
        self.record = RecordAgent()

    def process_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        violation = report.get("Violation") or {}
        narrative = violation.get("description", "")
        guid = report.get("guid", "")

        state: Dict[str, Any] = {
            "report": report,
            "narrative": narrative,
            "violation_guid": guid,
        }

        # Agent chain
        for agent in [
            self.parser,
            self.classifier,
            self.validator,
            self.case_creator,
            self.review,
            self.record,
        ]:
            state = agent.run(state)

        return state
