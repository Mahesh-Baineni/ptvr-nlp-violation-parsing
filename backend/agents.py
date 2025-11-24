import json
import os
import re
from typing import Dict, Any, List, Optional

from openai import OpenAI

from .ml_model import ImprovedViolationMLModel, ROLE_NAME_TO_ID, ROLE_ID_TO_NAME


class BaseAgent:
    name: str = "base"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError



# ParserAgent – LLM 

class ParserAgent(BaseAgent):
    """
    Uses OpenAI to parse narrative into violators with roles, addresses,
    and a short violation_description per violator.
    Falls back to existing Violator entries if LLM is not available.
    """
    name = "parser"

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client: Optional[OpenAI] = None
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            print("OPENAI_API_KEY not set. ParserAgent will use JSON-only fallback.")

    def _llm_parse(
        self,
        narrative: str,
        raw_violators: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Ask OpenAI to extract:
          - role (retailer / distributor / manufacturer)
          - name
          - address (address1, address2, city, state, zip)
          - violation_description (1–2 sentence summary)
          - confidence (0–1)
        """
        violator_hints: List[Dict[str, str]] = []
        for v in raw_violators:
            name = v.get("name", "")
            vt = v.get("violatorTypeIDs")
            role = ROLE_ID_TO_NAME.get(vt)
            if name and role:
                violator_hints.append({"name": name, "role": role})
        hints_text = json.dumps(violator_hints, ensure_ascii=False)

        prompt = f"""
You are a structured information extraction agent for tobacco violation reports.

Return ONLY a valid JSON array. Each element:
- "role": one of "retailer", "distributor", "manufacturer"
- "name": non-empty string
- "address": object with "address1","address2","city","state","zip" (empty strings if unknown)
- "violation_description": 1–2 sentences specific to this violator (no generic boilerplate)
- "confidence": float 0.0–1.0

Rules:
1) If address not in narrative, leave address fields empty strings.
2) The "violation_description" must reference actions/evidence for THIS violator only.
3) Use the hints when helpful.
4) Output ONLY JSON. No extra text.

Hints (may be empty):
{hints_text}

Narrative:
\"\"\"{narrative}\"\"\"
"""

        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Output only valid JSON. No commentary."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        content = resp.choices[0].message.content or ""
        try:
            start = content.find("[")
            end = content.rfind("]")
            content_json = content[start: end + 1] if (start != -1 and end != -1) else content
            parsed = json.loads(content_json)
            if isinstance(parsed, dict):
                parsed = [parsed]
            return parsed if isinstance(parsed, list) else []
        except Exception as e:
            print("LLM parse failed, returning empty list:", e)
            return []

    def _extract_violation_snippet(
        self, narrative: str, violator_name: str, role: str
    ) -> str:
        """
        Heuristic snippet:
        - keep only sentences that mention the violator's name OR role-specific verbs
        - cap to the first 2 matched sentences to avoid dumping the whole narrative
        """
        text = (narrative or "").strip()
        if not text:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", text)

        role_keywords = {
            "retailer": ["retailer", "store", "shop", "cashier", "sold", "sale", "id check", "underage"],
            "distributor": ["distributor", "wholesaler", "supplied", "supply", "shipment", "distribution"],
            "manufacturer": ["manufacturer", "manufactured", "produced", "labeled", "packaging", "warning"],
        }
        keywords = [k.lower() for k in role_keywords.get(role, [])]
        name_lower = (violator_name or "").lower()

        selected: List[str] = []
        for s in sentences:
            s_lower = s.lower()
            if name_lower and name_lower in s_lower:
                selected.append(s)
            elif any(k in s_lower for k in keywords):
                selected.append(s)
            if len(selected) >= 2:
                break

        return " ".join(selected) if selected else sentences[0] if sentences else text

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        report = state["report"]
        narrative = state["narrative"]
        raw_violators = (report.get("Violation") or {}).get("Violator", []) or []

        parsed: List[Dict[str, Any]] = []

        # 1) Try OpenAI-based parsing (if configured)
        if self.client:
            try:
                parsed = self._llm_parse(narrative, raw_violators)
            except Exception as e:
                print(" Error calling OpenAI in ParserAgent:", e)
                parsed = []

        # 2) If LLM returned nothing, fall back to JSON violators with heuristic snippets
        if not parsed:
            for v in raw_violators:
                vt = v.get("violatorTypeIDs")
                role = ROLE_ID_TO_NAME.get(vt)
                if not role:
                    continue

                addr = v.get("ViolatorAddress") or {}
                name = v.get("name", "") or f"Unknown {role.title()}"

                violation_snippet = self._extract_violation_snippet(
                    narrative=narrative,
                    violator_name=name,
                    role=role,
                )

                parsed.append(
                    {
                        "role": role,
                        "name": name,
                        "address": {
                            "address1": addr.get("address1", "") or "",
                            "address2": addr.get("address2", "") or "",
                            "city": addr.get("city", "") or "",
                            "state": (addr.get("State") or {}).get("name", "") or "",
                            "zip": addr.get("zip", "") or "",
                        },
                        "violation_description": violation_snippet,
                        "confidence": 1.0,
                    }
                )
        else:
            # 3) LLM responded: patch missing addresses from raw JSON if possible
            for rec in parsed:
                role = rec.get("role")
                name = rec.get("name", "")
                addr = rec.get("address") or {}
                if not isinstance(addr, dict):
                    addr = {}

                if not addr.get("address1"):
                    match = None
                    # (a) exact name
                    for v in raw_violators:
                        if name and v.get("name", "").lower() == name.lower():
                            match = v
                            break
                    # (b) by role
                    if not match:
                        for v in raw_violators:
                            vt = v.get("violatorTypeIDs")
                            if ROLE_ID_TO_NAME.get(vt) == role:
                                match = v
                                break

                    if match:
                        m_addr = match.get("ViolatorAddress") or {}
                        addr = {
                            "address1": m_addr.get("address1", "") or "",
                            "address2": m_addr.get("address2", "") or "",
                            "city": m_addr.get("city", "") or "",
                            "state": (m_addr.get("State") or {}).get("name", "") or "",
                            "zip": m_addr.get("zip", "") or "",
                        }

                addr = {
                    "address1": addr.get("address1", "") or "",
                    "address2": addr.get("address2", "") or "",
                    "city": addr.get("city", "") or "",
                    "state": addr.get("state", "") or "",
                    "zip": addr.get("zip", "") or "",
                }
                rec["address"] = addr

                if not rec.get("violation_description"):
                    rec["violation_description"] = self._extract_violation_snippet(
                        narrative=narrative, violator_name=name, role=role or ""
                    )

        state["parsed_violators"] = parsed
        return state


# ClassifierAgent – predict roles + probs

class ClassifierAgent(BaseAgent):
    name = "classifier"

    def __init__(self, model: ImprovedViolationMLModel):
        self.model = model

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        report = state["report"]
        roles, probs_by_role, max_conf = self.model.predict_roles_for_report(report)
        state["predicted_roles"] = roles
        state["probs_by_role"] = probs_by_role
        state["max_confidence"] = max_conf
        state["role_thresholds"] = getattr(self.model, "role_thresholds", {})
        return state



# ValidatorAgent – report-level needs_review (by role thr)

class ValidatorAgent(BaseAgent):
    """
    Computes an overall confidence summary and whether the report needs review,
    based on learned per-role thresholds (falls back to a global min_threshold).
    """
    name = "validator"

    def __init__(self, min_threshold: float = 0.8, role_thresholds: Optional[Dict[str, float]] = None):
        self.min_threshold = min_threshold
        self.role_thresholds = role_thresholds or {}

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        probs_by_role = state.get("probs_by_role", {})
        predicted_roles = state.get("predicted_roles", [])
        role_thresholds = self.role_thresholds or state.get("role_thresholds", {})

        if not predicted_roles:
            state["overall_confidence"] = 0.0
            state["needs_review"] = True
            return state

        probs = [float(probs_by_role.get(r, 0.0)) for r in predicted_roles]
        state["overall_confidence"] = float(min(probs)) if probs else 0.0

        # Needs review if any predicted role falls below its threshold
        if role_thresholds:
            needs_review = any(
                float(probs_by_role.get(r, 0.0)) < float(role_thresholds.get(r, self.min_threshold))
                for r in predicted_roles
            )
        else:
            needs_review = state["overall_confidence"] < self.min_threshold

        state["needs_review"] = bool(needs_review)
        return state


# CaseCreatorAgent – create cases per *real* violator

class CaseCreatorAgent(BaseAgent):
    """
    Creates one case per predicted role, but only when there is a matching
    parsed violator in the report. Adds per-case threshold and flag.
    """
    name = "case_creator"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        guid = state["violation_guid"]
        narrative = state["narrative"]
        report = state["report"]

        submitter = report.get("Submitter", {})
        sub_addr = (submitter.get("SubmitterAddress") or {}) or {}

        submitter_address = {
            "zip": sub_addr.get("zip", ""),
            "city": sub_addr.get("city", ""),
            "State": sub_addr.get("State", {}),
            "address1": sub_addr.get("address1", ""),
            "address2": sub_addr.get("address2", ""),
        }

        parsed_violators = state.get("parsed_violators", [])
        predicted_roles = state.get("predicted_roles", [])
        probs_by_role = state.get("probs_by_role", {})
        role_thresholds = state.get("role_thresholds", {}) or {}

        cases: List[Dict[str, Any]] = []

        for role in predicted_roles:
            match = next((v for v in parsed_violators if v.get("role") == role), None)
            if not match:
                continue  # don't invent violators

            addr = match.get("address", {}) or {}
            violator_address = {
                "zip": addr.get("zip", "") or "",
                "city": addr.get("city", "") or "",
                "State": {"id": "", "name": addr.get("state", "") or ""},
                "address1": addr.get("address1", "") or "",
                "address2": addr.get("address2", "") or "",
            }

            name = match.get("name", "") or f"Unknown {role.title()}"
            violation_desc = match.get("violation_description") or narrative

            conf = float(probs_by_role.get(role, 0.0))
            thr = float(role_thresholds.get(role, 0.8))
            case_needs_review = conf < thr

            case_id = f"{guid}-{role}-{len(cases) + 1}"

            cases.append(
                {
                    "case_id": case_id,
                    "guid": guid,
                    "violator_role": role,
                    "violator_name": name,
                    "violator_address": violator_address,
                    "submitter_address": submitter_address,
                    "confidence": conf,
                    "threshold": thr,
                    "case_needs_review": case_needs_review,
                    "violation_description": violation_desc,
                }
            )

        # Report-level confidence summary from cases
        confs = [c["confidence"] for c in cases] if cases else []
        state["report_min_conf"] = float(min(confs)) if confs else None
        state["report_avg_conf"] = float(sum(confs) / len(confs)) if confs else None
        state["report_max_conf"] = float(max(confs)) if confs else None

        state["cases"] = cases
        return state



# ReviewAgent – combine case-level flags

class ReviewAgent(BaseAgent):
    name = "review"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Report should be flagged for review *only* if at least one case has
        confidence < threshold (case_needs_review = True).
        We ignore the validator's report-level needs_review flag.
        """
        any_case_review = any(
            c.get("case_needs_review", False)
            for c in (state.get("cases") or [])
        )
        state["flag_for_review"] = bool(any_case_review)
        return state



# RecordAgent – simple counter

class RecordAgent(BaseAgent):
    name = "record"

    def __init__(self):
        self.total_cases_created = 0

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        cases = state.get("cases", []) or []
        self.total_cases_created += len(cases)
        state["total_cases_created"] = self.total_cases_created
        return state
