import json
import os
from typing import Dict, Any, List, Optional
import re

from openai import OpenAI

from .ml_model import ImprovedViolationMLModel, ROLE_NAME_TO_ID, ROLE_ID_TO_NAME


class BaseAgent:
    name: str = "base"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError



# ParserAgent – uses OpenAI for violator parsing + descriptions
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
            print(" OPENAI_API_KEY not set. ParserAgent will use JSON-only fallback.")

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

        # Build hints from existing JSON violators (name + role)
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

Your goal is to extract violators from the following narrative and describe what each one did wrong.

You will always return a valid JSON array, where each element contains:
- "role": one of "retailer", "distributor", "manufacturer"
- "name": the name of the violator (never empty)
- "address": an object with keys "address1", "address2", "city", "state", "zip"
- "violation_description": 1–2 sentences summarizing the specific violation for this violator
- "confidence": a number between 0.0 and 1.0

Rules:
1. If the narrative doesn’t explicitly state an address, leave address fields empty strings.
2. Always include a concise "violation_description" based on the narrative.
3. Use the violator hints below if helpful to identify names and roles.
4. Output only JSON, with no explanation text.

Example Output:
[
  {{
    "role": "retailer",
    "name": "Love's Travel Stops",
    "address": {{
      "address1": "1565 Oak Ave",
      "address2": "",
      "city": "Des Moines",
      "state": "Iowa",
      "zip": "50310"
    }},
    "violation_description": "Sold tobacco to minors without verifying age.",
    "confidence": 0.92
  }},
  {{
    "role": "distributor",
    "name": "Eby-Brown Company",
    "address": {{
      "address1": "6009 Oak St",
      "address2": "",
      "city": "Des Moines",
      "state": "Iowa",
      "zip": "50309"
    }},
    "violation_description": "Supplied cigarettes to a retailer that failed age-compliance checks.",
    "confidence": 0.87
  }}
]

Violator hints (may be empty):
{hints_text}

Narrative:
\"\"\"{narrative}\"\"\"
"""

        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You output only valid JSON. No explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        content = resp.choices[0].message.content
        try:
            # be tolerant of fences or extra text
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                content_json = content[start : end + 1]
            else:
                content_json = content
            parsed = json.loads(content_json)
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                return []
            return parsed
        except Exception as e:
            print("⚠️ LLM parse failed, returning empty list:", e)
            return []
    import re  

    def _extract_violation_snippet(
        self, narrative: str, violator_name: str, role: str
    ) -> str:
        """
        Very simple heuristic: return only the sentences from the narrative
        that mention this violator's name or role-related keywords.
        If nothing matches, fall back to the full narrative.
        """
        text = (narrative or "").strip()
        if not text:
            return ""

        # split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        role_keywords = {
            "retailer": ["retailer", "store", "shop", "cashier", "sold", "sale"],
            "distributor": ["distributor", "wholesaler", "supplied", "ship", "distribution"],
            "manufacturer": ["manufacturer", "produces", "makes", "company", "brand"],
        }
        keywords = [k.lower() for k in role_keywords.get(role, [])]

        name_lower = (violator_name or "").lower()

        selected: List[str] = []
        for s in sentences:
            s_lower = s.lower()
            if name_lower and name_lower in s_lower:
                selected.append(s)
                continue
            if any(k in s_lower for k in keywords):
                selected.append(s)

        if selected:
            return " ".join(selected)
        return text  


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
                print("⚠️ Error calling OpenAI in ParserAgent:", e)
                parsed = []

        # 2) If LLM returned nothing, fall back to JSON violators
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

                # If address1 empty, try to match a JSON violator to fill address
                if not addr.get("address1"):
                    match = None

                    # a) match by exact name
                    for v in raw_violators:
                        if name and v.get("name", "").lower() == name.lower():
                            match = v
                            break

                    # b) match by role only
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

                # Ensure all keys exist
                addr = {
                    "address1": addr.get("address1", "") or "",
                    "address2": addr.get("address2", "") or "",
                    "city": addr.get("city", "") or "",
                    "state": addr.get("state", "") or "",
                    "zip": addr.get("zip", "") or "",
                }
                rec["address"] = addr

                # Ensure we always have a violation_description
                if not rec.get("violation_description"):
                    rec["violation_description"] = narrative

        state["parsed_violators"] = parsed
        return state



# ClassifierAgent – uses ML model to predict roles present

class ClassifierAgent(BaseAgent):
    """
    Uses the trained ML model to classify which roles are present.
    """
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



# ValidatorAgent – per-role thresholds + overall confidence
class ValidatorAgent(BaseAgent):
    """
    Validates confidences and prepares a single confidence score.
    Uses per-role thresholds if available; otherwise falls back to
    a single global threshold (min_threshold).
    """
    name = "validator"

    def __init__(self, min_threshold: float = 0.8, role_thresholds: Optional[Dict[str, float]] = None):
        self.min_threshold = min_threshold
        self.role_thresholds = role_thresholds or {}

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        probs_by_role = state.get("probs_by_role", {})
        predicted_roles = state.get("predicted_roles", [])

        if not predicted_roles:
            state["overall_confidence"] = 0.0
            state["needs_review"] = True
            return state

        # per-role thresholds from either ctor or state (model)
        role_thresholds = self.role_thresholds or state.get("role_thresholds", {})

        probs = [probs_by_role.get(r, 0.0) for r in predicted_roles]
        overall_conf = float(min(probs)) if probs else 0.0
        state["overall_confidence"] = overall_conf

        if role_thresholds:
            needs_review = any(
                probs_by_role.get(r, 0.0) < role_thresholds.get(r, self.min_threshold)
                for r in predicted_roles
            )
        else:
            # fallback: single global threshold
            needs_review = overall_conf < self.min_threshold

        state["needs_review"] = needs_review
        return state


# CaseCreatorAgent – one case per *real* violator
class CaseCreatorAgent(BaseAgent):
    """
    Creates one case per predicted role, but **only** when there is a
    corresponding parsed violator. We no longer invent "Unknown Distributor"
    (or other roles) when no such violator exists in the report.
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
        cases: List[Dict[str, Any]] = []

        for role in predicted_roles:
            match = next((v for v in parsed_violators if v.get("role") == role), None)

            # If there is no violator for this role, skip creating a case
            if not match:
                continue

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
            # Use len(cases)+1 so indices remain sequential even if we skip roles
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
                    "violation_description": violation_desc,
                }
            )

        state["cases"] = cases
        return state



# ReviewAgent – turns needs_review into flag_for_review
class ReviewAgent(BaseAgent):
    """
    Flags for human review based on confidence.
    """
    name = "review"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        needs_review = state.get("needs_review", False)
        state["flag_for_review"] = bool(needs_review)
        return state



# RecordAgent – tracks total cases created
class RecordAgent(BaseAgent):
    """
    Updates counters, logging, metrics etc.
    For now, just increments a simple count in memory.
    """
    name = "record"

    def __init__(self):
        self.total_cases_created = 0

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        cases = state.get("cases", []) or []
        self.total_cases_created += len(cases)
        state["total_cases_created"] = self.total_cases_created
        return state
