from typing import List, Optional, Dict, Any
from pydantic import BaseModel


# --- Nested models for addresses and entities --- #

class StateModel(BaseModel):
    id: Optional[str] = ""
    name: Optional[str] = ""


class AddressModel(BaseModel):
    zip: Optional[str] = ""
    city: Optional[str] = ""
    State: Optional[StateModel] = None
    address1: Optional[str] = ""
    address2: Optional[str] = ""


class SubmitterModel(BaseModel):
    anonymous: Optional[str] = "0"
    allowContact: Optional[int] = 1
    notify: Optional[str] = "1"
    lname: Optional[str] = ""
    phoneNumber: Optional[str] = ""
    SubmitterAddress: Optional[AddressModel] = None
    affiliation: Optional[str] = ""
    phoneExt: Optional[str] = ""
    email: Optional[str] = ""
    fname: Optional[str] = ""


class ViolatorModel(BaseModel):
    ViolatorAddress: Optional[AddressModel] = None
    violatorTypeIDs: Optional[int] = None  # 1=retailer,2=distributor,3=manufacturer
    websiteURL: Optional[str] = ""
    name: Optional[str] = ""


class ViolationModel(BaseModel):
    producttype: Optional[str] = ""
    description: str
    tobaccoBrand: Optional[str] = ""
    violationMaterialIDs: Optional[List[str]] = []
    State: Optional[StateModel] = None
    productsubtype: Optional[str] = ""
    violationTypeID: Optional[List[str]] = []
    Violator: Optional[List[ViolatorModel]] = []


class ReportModel(BaseModel):
    methodOfSubmission: Optional[int] = None
    guid: str
    State: Optional[StateModel] = None
    Submitter: SubmitterModel
    Violation: ViolationModel


class ReportsFile(BaseModel):
    reports: List[ReportModel]


# --- Pipeline output schemas --- #

class CaseOutput(BaseModel):
    case_id: str
    guid: str
    violator_role: str
    violator_name: str
    violator_address: AddressModel
    submitter_address: AddressModel
    confidence: float
    # 👇 this is what the frontend needs
    violation_description: str


class ReportResult(BaseModel):
    guid: str
    submitter_address: AddressModel
    cases: List[CaseOutput]
    flag_for_review: bool
    max_confidence: float


class PredictFileResponse(BaseModel):
    results: List[ReportResult]


class MetricsResponse(BaseModel):
    accuracy: float
    micro_f1: float
    macro_f1: float
    classification_report: Dict[str, Any]
