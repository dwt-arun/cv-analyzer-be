from pydantic import BaseModel, Field, field_serializer
from typing import Literal, List, Optional, Dict


class EmploymentHistory(BaseModel):
    company: str
    role: Optional[str] = ""
    start_date: Optional[str] = ""   
    end_date: Optional[str] = ""     
    location: Optional[str] = ""
    summary: Optional[str] = ""   

class CandidateProfile(BaseModel):
    name: str
    gender: Optional[str] = ""
    email: Optional[str] = ""
    linkedin: Optional[str] = ""
    phone: Optional[str] = ""
    location: Optional[str] = ""

    languages: Optional[List[str]] = Field(default_factory=list)
    language_proficiency: Optional[Dict[str, str]] = Field(default_factory=dict)

    current_role: Optional[str] = ""
    current_company: Optional[str] = ""

    summary: Optional[str] = ""

    education: Optional[List[str]] = Field(default_factory=list)
    certifications: Optional[List[str]] = Field(default_factory=list)

    employment_history: Optional[List[EmploymentHistory]] = Field(default_factory=list)

class SkillMatch(BaseModel):
    skill_name: str
    evidence_from_cv: str
    evidence_from_jd: str


class DimensionScore(BaseModel):
    pillar_name: str
    dimension_name: str
    weight_percent: float
    behavioral_anchor_score_1_5: int = Field(ge=1, le=5)
    score_10_scale: float
    weighted_score_out_of_100: float
    evidence_from_cv: str
    evidence_from_jd: str
    comments: Optional[str]

    @field_serializer("weight_percent", "score_10_scale", "weighted_score_out_of_100")
    def serialize_two_decimals(self, v):
        return float(f"{v:.2f}")


class PillarScore(BaseModel):
    pillar_name: str
    pillar_weight_percent: float
    total_pillar_score_out_of_100: float
    dimensions: List[DimensionScore]



class FinalStructuredOutput(BaseModel):
    # candidate_profile: CandidateProfile
    job_fit_category: Literal["High Fit", "Medium Fit", "Low Fit", "No Fit"]
    pillars: List[PillarScore]
    overall_match_score_100: float
    skill_matches: List[SkillMatch] = Field(default_factory=list)
    justification_summary: str

    skill_match_score_breakdown: Optional[str]
    experience_match_breakdown: Optional[str] 
    additional_notes: Optional[str] = None


def recompute_total_score(data: FinalStructuredOutput) -> float:
    total = 0.0
    for pillar in data.pillars:
        pillar_sum = 0.0
        for dim in pillar.dimensions:
            score_10 = dim.behavioral_anchor_score_1_5 * 2
            weighted = score_10 * (dim.weight_percent / 100) * 10
            dim.score_10_scale = score_10
            dim.weighted_score_out_of_100 = weighted
            pillar_sum += weighted

        pillar.total_pillar_score_out_of_100 = pillar_sum
        total += pillar_sum

    data.overall_match_score_100 = round(total, 2)
    return data.overall_match_score_100

