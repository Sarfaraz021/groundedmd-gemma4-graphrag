"""
Agent skills — prompt-driven specializations injected into the RAG pipeline.

Each skill provides domain-specific interpretation guidance that constrains
and focuses the LLM response. Skills follow the progressive disclosure
pattern: load only what is needed per query context.

Reference: LangChain Skills pattern (https://python.langchain.com/docs/...)
"""

from dataclasses import dataclass


@dataclass
class Skill:
    name: str
    description: str
    guidance: str


# ---------------------------------------------------------------------------
# TBI Evidence Interpretation Skill
#
# Guides the LLM to interpret TBI clinical evidence from the 8 peer-reviewed
# publications in the corpus — covering AI diagnostics, blood biomarkers,
# outcome prediction, and neurorehabilitation.
# ---------------------------------------------------------------------------

TBI_EVIDENCE_SKILL = Skill(
    name="tbi_evidence_interpretation",
    description=(
        "Specialised guidance for interpreting TBI clinical evidence from "
        "peer-reviewed publications on AI diagnostics, biomarkers, outcome "
        "prediction, conformal prediction, and neurorehabilitation."
    ),
    guidance="""
## Skill: TBI Evidence Interpretation

Apply the following standards when formulating your answer:

### Classification Framework
- Use standard TBI severity grades: Mild, Moderate, Severe.
- Mild TBI (mTBI / concussion): GCS 13–15, LOC < 30 min, PTA < 24 h.
- Moderate TBI: GCS 9–12, LOC 30 min – 24 h, PTA 1–7 days.
- Severe TBI: GCS ≤ 8, LOC > 24 h, PTA > 7 days.
- Cite the specific source publication when referencing these thresholds.

### Biomarker Interpretation
- Report biomarker values with units and validated threshold ranges.
- Specify the time window of measurement (acute: 0–24 h; subacute: 1–7 days;
  chronic: > 7 days) — sensitivity varies significantly by window.
- Distinguish fluid-specific markers (blood/serum vs. CSF) and note
  accessibility implications for clinical use.
- Key markers: GFAP (astrocytic injury), UCH-L1 (neuronal injury),
  NfL (axonal degeneration), S100B (early screening), p-tau (chronic/CTE).

### Imaging Evidence
- Differentiate CT (structural, acute haemorrhage) from MRI
  (diffuse axonal injury, subacute findings).
- Reference the imaging criteria from the source publications when classifying
  imaging-based severity.
- Note that CT-negative does not exclude mTBI — correlate with clinical
  and biomarker data.

### Clinical Symptom Reporting
- Report symptom timelines relative to injury (Day 1–14 framework).
- Distinguish acute (< 72 h), subacute (3–14 days), and persistent (> 14 days)
  symptom profiles.
- Flag psychosocial and environmental modifiers that influence recovery.

### Evidence Strength Language
- Use precise language that reflects evidence quality:
  - "Established" — consensus-level evidence in the source documents.
  - "Emerging" — referenced but not yet consensus.
  - "Insufficient evidence" — explicitly noted in the source documents.
- Never infer evidence strength not stated in the retrieved context.
""",
)


# ---------------------------------------------------------------------------
# Skill registry and loader
# ---------------------------------------------------------------------------

SKILL_REGISTRY: dict[str, Skill] = {
    TBI_EVIDENCE_SKILL.name: TBI_EVIDENCE_SKILL,
}


def load_skill(name: str) -> Skill:
    """Return a skill by name. Raises KeyError if not found."""
    if name not in SKILL_REGISTRY:
        raise KeyError(f"Skill '{name}' not found. Available: {list(SKILL_REGISTRY)}")
    return SKILL_REGISTRY[name]


def get_skill_context(name: str) -> str:
    """Return the formatted guidance block for injection into a prompt."""
    skill = load_skill(name)
    return f"## Active Skill: {skill.description}\n{skill.guidance}"
