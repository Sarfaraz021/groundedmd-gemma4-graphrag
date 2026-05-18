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
## TBI Evidence Interpretation

- Use standard severity grades: Mild (GCS 13–15), Moderate (GCS 9–12), Severe (GCS ≤ 8).
- Report biomarker values with units, time window (acute/subacute/chronic), and fluid type.
- Key markers: GFAP, UCH-L1, NfL, S100B, p-tau — cite thresholds from the source chunks.
- Differentiate CT (structural/acute) from MRI (axonal injury/subacute).
- Use evidence-strength language: "Established", "Emerging", or "Insufficient evidence".
- Cite each claim with [n] referring to the chunk number.
- If the retrieved context does not contain the answer, say so explicitly — do not guess.
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
