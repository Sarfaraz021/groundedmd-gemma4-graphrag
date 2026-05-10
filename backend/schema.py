"""
TBI-specific knowledge graph schema.
Node types, relationship types, and valid patterns aligned with
NINDS TBI Common Data Elements (CDE) terminology.
"""

# ---------------------------------------------------------------------------
# Node Types
# ---------------------------------------------------------------------------

BASIC_NODE_LABELS = [
    "Entity",
    "Group",
    "Organization",
    "Person",
]

ACADEMIC_NODE_LABELS = [
    "ArticleOrPaper",
    "PublicationOrJournal",
]

MEDICAL_NODE_LABELS = [
    "Anatomy",
    "BiologicalProcess",
    "Biomarker",
    "ClinicalOutcome",
    "ClinicalScale",
    "Condition",
    "Disease",
    "Drug",
    "EffectOrPhenotype",
    "GeneOrProtein",
    "ImagingFinding",
    "ImagingModality",
    "Molecule",
    "Pathway",
    "Symptom",
    "TBIClassification",
    "Treatment",
    "TimeWindow",
]

NODE_LABELS = BASIC_NODE_LABELS + ACADEMIC_NODE_LABELS + MEDICAL_NODE_LABELS


# ---------------------------------------------------------------------------
# Relationship Types
# ---------------------------------------------------------------------------

RELATIONSHIP_TYPES = [
    "ACTIVATES",
    "AFFECTS",
    "ASSESSES",
    "ASSOCIATED_WITH",
    "AUTHORED",
    "BIOMARKER_FOR",
    "CLASSIFIED_AS",
    "CORRELATES_WITH",
    "DIAGNOSES",
    "FOUND_IN",
    "IMPROVES",
    "INDICATES",
    "MEASURED_BY",
    "MEASURED_DURING",
    "MODIFIES",
    "PREDICTS",
    "REFERENCED_IN",
    "RELATED_TO",
    "TREATS",
    "USED_FOR",
    "VALIDATES",
]


# ---------------------------------------------------------------------------
# Patterns — (source_label, relationship_label, target_label)
# Defines valid connections per TBI CDE framework
# ---------------------------------------------------------------------------

PATTERNS = [
    # Biomarker relationships
    ("Biomarker",         "BIOMARKER_FOR",    "TBIClassification"),
    ("Biomarker",         "INDICATES",        "Condition"),
    ("Biomarker",         "INDICATES",        "TBIClassification"),
    ("Biomarker",         "CORRELATES_WITH",  "ImagingFinding"),
    ("Biomarker",         "CORRELATES_WITH",  "ClinicalOutcome"),
    ("Biomarker",         "MEASURED_DURING",  "TimeWindow"),
    ("Biomarker",         "MEASURED_BY",      "ClinicalScale"),

    # Imaging relationships
    ("ImagingFinding",    "CORRELATES_WITH",  "TBIClassification"),
    ("ImagingFinding",    "CORRELATES_WITH",  "Symptom"),
    ("ImagingFinding",    "FOUND_IN",         "Anatomy"),
    ("ImagingModality",   "ASSESSES",         "ImagingFinding"),
    ("ImagingModality",   "USED_FOR",         "TBIClassification"),

    # Clinical symptom relationships
    ("Symptom",           "ASSOCIATED_WITH",  "TBIClassification"),
    ("Symptom",           "ASSOCIATED_WITH",  "Condition"),
    ("Symptom",           "MEASURED_BY",      "ClinicalScale"),
    ("Symptom",           "AFFECTS",          "ClinicalOutcome"),

    # Classification relationships
    ("TBIClassification", "CLASSIFIED_AS",    "Condition"),
    ("TBIClassification", "ASSOCIATED_WITH",  "ClinicalOutcome"),

    # Treatment relationships
    ("Drug",              "TREATS",           "Condition"),
    ("Drug",              "AFFECTS",          "Biomarker"),
    ("Treatment",         "TREATS",           "Condition"),
    ("Treatment",         "IMPROVES",         "ClinicalOutcome"),

    # Biological relationships
    ("GeneOrProtein",     "ASSOCIATED_WITH",  "Condition"),
    ("GeneOrProtein",     "ASSOCIATED_WITH",  "TBIClassification"),
    ("Molecule",          "ACTIVATES",        "BiologicalProcess"),
    ("BiologicalProcess", "AFFECTS",          "ClinicalOutcome"),

    # Academic relationships
    ("ArticleOrPaper",    "AUTHORED",         "Person"),
    ("ArticleOrPaper",    "REFERENCED_IN",    "ArticleOrPaper"),
    ("Person",            "AUTHORED",         "ArticleOrPaper"),
]
