"""
Canonical Section Intent Map
Global configuration for mapping semantically equivalent section terms
to a single canonical intent group.

Rules:
- Groups are global, NOT drug-specific.
- All terms are lowercase normalized.
- No dynamic rule growth at runtime.
- Maximum ~25-30 canonical groups.
- Each group maps to a list of synonym phrases that users or SPL titles may use.
"""

# Canonical group key -> list of synonym phrases (lowercase)
SECTION_INTENT_MAP = {
    "CLINICAL_PHARMACOLOGY": [
        "clinical pharmacology",
        "mechanism of action",
        "mechanism",
        "pharmacodynamics",
        "pharmacologic action",
        "pharmacological action",
        "mode of action",
        "action",
        "pharmacology",
    ],
    "OVERDOSAGE": [
        "overdose",
        "overdosage",
        "toxicity",
        "poisoning",
        "toxic effects",
    ],
    "ADVERSE_REACTIONS": [
        "adverse reactions",
        "adverse effects",
        "side effects",
        "undesirable effects",
        "adverse events",
    ],
    "CONTRAINDICATIONS": [
        "contraindications",
        "contraindicated",
        "should not be used",
    ],
    "WARNINGS_AND_PRECAUTIONS": [
        "warnings and precautions",
        "warnings",
        "precautions",
        "boxed warning",
        "black box warning",
        "black box",
        "cautions",
    ],
    "DRUG_INTERACTIONS": [
        "drug interactions",
        "interactions",
        "drug-drug interactions",
        "concomitant use",
    ],
    "DOSAGE_AND_ADMINISTRATION": [
        "dosage and administration",
        "dosage",
        "dosing",
        "administration",
        "dose",
        "how to take",
        "recommended dose",
    ],
    "INDICATIONS_AND_USAGE": [
        "indications and usage",
        "indications",
        "indicated for",
        "uses",
        "therapeutic use",
    ],
    "HOW_SUPPLIED": [
        "how supplied",
        "supply",
        "storage and handling",
        "available forms",
        "packaging",
    ],
    "DESCRIPTION": [
        "description",
        "composition",
        "formulation",
        "ingredients",
        "active ingredient",
    ],
    "PHARMACOKINETICS": [
        "pharmacokinetics",
        "absorption",
        "distribution",
        "metabolism",
        "elimination",
        "half-life",
        "bioavailability",
    ],
    "PREGNANCY_AND_LACTATION": [
        "pregnancy",
        "lactation",
        "breastfeeding",
        "nursing mothers",
        "reproductive",
        "teratogenic",
    ],
    "GERIATRIC_USE": [
        "geriatric use",
        "geriatric",
        "elderly",
        "older adults",
        "aged patients",
    ],
    "PEDIATRIC_USE": [
        "pediatric use",
        "pediatric",
        "children",
        "pediatric patients",
        "neonatal",
    ],
    "CLINICAL_STUDIES": [
        "clinical studies",
        "clinical trials",
        "efficacy studies",
        "safety studies",
    ],
    "ABUSE_AND_DEPENDENCE": [
        "abuse",
        "dependence",
        "drug abuse",
        "controlled substance",
        "addiction",
        "withdrawal",
    ],
    "NONCLINICAL_TOXICOLOGY": [
        "nonclinical toxicology",
        "carcinogenesis",
        "mutagenesis",
        "impairment of fertility",
        "animal pharmacology",
        "animal toxicology",
    ],
    "PATIENT_COUNSELING": [
        "patient counseling",
        "patient information",
        "medication guide",
        "patient medication information",
    ],
    "HEPATIC_IMPAIRMENT": [
        "hepatic impairment",
        "liver impairment",
        "liver disease",
        "hepatic",
    ],
    "RENAL_IMPAIRMENT": [
        "renal impairment",
        "kidney impairment",
        "kidney disease",
        "renal",
    ],
    "LABORATORY_TESTS": [
        "laboratory tests",
        "lab tests",
        "laboratory test interactions",
        "diagnostic tests",
    ],
    "MICROBIOLOGY": [
        "microbiology",
        "susceptibility",
        "resistance",
        "antimicrobial",
    ],
}

# Default boost weight applied when section intent matches a candidate chunk.
# This value is additive to the cross-encoder rerank score.
# Tunable: increase for stronger section preference, decrease for softer guidance.
DEFAULT_SECTION_BOOST_WEIGHT = 0.15
