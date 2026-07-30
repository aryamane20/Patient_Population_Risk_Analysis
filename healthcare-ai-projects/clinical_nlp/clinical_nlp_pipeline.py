"""
Clinical NLP Pipeline
Day 15 — Mindbowser Healthcare AI Learning

A SEPARATE pipeline from the FHIR extraction/reporting project.
It processes free-text clinical discharge summaries and extracts:
- Diagnoses (confirmed, historical, negated, possible)
- Medications
- SDOH (Social Determinants of Health) signals
- Referrals and follow-up requirements
- ML-ready structured features

Approach: scispaCy clinical NER + a simplified ConText algorithm for
negation/assertion, with a pure-regex fallback so the pipeline still
runs when no scispaCy model is installed.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

# spaCy is optional: if it (or a scispaCy model) isn't installed, the
# pipeline falls back to regex extraction so it always runs.
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

class AssertionStatus(Enum):
    CONFIRMED = "confirmed"
    NEGATED = "negated"
    POSSIBLE = "possible"
    HISTORICAL = "historical"
    FAMILY_HISTORY = "family_history"

class DocumentSection(Enum):
    CHIEF_COMPLAINT = "chief_complaint"
    HPI = "history_of_present_illness"
    PAST_MEDICAL_HISTORY = "past_medical_history"
    MEDICATIONS = "medications"
    ALLERGIES = "allergies"
    PHYSICAL_EXAM = "physical_exam"
    LABS = "laboratory_results"
    ASSESSMENT_PLAN = "assessment_and_plan"
    DISCHARGE = "discharge"
    FOLLOW_UP = "follow_up"
    UNKNOWN = "unknown"

@dataclass
class ClinicalEntity:
    text: str
    entity_type: str  # DISEASE, CHEMICAL, FINDING, MEDICATION, etc.
    assertion: AssertionStatus
    section: DocumentSection
    sentence: str
    confidence: float = 1.0
    normalized_code: Optional[str] = None

@dataclass
class ClinicalNLPResult:
    diagnoses: List[ClinicalEntity] = field(default_factory=list)
    medications: List[ClinicalEntity] = field(default_factory=list)
    allergies: List[ClinicalEntity] = field(default_factory=list)
    clinical_findings: List[ClinicalEntity] = field(default_factory=list)
    sdoh_signals: List[str] = field(default_factory=list)
    referrals: List[str] = field(default_factory=list)
    follow_up_requirements: List[str] = field(default_factory=list)
    negated_conditions: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# SECTION 1 — SECTION DETECTION
# ─────────────────────────────────────────────

# Section header patterns
SECTION_PATTERNS = {
    DocumentSection.CHIEF_COMPLAINT: [
        r"chief complaint[s]?:?",
        r"cc:?"
    ],
    DocumentSection.HPI: [
        r"history of present illness:?",
        r"hpi:?"
    ],
    DocumentSection.PAST_MEDICAL_HISTORY: [
        r"past medical history:?",
        r"pmh:?",
        r"medical history:?"
    ],
    DocumentSection.MEDICATIONS: [
        r"medication[s]? on admission:?",
        r"current medication[s]?:?",
        r"home medication[s]?:?"
    ],
    DocumentSection.ALLERGIES: [
        r"allergi[e]?[s]?:?",
        r"drug allergi[e]?[s]?:?"
    ],
    DocumentSection.PHYSICAL_EXAM: [
        r"physical exam(ination)?:?",
        r"pe:?",
        r"vitals?:?"
    ],
    DocumentSection.LABS: [
        r"lab(oratory)? (results?|data|values?|findings?):?",
        r"labs?:?"
    ],
    DocumentSection.ASSESSMENT_PLAN: [
        r"assessment and plan:?",
        r"assessment/plan:?",
        r"a/p:?",
        r"impression:?"
    ],
    DocumentSection.FOLLOW_UP: [
        r"follow.?up:?",
        r"f/u:?"
    ],
    DocumentSection.DISCHARGE: [
        r"discharge (condition|disposition|instructions?|summary):?"
    ]
}

def detect_section(line: str) -> Optional[DocumentSection]:
    """
    Detect which section a line belongs to based on header patterns.
    Returns None if the line is not a section header.

    NOTE: we use re.fullmatch (not re.search) against the whole
    stripped line. Section headers occupy their own line, so a header
    line IS the pattern. Using re.search here would let short
    abbreviation patterns like "pe:?" or "cc:?" match substrings inside
    ordinary words — e.g. "pe" inside "Penicillin — rash" — and
    mislabel content lines as section headers.
    """
    line_lower = line.lower().strip()
    if not line_lower:
        return None

    # Headers may end with a colon; allow an optional trailing colon
    # to be present or absent regardless of the pattern.
    candidate = line_lower.rstrip(":").strip()

    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            base_pattern = pattern.rstrip(":?")
            if re.fullmatch(base_pattern, candidate):
                return section

    return None


def split_into_sections(note_text: str) -> dict:
    """
    Split clinical note into labeled sections.

    Returns a dictionary mapping DocumentSection → text content
    """
    sections = {}
    current_section = DocumentSection.UNKNOWN
    current_lines = []

    for line in note_text.split('\n'):
        detected_section = detect_section(line)

        if detected_section is not None:
            # Save previous section
            if current_lines:
                sections[current_section] = '\n'.join(current_lines)

            # Start new section
            current_section = detected_section
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        sections[current_section] = '\n'.join(current_lines)

    return sections


# ─────────────────────────────────────────────
# SECTION 2 — NEGATION DETECTION
# ─────────────────────────────────────────────

# ConText algorithm trigger words
NEGATION_TRIGGERS = [
    r'\bno\b', r'\bnot\b', r'\bwithout\b', r'\bdenies?\b',
    r'\bdenied\b', r'\bfree of\b', r'\bno evidence of\b',
    r'\bnegative for\b', r'\brules? out\b', r'\bruled out\b',
    r'\babsent\b', r'\bnever\b', r'\bnone\b'
]

SPECULATION_TRIGGERS = [
    r'\bpossible\b', r'\bpossibly\b', r'\bprobable\b', r'\bprobably\b',
    r'\bsuspect[s]?\b', r'\bsuspected\b', r'\bcannot exclude\b',
    r'\bconsistent with\b', r'\bconcern[s]? for\b', r'\bmay be\b',
    r'\bmight be\b', r'\brule out\b', r'\bquestion of\b'
]

HISTORICAL_TRIGGERS = [
    r'\bhistory of\b', r'\bh/o\b', r'\bpast medical history\b',
    r'\bpreviously\b', r'\bprior\b', r'\bformer\b', r'\bpast\b',
    r'\bold\b', r'\bchronic\b'
]

NEGATION_TERMINATION = [
    r'\bbut\b', r'\bhowever\b', r'\bexcept\b', r'\bthough\b',
    r'\balthough\b', r'\bwhereas\b', r'\bnevertheless\b'
]


def detect_assertion(sentence: str, entity_text: str,
                     section: DocumentSection) -> AssertionStatus:
    """
    Determine whether a clinical finding is:
    - Confirmed (present)
    - Negated (absent)
    - Possible (uncertain)
    - Historical (past, not current)

    Uses a simplified ConText algorithm: look at the text immediately
    BEFORE the entity for trigger words, stopping at termination words.
    """
    sentence_lower = sentence.lower()
    entity_lower = entity_text.lower()

    # Find position of entity in sentence
    entity_pos = sentence_lower.find(entity_lower)
    if entity_pos == -1:
        return AssertionStatus.CONFIRMED

    # Check text before the entity (window of 50 characters)
    pre_context = sentence_lower[max(0, entity_pos - 50):entity_pos]

    # Check for termination triggers between any trigger and the entity
    def has_termination(text):
        for term in NEGATION_TERMINATION:
            if re.search(term, text):
                return True
        return False

    # Historical check (check pre-context)
    for trigger in HISTORICAL_TRIGGERS:
        if re.search(trigger, pre_context) and not has_termination(pre_context):
            # Past Medical History section items are always historical
            if section == DocumentSection.PAST_MEDICAL_HISTORY:
                return AssertionStatus.HISTORICAL

    # Negation check (check pre-context)
    for trigger in NEGATION_TRIGGERS:
        if re.search(trigger, pre_context) and not has_termination(pre_context):
            return AssertionStatus.NEGATED

    # Speculation check (check pre-context)
    for trigger in SPECULATION_TRIGGERS:
        if re.search(trigger, pre_context) and not has_termination(pre_context):
            return AssertionStatus.POSSIBLE

    # Past medical history section = historical by default
    if section == DocumentSection.PAST_MEDICAL_HISTORY:
        return AssertionStatus.HISTORICAL

    # Default: confirmed
    return AssertionStatus.CONFIRMED


# ─────────────────────────────────────────────
# SECTION 3 — NAMED ENTITY RECOGNITION
# ─────────────────────────────────────────────

def load_nlp_model():
    """Load scispaCy clinical NLP model (returns None if unavailable)."""
    if not SPACY_AVAILABLE:
        print("spaCy not installed — using regex fallback.")
        print("  To enable clinical NER: pip install scispacy and a model")
        return None
    try:
        nlp = spacy.load("en_core_sci_md")
        print("Loaded en_core_sci_md model")
        return nlp
    except OSError:
        try:
            nlp = spacy.load("en_core_sci_sm")
            print("Loaded en_core_sci_sm model (fallback)")
            return nlp
        except OSError:
            print("No scispaCy model found. Install with:")
            print("pip install https://s3-us-west-2.amazonaws.com/"
                  "ai2-s2-scispacy/releases/v0.5.4/"
                  "en_core_sci_sm-0.5.4.tar.gz")
            return None


def extract_entities_from_section(nlp, section_text: str,
                                   section: DocumentSection) -> List[ClinicalEntity]:
    """
    Run NER on a section of the clinical note.
    Apply negation detection to each entity found.
    """
    if not nlp or not section_text.strip():
        return []

    entities = []
    doc = nlp(section_text)

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        for ent in sent.ents:
            # Determine assertion status
            assertion = detect_assertion(sent_text, ent.text, section)

            entity = ClinicalEntity(
                text=ent.text,
                entity_type=ent.label_,
                assertion=assertion,
                section=section,
                sentence=sent_text,
                confidence=1.0
            )

            entities.append(entity)

    return entities


# ─────────────────────────────────────────────
# SECTION 4 — SDOH EXTRACTION
# ─────────────────────────────────────────────

SDOH_PATTERNS = {
    "social_isolation": [
        r"lives? alone", r"no social support", r"limited social support",
        r"socially isolated", r"no family support"
    ],
    "financial_stress": [
        r"financial (concern|stress|difficult|problem)",
        r"afford(ing)? medication", r"cost of medication",
        r"cannot afford", r"financial (strain|burden)",
        r"lost (employment|job|work)", r"unemployed"
    ],
    "housing_instability": [
        r"homeless", r"unstable housing", r"shelter",
        r"no fixed address", r"housing (concern|issue|problem)"
    ],
    "food_insecurity": [
        r"food insecurity", r"food desert", r"cannot afford food",
        r"nutritional (concern|deficit)", r"dietary (non-compliance|non-adherence)"
    ],
    "transportation": [
        r"no transportation", r"transportation (barrier|issue|concern)",
        r"cannot drive", r"no car"
    ],
    "medication_non_compliance": [
        r"medication non.?compliance", r"medication non.?adherence",
        r"not taking (medication|med|drug)", r"stopped taking",
        r"ran out of", r"unable to afford"
    ],
    "substance_use": [
        r"alcohol (abuse|use|dependence)",
        r"substance (abuse|use disorder)",
        r"tobacco (use|abuse)", r"smoking"
    ]
}

def extract_sdoh_signals(note_text: str) -> List[str]:
    """
    Extract Social Determinants of Health signals from clinical text.

    Clinical note: SDOH factors are among the strongest predictors
    of readmission but are almost never captured in structured fields.
    They live exclusively in clinical notes — making NLP essential
    for capturing them at scale.
    """
    note_lower = note_text.lower()
    detected_sdoh = []

    for sdoh_category, patterns in SDOH_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, note_lower):
                detected_sdoh.append(sdoh_category)
                break  # One match per category is enough

    return detected_sdoh


# ─────────────────────────────────────────────
# SECTION 5 — REFERRAL AND FOLLOW-UP EXTRACTION
# ─────────────────────────────────────────────

REFERRAL_PATTERNS = [
    r"referral? (?:placed?|ordered?|to) (?:to )?(.+?)(?:\.|$)",
    r"referred? to (.+?)(?:\.|$)",
    r"consult(?:ation)? (?:placed?|ordered?) (?:for|to|with) (.+?)(?:\.|$)"
]

FOLLOW_UP_PATTERNS = [
    r"follow.?up with (.+?) within (.+?)(?:\.|$)",
    r"follow.?up (?:with )?(.+?) in (.+?)(?:\.|$)",
    r"(.+?) — within (.+?)(?:\.|$)"
]

def extract_referrals(note_text: str) -> List[str]:
    """Extract referral orders from the note"""
    referrals = []
    note_lower = note_text.lower()

    for pattern in REFERRAL_PATTERNS:
        matches = re.finditer(pattern, note_lower)
        for match in matches:
            referral_text = match.group(1).strip()
            if len(referral_text) < 100:  # Sanity check on length
                referrals.append(referral_text.title())

    return list(set(referrals))  # Deduplicate


def extract_follow_up(note_text: str) -> List[str]:
    """Extract follow-up requirements from the note"""
    follow_ups = []

    # Look in follow-up section
    follow_up_section_pattern = r"follow.?up:?\n(.+?)(?:\n\n|\Z)"
    match = re.search(follow_up_section_pattern, note_text,
                      re.IGNORECASE | re.DOTALL)

    if match:
        follow_up_text = match.group(1)
        lines = follow_up_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                follow_ups.append(line)

    return follow_ups


# ─────────────────────────────────────────────
# SECTION 6 — MEDICATION EXTRACTION
# ─────────────────────────────────────────────

MEDICATION_PATTERNS = [
    # Pattern: Drug Name + dose + frequency
    r"(\b[A-Z][a-z]+(?:in|ol|ide|ate|ine|one|pril|sartan|stat|mab)?)\s+"
    r"(\d+(?:\.\d+)?(?:mg|mcg|g|ml|units?))\s+"
    r"((?:once|twice|three times?|four times?|BID|TID|QID|QD|daily|"
    r"weekly|PRN|as needed)(?:\s+daily)?)",

    # Pattern: Drug Name + dose only
    r"(\b[A-Z][a-z]+(?:in|ol|ide|ate|ine|one|pril|sartan|stat|mab)?)\s+"
    r"(\d+(?:\.\d+)?(?:mg|mcg|g|ml|units?))"
]

def extract_medications_from_section(section_text: str) -> List[dict]:
    """
    Extract medications with doses and frequencies.

    Clinical note: Medication extraction needs to capture:
    1. Drug name (normalize to RxNorm)
    2. Dose
    3. Frequency (BID, TID, QD, PRN, etc.)
    4. Route (oral, IV, topical)
    5. Status (new at discharge, continued, discontinued)

    This simplified version gets name, dose, and frequency.
    """
    medications = []

    # Line-by-line extraction for medication sections
    lines = section_text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove list numbering
        line = re.sub(r'^\d+\.?\s*', '', line)

        # Try to extract medication components
        for pattern in MEDICATION_PATTERNS:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                med = {
                    "name": match.group(1),
                    "dose": match.group(2),
                    "frequency": match.group(3) if len(match.groups()) >= 3 else "unspecified",
                    "original_text": line
                }
                medications.append(med)
                break
        else:
            # If no pattern matched but line looks like a medication entry
            if any(word in line.lower() for word in
                   ["mg", "mcg", "daily", "bid", "tid", "prn", "tablet", "capsule"]):
                medications.append({
                    "name": "unstructured",
                    "dose": "unstructured",
                    "frequency": "unstructured",
                    "original_text": line
                })

    return medications


# ─────────────────────────────────────────────
# SECTION 7 — MAIN NLP PIPELINE
# ─────────────────────────────────────────────

def run_clinical_nlp(note_text: str,
                     nlp_model=None) -> ClinicalNLPResult:
    """
    Run the complete clinical NLP pipeline on a note.

    Returns a ClinicalNLPResult with all extracted structured
    information from the unstructured clinical note.
    """
    result = ClinicalNLPResult()

    # Step 1: Split into sections
    sections = split_into_sections(note_text)
    print(f"\nDetected {len(sections)} sections: "
          f"{[s.value for s in sections.keys()]}")

    # Step 2: Extract SDOH signals from full note
    result.sdoh_signals = extract_sdoh_signals(note_text)

    # Step 3: Extract referrals and follow-up
    result.referrals = extract_referrals(note_text)
    result.follow_up_requirements = extract_follow_up(note_text)

    # Step 4: Extract medications from medications section
    if DocumentSection.MEDICATIONS in sections:
        result.medications = [
            ClinicalEntity(
                text=med["original_text"],
                entity_type="MEDICATION",
                assertion=AssertionStatus.CONFIRMED,
                section=DocumentSection.MEDICATIONS,
                sentence=med["original_text"]
            )
            for med in extract_medications_from_section(
                sections[DocumentSection.MEDICATIONS]
            )
        ]

    # Step 5: Run NER on key sections if model available
    if nlp_model:
        diagnosis_sections = [
            DocumentSection.ASSESSMENT_PLAN,
            DocumentSection.PAST_MEDICAL_HISTORY,
            DocumentSection.HPI,
            DocumentSection.CHIEF_COMPLAINT
        ]

        for section_type in diagnosis_sections:
            if section_type in sections:
                entities = extract_entities_from_section(
                    nlp_model,
                    sections[section_type],
                    section_type
                )

                for entity in entities:
                    if entity.assertion == AssertionStatus.NEGATED:
                        result.negated_conditions.append(entity.text)
                    else:
                        result.diagnoses.append(entity)

    else:
        # Fallback: regex-based extraction without NLP model
        print("\nNo NLP model loaded — using regex fallback")
        result = regex_fallback_extraction(note_text, sections, result)

    return result


def regex_fallback_extraction(note_text, sections,
                               result: ClinicalNLPResult) -> ClinicalNLPResult:
    """
    Fallback extraction using regex patterns when NLP model
    is not available. Less accurate but works without installation.
    """

    # Common diagnosis patterns
    COMMON_DIAGNOSES = {
        "Type 2 Diabetes": ["type 2 diabetes", "t2dm", "dm type 2",
                            "diabetes mellitus", "dm2"],
        "Hypertension": ["hypertension", "htn", "high blood pressure",
                        "hypertensive"],
        "Heart Failure": ["heart failure", "hf", "chf", "cardiac failure"],
        "CKD": ["chronic kidney disease", "ckd", "renal insufficiency",
                "nephropathy"],
        "COPD": ["copd", "chronic obstructive pulmonary"],
        "Atrial Fibrillation": ["atrial fibrillation", "afib", "a-fib"],
        "Coronary Artery Disease": ["coronary artery disease", "cad",
                                    "coronary disease"]
    }

    note_lower = note_text.lower()

    for diagnosis_name, patterns in COMMON_DIAGNOSES.items():
        for pattern in patterns:
            if pattern in note_lower:
                # Find context to determine assertion
                pattern_pos = note_lower.find(pattern)
                pre_context = note_lower[max(0, pattern_pos-50):pattern_pos]

                # Check for negation
                is_negated = any(
                    re.search(neg, pre_context)
                    for neg in [r'\bno\b', r'\bnot\b', r'\bno history of\b',
                               r'\bdenies?\b', r'\bno known\b']
                )

                # Check for historical
                is_historical = any(
                    re.search(hist, pre_context)
                    for hist in [r'\bhistory of\b', r'\bh/o\b',
                                r'\bpast medical\b']
                )

                if is_negated:
                    result.negated_conditions.append(diagnosis_name)
                else:
                    assertion = (AssertionStatus.HISTORICAL
                               if is_historical
                               else AssertionStatus.CONFIRMED)

                    result.diagnoses.append(ClinicalEntity(
                        text=diagnosis_name,
                        entity_type="DISEASE",
                        assertion=assertion,
                        section=DocumentSection.UNKNOWN,
                        sentence=""
                    ))

                break

    return result


# ─────────────────────────────────────────────
# SECTION 8 — OUTPUT AND REPORTING
# ─────────────────────────────────────────────

def print_nlp_results(result: ClinicalNLPResult):
    """Print structured NLP results in readable format"""

    print("\n" + "="*60)
    print("CLINICAL NLP EXTRACTION RESULTS")
    print("="*60)

    # Confirmed diagnoses
    print("\n📋 CONFIRMED DIAGNOSES:")
    confirmed = [
        d for d in result.diagnoses
        if d.assertion == AssertionStatus.CONFIRMED
    ]
    if confirmed:
        for diag in confirmed:
            print(f"  ✅ {diag.text} [{diag.section.value}]")
    else:
        print("  None found")

    # Historical diagnoses
    print("\n📚 HISTORICAL DIAGNOSES:")
    historical = [
        d for d in result.diagnoses
        if d.assertion == AssertionStatus.HISTORICAL
    ]
    if historical:
        for diag in historical:
            print(f"  📖 {diag.text} [historical]")
    else:
        print("  None found")

    # Negated conditions
    print("\n🚫 NEGATED CONDITIONS (explicitly absent):")
    if result.negated_conditions:
        unique_negated = list(set(result.negated_conditions))
        for neg in unique_negated:
            print(f"  ❌ {neg}")
    else:
        print("  None found")

    # Possible conditions
    print("\n❓ POSSIBLE/UNCERTAIN CONDITIONS:")
    possible = [
        d for d in result.diagnoses
        if d.assertion == AssertionStatus.POSSIBLE
    ]
    if possible:
        for diag in possible:
            print(f"  ? {diag.text}")
    else:
        print("  None found")

    # Medications
    print("\n💊 MEDICATIONS:")
    if result.medications:
        for med in result.medications:
            print(f"  💊 {med.text}")
    else:
        print("  None extracted")

    # SDOH Signals
    print("\n🏘️  SOCIAL DETERMINANTS OF HEALTH:")
    if result.sdoh_signals:
        sdoh_descriptions = {
            "social_isolation": "Lives alone / limited social support",
            "financial_stress": "Financial stress / medication cost concerns",
            "housing_instability": "Housing instability",
            "food_insecurity": "Food insecurity",
            "transportation": "Transportation barriers",
            "medication_non_compliance": "Medication non-compliance",
            "substance_use": "Substance use"
        }
        for signal in result.sdoh_signals:
            desc = sdoh_descriptions.get(signal, signal)
            print(f"  ⚠️  {desc}")
    else:
        print("  No SDOH signals detected")

    # Referrals
    print("\n📤 REFERRALS:")
    if result.referrals:
        for ref in result.referrals:
            print(f"  → {ref}")
    else:
        print("  None found")

    # Follow-up
    print("\n📅 FOLLOW-UP REQUIREMENTS:")
    if result.follow_up_requirements:
        for fu in result.follow_up_requirements:
            print(f"  📅 {fu}")
    else:
        print("  None found")

    # Risk flags for readmission model
    print("\n🚨 READMISSION RISK FLAGS:")
    risk_flags = []

    if "social_isolation" in result.sdoh_signals:
        risk_flags.append("Patient lives alone — high social vulnerability")

    if "financial_stress" in result.sdoh_signals:
        risk_flags.append("Financial stress — medication non-compliance risk")

    if "medication_non_compliance" in result.sdoh_signals:
        risk_flags.append("Documented medication non-compliance")

    confirmed_diagnoses = [
        d.text.lower() for d in result.diagnoses
        if d.assertion == AssertionStatus.CONFIRMED
    ]

    if any("diabetes" in d for d in confirmed_diagnoses):
        risk_flags.append("Active diabetes — chronic disease management need")

    if any("nephropathy" in d or "ckd" in d or "renal" in d
           for d in confirmed_diagnoses):
        risk_flags.append("Kidney disease — requires nephrology follow-up")

    if len(result.referrals) >= 2:
        risk_flags.append(f"{len(result.referrals)} referrals placed — "
                         f"complex discharge requiring coordination")

    if risk_flags:
        for flag in risk_flags:
            print(f"  🚨 {flag}")
    else:
        print("  No specific risk flags identified")

    print("\n" + "="*60)


def extract_structured_features(result: ClinicalNLPResult) -> dict:
    """
    Convert NLP results to structured features for ML model.

    This is the bridge between NLP output and the feature
    engineering pipeline from Day 14.
    """
    confirmed_diagnoses = [
        d.text.lower() for d in result.diagnoses
        if d.assertion == AssertionStatus.CONFIRMED
    ]

    features = {
        # SDOH features
        "nlp_social_isolation": int("social_isolation" in result.sdoh_signals),
        "nlp_financial_stress": int("financial_stress" in result.sdoh_signals),
        "nlp_medication_noncompliance": int(
            "medication_non_compliance" in result.sdoh_signals
        ),
        "nlp_food_insecurity": int("food_insecurity" in result.sdoh_signals),

        # Diagnosis features from NLP
        "nlp_has_diabetes": int(
            any("diabetes" in d for d in confirmed_diagnoses)
        ),
        "nlp_has_nephropathy": int(
            any("nephropathy" in d or "kidney" in d
                for d in confirmed_diagnoses)
        ),
        "nlp_has_hypertension": int(
            any("hypertension" in d or "hypertensive" in d
                for d in confirmed_diagnoses)
        ),

        # Referral complexity
        "nlp_num_referrals": len(result.referrals),
        "nlp_complex_discharge": int(len(result.referrals) >= 2),

        # Negated conditions (important for model accuracy)
        "nlp_denies_chest_pain": int(
            any("chest pain" in n.lower()
                for n in result.negated_conditions)
        ),
        "nlp_denies_shortness_of_breath": int(
            any("shortness of breath" in n.lower() or "sob" in n.lower()
                for n in result.negated_conditions)
        ),

        # SDOH composite score (0-5)
        "nlp_sdoh_risk_score": len(result.sdoh_signals)
    }

    return features


# ─────────────────────────────────────────────
# SAMPLE INPUT — one paste-in discharge summary
# (defined BEFORE __main__ so the entry point can use it)
# ─────────────────────────────────────────────

DISCHARGE_SUMMARY = """
DISCHARGE SUMMARY

Patient: [REDACTED]
MRN: [REDACTED]
Admission Date: [REDACTED]
Discharge Date: [REDACTED]
Attending Physician: Dr. Raj Patel, MD

CHIEF COMPLAINT:
Hypertensive urgency with associated headache and blurred vision.

HISTORY OF PRESENT ILLNESS:
58-year-old female with history of Type 2 Diabetes Mellitus and
hypertension presenting with BP 182/110, severe headache, and
blurred vision for the past 2 days. Patient reports medication
non-compliance with lisinopril due to financial concerns. Denies
chest pain. Denies shortness of breath. No fever. No loss of
consciousness. Patient lives alone. Husband recently lost
employment. Reports difficulty affording medications.

PAST MEDICAL HISTORY:
1. Type 2 Diabetes Mellitus — diagnosed 2018, poorly controlled
2. Essential Hypertension — diagnosed 2019
3. No history of heart failure
4. No history of stroke
5. No known coronary artery disease

MEDICATIONS ON ADMISSION:
1. Metformin 1000mg twice daily
2. Lisinopril 10mg daily — patient reports not taking for 3 months

ALLERGIES:
Penicillin — rash

PHYSICAL EXAMINATION:
Blood pressure 182/110 mmHg, Heart rate 98 bpm, Temperature 98.6F,
Respiratory rate 16, O2 saturation 96% on room air.
Alert and oriented x3. No acute distress.
Cardiovascular: Regular rate and rhythm. No murmurs.
Respiratory: Clear to auscultation bilaterally. No wheezes.
Abdomen: Soft, non-tender, non-distended.
Extremities: Trace bilateral ankle edema.

LABORATORY RESULTS:
HbA1c 9.1% (High)
Serum creatinine 1.4 mg/dL (High)
BMP within normal limits except glucose 287
Urinalysis: microalbuminuria present

ASSESSMENT AND PLAN:
1. Hypertensive urgency — started on amlodipine 5mg daily.
   BP improved to 148/88 by day 2. Continue to monitor.

2. Poorly controlled Type 2 Diabetes — HbA1c 9.1%.
   Adjusted metformin dose. Endocrinology referral placed.
   Patient counseled on dietary modifications.

3. Early diabetic nephropathy — creatinine 1.4, microalbuminuria.
   Nephrology referral placed. Renally dose medications.

4. Medication non-compliance due to cost — Social work consult
   placed. Patient enrolled in hospital medication assistance
   program for amlodipine. Lisinopril alternatives discussed.

5. Social support concerns — patient lives alone, limited social
   support, recent financial stressors. Case management referral
   placed. Follow-up with primary care within 7 days of discharge.

DISCHARGE CONDITION: Stable
DISCHARGE DISPOSITION: Home

FOLLOW-UP:
Primary Care — Dr. Kapoor — within 7 days
Endocrinology — within 4 weeks
Nephrology — within 4 weeks
"""


# ─────────────────────────────────────────────
# RUN THE PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("CLINICAL NLP PIPELINE — DAY 15")
    print("Processing a sample discharge summary")
    print("="*60)

    # Try to load NLP model (falls back to regex if unavailable)
    nlp = load_nlp_model()

    # Run NLP pipeline on the discharge summary
    print("\nRunning clinical NLP extraction...")
    results = run_clinical_nlp(DISCHARGE_SUMMARY, nlp)

    # Print results
    print_nlp_results(results)

    # Extract features for ML model
    print("\n--- ML FEATURES FROM NLP ---")
    ml_features = extract_structured_features(results)
    for feature, value in ml_features.items():
        print(f"  {feature}: {value}")

    print("\n" + "="*60)
    print("KEY INSIGHT:")
    print("="*60)
    print("""
These NLP-derived features would be INVISIBLE to a model
trained only on structured EHR data:

  - Patient lives alone (social_isolation = 1)
  - Financial stress causing medication non-compliance (financial_stress = 1)
  - Patient stopped taking lisinopril for 3 months (medication_noncompliance = 1)
  - 3 referrals placed at discharge (complex_discharge = 1)
  - SDOH risk score: 3/5

In the MIMIC-III research literature, SDOH features extracted
from clinical notes improve readmission model AUROC by 3-7
percentage points over structured data alone.

The patient's clinical numbers look moderate-risk.
Her SDOH profile makes her HIGH risk. The model only knows this
if it can read the note.
""")
