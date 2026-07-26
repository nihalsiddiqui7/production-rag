# from presidio_analyzer import AnalyzerEngine
# from presidio_anonymizer import AnonymizerEngine
# from langsmith import traceable

# analyzer = AnalyzerEngine()

# anonymizer = AnonymizerEngine()

# @traceable(name="anonymize_pii")
# def anonymize_pii(text: str):

#     results = analyzer.analyze(
#         text=text,
#         language="en"
#     )

#     anonymized = anonymizer.anonymize(
#         text=text,
#         analyzer_results=results
#     )

#     return anonymized.text


code = """
Production-grade PII anonymization with domain-aware false-positive filtering.

Why this matters:
    - Presidio (and most NER-based PII tools) flag common names as PERSON
    - "Adam optimizer" → "<PERSON> optimizer" breaks RAG retrieval
    - "Ada" (algorithm) → "<PERSON>" breaks queries
    - "Bert" (model) → "<PERSON>" breaks queries

Solution:
    - Maintain a domain whitelist of ML/tech terms that look like names
    - Filter analyzer results BEFORE anonymization
    - Log what was masked for auditability
    - Return both sanitized text and mask mapping for reversibility
"""

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from langsmith import traceable
import logging

logger = logging.getLogger("rag-api")

# ═══════════════════════════════════════════════════════════════
#  DOMAIN WHITELIST: ML/tech terms that NER falsely flags as PERSON
# ═══════════════════════════════════════════════════════════════

ML_WHITELIST = frozenset({
    # Optimizers
    "adam", "adamw", "sgd", "rmsprop", "adagrad", "adadelta", "nadam",
    # Activations
    "relu", "sigmoid", "softmax", "tanh", "leakyrelu", "gelu", "swish",
    # Models / Architectures
    "bert", "gpt", "t5", "resnet", "vgg", "inception", "transformer",
    "lstm", "gru", "cnn", "rnn", "gan", "vae", "diffusion",
    # Libraries / Frameworks
    "pytorch", "tensorflow", "keras", "scikit", "sklearn", "xgboost",
    "lightgbm", "catboost", "huggingface", "langchain", "openai",
    # Concepts
    "dropout", "batchnorm", "layernorm", "attention", "embedding",
    "backpropagation", "gradient", "descent", "momentum", "nesterov",
    # Misc names that are also ML terms
    "alex", "alexnet", "yann", "lecun", "geoffrey", "hinton", "yoshua", "bengio",
})

# PII types we actually care about masking
PII_ENTITIES = frozenset({
    "PERSON",          # names
    "PHONE_NUMBER",    # phone numbers
    "EMAIL_ADDRESS",   # emails
    "CREDIT_CARD",     # credit cards
    "IBAN",            # bank accounts
    "US_SSN",          # social security
    "US_PASSPORT",     # passport numbers
    "IP_ADDRESS",      # IPs
    "LOCATION",        # addresses (optional — can be noisy)
})


# ═══════════════════════════════════════════════════════════════
#  INIT ENGINES (singleton — reuse across requests)
# ═══════════════════════════════════════════════════════════════

_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()


@traceable(name="anonymize_pii")
def anonymize_pii(text: str) -> str:
    """
    Anonymize PII while preserving ML/technical terminology.

    Flow:
        1. Run Presidio analyzer on input text
        2. Filter out false positives (domain whitelist)
        3. Filter to only high-risk entity types (ignore TITLE, etc.)
        4. Anonymize remaining entities
        5. Log what was masked for audit trail

    Returns:
        Sanitized text with PII replaced by <ENTITY_TYPE> placeholders.
    """

    if not text or not text.strip():
        return text

    # 1. Analyze
    results = _analyzer.analyze(text=text, language="en")

    if not results:
        return text

    # 2. Filter: skip domain whitelist terms
    filtered = []
    skipped = []

    for r in results:
        entity_text = text[r.start:r.end]
        entity_lower = entity_text.lower().strip(".,;:!?")

        # Skip if entity is a known ML term
        if entity_lower in ML_WHITELIST:
            skipped.append((entity_text, r.entity_type))
            continue

        # Skip if entity type is not in our high-risk list
        if r.entity_type not in PII_ENTITIES:
            skipped.append((entity_text, r.entity_type))
            continue

        filtered.append(r)

    if skipped:
        logger.debug(f"PII filter skipped (false positives): {skipped}")

    if not filtered:
        return text

    # 3. Log what we\'re actually masking (audit trail)
    masked = [(text[r.start:r.end], r.entity_type) for r in filtered]
    logger.info(f"PII masked: {masked}")

    # 4. Anonymize
    anonymized = _anonymizer.anonymize(
        text=text,
        analyzer_results=filtered
    )

    return anonymized.text


@traceable(name="anonymize_pii_with_mapping")
def anonymize_pii_with_mapping(text: str) -> dict:
    """
    Same as anonymize_pii but returns a reversible mapping.

    Returns:
        {
            "sanitized": "Hi, I am <PERSON> and my email is <EMAIL_ADDRESS>",
            "mapping": {
                "<PERSON>": "John Doe",
                "<EMAIL_ADDRESS>": "john@example.com"
            }
        }

    Use case:
        - Store sanitized text in cache/logs
        - Reconstruct original if needed for debugging
    """

    if not text or not text.strip():
        return {"sanitized": text, "mapping": {}}

    results = _analyzer.analyze(text=text, language="en")

    if not results:
        return {"sanitized": text, "mapping": {}}

    filtered = []
    for r in results:
        entity_text = text[r.start:r.end]
        entity_lower = entity_text.lower().strip(".,;:!?")

        if entity_lower in ML_WHITELIST:
            continue
        if r.entity_type not in PII_ENTITIES:
            continue

        filtered.append(r)

    if not filtered:
        return {"sanitized": text, "mapping": {}}

    # Build mapping from placeholder → original value
    mapping = {}
    for r in filtered:
        placeholder = f"<{r.entity_type}>"
        original = text[r.start:r.end]
        # Handle duplicates: append index if same type appears multiple times
        if placeholder in mapping:
            idx = 1
            while f"{placeholder}_{idx}" in mapping:
                idx += 1
            placeholder = f"{placeholder}_{idx}"
        mapping[placeholder] = original

    anonymized = _anonymizer.anonymize(text=text, analyzer_results=filtered)

    return {
        "sanitized": anonymized.text,
        "mapping": mapping,
    }


