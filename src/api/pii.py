from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from langsmith import traceable

analyzer = AnalyzerEngine()

anonymizer = AnonymizerEngine()

@traceable(name="anonymize_pii")
def anonymize_pii(text: str):

    results = analyzer.analyze(
        text=text,
        language="en"
    )

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )

    return anonymized.text