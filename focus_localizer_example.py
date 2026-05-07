"""
Minimal integration example for base_Omni -> focus_localizer.
"""

from base_Omni import run_query_extraction
from focus_localizer import default_config, localize_all_queries


def run_example(video_path: str, report_text: str, query_model_path: str) -> list[dict]:
    query_result = run_query_extraction(
        report_text=report_text,
        query_model_path=query_model_path,
    )

    evidence_results = localize_all_queries(
        video_path=video_path,
        retrieval_queries=query_result.retrieval_queries,
        config=default_config,
    )
    return evidence_results
