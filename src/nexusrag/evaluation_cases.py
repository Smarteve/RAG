"""Ground truth for the repository's fictional two-page employee handbook."""

from nexusrag.schemas.evaluation import RetrievalDataset


def handbook_cases() -> RetrievalDataset:
    return RetrievalDataset.model_validate(
        {
            "dataset_id": "employee-handbook-v1",
            "document_sha256": "fa98cd8876120e82840d396a8060e3d91ed6eec1f5451ec41617ee0fe58fbd69",
            "cases": [
                {
                    "case_id": "annual-leave-days",
                    "question": "How many vacation days do full-time employees get each year?",
                    "expected_passages": [
                        {"page_number": 1, "text": "25 days of annual leave"}
                    ],
                },
                {
                    "case_id": "carry-over-limit",
                    "question": "How much unused holiday can I bring into next year?",
                    "expected_passages": [
                        {"page_number": 1, "text": "Up to 5 unused days"}
                    ],
                },
                {
                    "case_id": "carry-over-deadline",
                    "question": "When do I need to use the days carried over from last year?",
                    "expected_passages": [
                        {"page_number": 1, "text": "must be used by 31 March"}
                    ],
                },
                {
                    "case_id": "leave-request-notice",
                    "question": "How far in advance should I request time off?",
                    "expected_passages": [
                        {"page_number": 1, "text": "at least 10 working days"}
                    ],
                },
                {
                    "case_id": "remote-work-days",
                    "question": "How often can I work from home?",
                    "expected_passages": [
                        {
                            "page_number": 2,
                            "text": "up to 3 days per week, with manager approval",
                        }
                    ],
                },
                {
                    "case_id": "learning-budget",
                    "question": "How much money is available for my training each year?",
                    "expected_passages": [
                        {"page_number": 2, "text": "annual learning budget of GBP 600"}
                    ],
                },
                {
                    "case_id": "lost-laptop",
                    "question": "Who should I contact if I lose my work laptop?",
                    "expected_passages": [
                        {
                            "page_number": 2,
                            "text": "Report a lost company laptop to the IT team immediately.",
                        }
                    ],
                },
                {
                    "case_id": "unsupported-parking",
                    "question": "How much does employee parking cost per month?",
                    "answerable": False,
                    "expected_passages": [],
                },
            ],
        }
    )
