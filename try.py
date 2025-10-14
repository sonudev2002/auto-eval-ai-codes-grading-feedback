"""
try.py — Local tester for the full code evaluation pipeline
-----------------------------------------------------------
Simulates the entire process:
1. Syntax validation
2. Code quality evaluation
3. Test case execution (simulated)
4. Plagiarism detection (mock)
5. Feedback generation (mock)
6. Score and grade computation
7. DB save simulation

Run:  python try.py
"""

import logging
from backend.evaluation_pipeline import EvaluationPipeline
from backend.code_evaluation import evaluate_quality

# Mocking dependent functions (if OpenAI/Plagiarism/DB unavailable)
from backend.feedback_generate import generate_feedback
from backend.plagiarism_detector import check_plagiarism
from backend.db import get_connection

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("try_pipeline")

# -----------------------------
# Input Section
# -----------------------------
sample_student_id = 101
sample_assignment_id = 55
sample_language = "python3"

# --- ✅ Try various source codes ---
# ✅ Correct code example
correct_code = """\
def add(a, b):
    return a + b

print(add(2, 3))
"""

# ❌ Syntax error example
syntax_error_code = """\
def add(a, b)
    return a + b
"""

# Choose which one to test
source_code = correct_code  # ⬅️ Change to syntax_error_code to test syntax error path


# -----------------------------
# Pipeline Execution
# -----------------------------
pipeline = EvaluationPipeline(timeout=10)
logger.info(f"🚀 Starting pipeline test for assignment={sample_assignment_id}")

result = pipeline.evaluate(
    assignment_id=sample_assignment_id,
    student_id=sample_student_id,
    code=source_code,
    language=sample_language,
)

# -----------------------------
# Results Display
# -----------------------------
print("\n" + "=" * 80)
if result.get("status") == "error" and result.get("stage") == "syntax_check":
    print("❌ Syntax Error detected:")
    print(f"Message: {result.get('message')}")
    print(f"Details: {result.get('details')}")
else:
    print("✅ Evaluation Completed Successfully")
    print(f"Assignment ID: {result['assignment_id']}")
    print(f"Student ID: {result['student_id']}")
    print(f"Language: {result['language']}")
    print(f"Grade: {result['grade']}")
    print(f"Score: {result['score']:.2f}")
    print(f"Feedback: {result['feedback']}")
    print(f"Plagiarism Score: {result['plagiarism']['score']}")
    print(
        f"Test Cases Passed: {result['test_results']['passed']}/{result['test_results']['total']}"
    )
    print(f"Syntax Error: {result['quality_metrics'].get('syntax_error')}")
    print(f"Code Length: {result['quality_metrics'].get('length')}")
print("=" * 80 + "\n")
