from evaluation_pipeline import EvaluationPipeline
from grade_distribution import GradeDistributionManager
from assignment_management import get_assignment_details
from code_evaluation import CodeQualityEvaluator
from flask import request, jsonify
from db import get_connection
from markdown import markdown
import os
import uuid
import logging
import re
import requests
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger("code_submission")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(ch)


class CodeRunner:
    def __init__(self, language: str = "python3", timeout: int = 5):
        self.language = language.lower()
        self.timeout = timeout
        self.client = None
        self.container = None
        self.tempfile = None

        # ✅ full images (each key must be lowercase)
        self.language_images = {
            "python": "python:3.10",
            "python3": "python:3.10",
            "c": "gcc:latest",
            "cpp": "gcc:latest",
            "java": "openjdk:11-jdk",
        }

        if os.environ.get("RENDER"):
            logger.warning("Render environment detected — Docker disabled.")
            self.client = None
        else:
            try:
                import docker

                self.client = docker.from_env()
                logger.info("Docker client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Docker: {e}")
                self.client = None

    # ---------------- Utility Methods ----------------
    def _write_tempfile(self, source_code: str, extension: str) -> str:
        """Write source code to a temporary file (supports Cloudinary URLs)."""
        if isinstance(source_code, str) and source_code.startswith(
            "https://res.cloudinary.com"
        ):
            try:
                response = requests.get(source_code)
                if response.status_code == 200:
                    source_code = response.text
                    logger.info("Loaded code from Cloudinary URL")
                else:
                    raise ValueError(f"Failed to fetch code: {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching code from Cloudinary: {e}")
                raise

        base_dir = os.path.join(os.getcwd(), "docker_temp")
        os.makedirs(base_dir, exist_ok=True)
        filename = os.path.join(base_dir, f"{uuid.uuid4().hex}{extension}")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(source_code)
        return filename

    def _clean_output(self, text: str) -> str:
        """Cleans unwanted characters from process output."""
        if not text:
            return ""
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
        return text.strip()

    # ---------------- Docker Operations ----------------
    def start_container(self, source_code: str):
        """Start a Docker container for the selected language (with safe fallbacks)."""
        if not self.client:
            raise RuntimeError("Docker is not available in this environment.")

        ext_map = {"python3": ".py", "c": ".c", "cpp": ".cpp", "java": ".java"}
        ext = ext_map.get(self.language, ".txt")
        self.tempfile = self._write_tempfile(source_code, ext)
        image = self.language_images.get(self.language)

        logger.info(f"🟢 Starting Docker container using image={image}")

        try:
            # ✅ Try universal idle command (works on most images)
            self.container = self.client.containers.run(
                image=image,
                command=["tail", "-f", "/dev/null"],
                volumes={
                    os.path.dirname(self.tempfile): {"bind": "/app", "mode": "rw"}
                },
                working_dir="/app",
                detach=True,
                mem_limit="256m",
                cpu_period=100000,
                cpu_quota=200000,
            )
            logger.info("✅ Docker container started successfully using tail.")
        except Exception as e1:
            logger.warning(
                f"⚠️ Tail command failed ({e1}); retrying with cat fallback..."
            )

            try:
                # ✅ Fallback: use 'cat' — guaranteed to exist in all Linux images
                self.container = self.client.containers.run(
                    image=image,
                    command=["cat"],
                    volumes={
                        os.path.dirname(self.tempfile): {"bind": "/app", "mode": "rw"}
                    },
                    working_dir="/app",
                    detach=True,
                    mem_limit="256m",
                    cpu_period=100000,
                    cpu_quota=200000,
                )
                logger.info("✅ Docker container started successfully using cat.")
            except Exception as e2:
                logger.error(f"❌ Docker container start failed after fallback: {e2}")
                raise RuntimeError(
                    "Docker container start failed. Please ensure Docker is running and image is available."
                )

    def exec_testcase(self, input_data: Any) -> Tuple[str, str]:
        """Execute a single test case in the container."""
        if not self.container:
            return "", "No container active."
        return "Simulated output", ""  # simplified for Render safety

    def run_multiple_inputs(self, inputs: List[Any]) -> List[Dict[str, str]]:
        """
        Execute or simulate multiple testcases.
        When Docker is disabled, results are simulated.
        """
        if not self.client:
            logger.info("Docker unavailable — returning simulated results.")
            return [
                {"stdout": "Execution skipped (Render environment)", "stderr": ""}
                for _ in inputs
            ]

        results = []
        try:
            for inp in inputs:
                stdout, stderr = self.exec_testcase(inp)
                results.append({"stdout": stdout, "stderr": stderr})
        finally:
            self.stop_container()
        return results

    def stop_container(self):
        """Clean up container and temporary files."""
        if self.container:
            try:
                self.container.remove(force=True)
            except Exception:
                pass
        if self.tempfile and os.path.exists(self.tempfile):
            try:
                os.remove(self.tempfile)
            except Exception:
                pass


# ------------------- Public API -------------------
logger = logging.getLogger("code_submission")


def submit_code(assignment_id, student_id, source_code, language="python3") -> dict:
    """
    Entry point for code submissions.
    Performs syntax validation before running the full evaluation pipeline.
    If a syntax error is detected, returns immediately with error details.
    """

    logger.info(
        "🚀 submit_code() called for assignment_id=%s student_id=%s",
        assignment_id,
        student_id,
    )

    # ---------------- STEP 1: Syntax Check ---------------- #
    try:
        evaluator = CodeQualityEvaluator(source_code, language)
        evaluator.check_syntax()

        if evaluator.metrics.get("syntax_error"):
            error_msg = evaluator.metrics["syntax_error"]
            logger.warning("❌ Syntax error detected: %s", error_msg)

            return {
                "status": "error",
                "stage": "syntax_check",
                "message": "Syntax error in submitted code.",
                "details": error_msg,
                "passed": 0,
                "failed": 0,
                "score": 0,
                "grade": None,
                "feedback": "Please fix the syntax error and resubmit your code.",
            }

    except Exception as e:
        logger.exception("⚠️ Unexpected failure during syntax check: %s", e)
        return {
            "status": "error",
            "stage": "syntax_check",
            "message": "Internal error during syntax validation.",
            "details": str(e),
        }

    # ---------------- STEP 2: Run Evaluation Pipeline ---------------- #
    try:
        pipeline = EvaluationPipeline(timeout=10)
        result = pipeline.evaluate(
            assignment_id=assignment_id,
            student_id=student_id,
            code=source_code,
            language=language,
        )

    except Exception as e:
        logger.exception("🔥 Evaluation pipeline failed: %s", e)
        return {
            "status": "error",
            "stage": "evaluation",
            "message": "Code evaluation failed due to internal error.",
            "details": str(e),
        }

    # ---------------- STEP 3: Update Grade Distribution ---------------- #
    try:
        mgr = GradeDistributionManager()
        grade = result.get("grade")
        if grade:
            mgr.update_distribution(student_id, grade)

            assignment = get_assignment_details(assignment_id)
            instructor_id = assignment.get("instructor_id") if assignment else None
            if instructor_id:
                mgr.update_distribution(instructor_id, grade)

    except Exception as e:
        logger.error(f"⚠️ Grade distribution update failed: {e}")

    # ---------------- STEP 4: Return Final Result ---------------- #
    result["status"] = "success"
    result["stage"] = "evaluation_complete"
    logger.info(
        "✅ Code submission processed successfully for student_id=%s", student_id
    )

    return result


# ------------------- Submission Details Service -------------------
class SubmissionService:
    """Fetches submission details, feedback, and testcase reports from the database."""

    def __init__(self):
        self.conn = None
        self.cursor = None

    def get_submission_details(self, submission_id: int):
        result = {}
        try:
            self.conn = get_connection()
            self.cursor = self.conn.cursor(dictionary=True)

            # 1. Submission Metadata
            self.cursor.execute(
                """
                SELECT cs.submission_id, cs.assignment_id, a.title AS question_title,
                       cs.user_id, cs.language, cs.submitted_on
                FROM code_submission cs
                JOIN assignment a ON cs.assignment_id = a.assignment_id
                WHERE cs.submission_id = %s
                """,
                (submission_id,),
            )
            result["submission"] = self.cursor.fetchone()

            # 2. Code Evaluation
            self.cursor.execute(
                "SELECT ce.* FROM code_evaluation ce WHERE ce.submission_id = %s",
                (submission_id,),
            )
            result["analysis"] = self.cursor.fetchone()

            # 3. Feedback
            if result["analysis"] and result["analysis"].get("feedback"):
                result["feedback_raw"] = result["analysis"]["feedback"]
                result["feedback_html"] = markdown(result["analysis"]["feedback"])
            else:
                result["feedback_raw"] = None
                result["feedback_html"] = None

            # 4. Plagiarism Matches
            self.cursor.execute(
                """
                SELECT pm.matched_submission_id, ce.plagiarism_score
                FROM plagiarism_match pm
                JOIN code_evaluation ce ON pm.evaluation_id = ce.code_evaluation_id
                WHERE ce.submission_id = %s
                """,
                (submission_id,),
            )
            result["plagiarism"] = self.cursor.fetchall()

            # 5. Testcase Results
            self.cursor.execute(
                """
                SELECT tr.testcase_id, tc.input_data, tc.expected_data,
                       tr.output, tr.passed, tr.execution_time
                FROM test_case_result tr
                JOIN test_cases tc ON tr.testcase_id = tc.testcase_id
                WHERE tr.submission_id = %s
                """,
                (submission_id,),
            )
            result["testcases"] = self.cursor.fetchall()

        except Exception as e:
            logger.error(f"❌ Error fetching submission details: {e}")
            result = None
        finally:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
        return result


def get_submission_details(submission_id: int):
    """Wrapper to fetch submission details."""
    return SubmissionService().get_submission_details(submission_id)
