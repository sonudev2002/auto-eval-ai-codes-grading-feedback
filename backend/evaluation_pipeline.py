import os
import subprocess
import logging
import uuid
import datetime
import time
import psutil
import io

from db import get_connection
from code_evaluation import evaluate_quality
from feedback_generate import generate_feedback
from plagiarism_detector import check_plagiarism
import cloudinary
import cloudinary.uploader
from config import Config

# ---------------------------------------------------------
# ✅ Detect if the system is running on Render
# Render containers cannot access Docker, so tests are simulated.
# ---------------------------------------------------------
IS_RENDER = os.environ.get("RENDER") is not None

logger = logging.getLogger("evaluation_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(ch)


class EvaluationPipeline:
    """
    Handles the full process of evaluating a student's code submission.
    On Render, skips Docker-based execution for stability.
    """

    def __init__(self, timeout: int = 100):
        self.timeout = timeout
        logger.info("EvaluationPipeline initialized")

    # --------------------- Run Testcases ---------------------
    def run_testcases(self, assignment_id, submission_id, code, language="python3"):
        """
        Fetches testcases from the database and executes them
        inside a Docker container (or simulates results on Render).
        """
        from code_submission import CodeRunner

        runner = None

        try:
            conn = get_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute(
                "SELECT testcase_id, input_data, expected_data FROM test_cases WHERE assignment_id = %s",
                (assignment_id,),
            )
            testcases = cursor.fetchall()

            if not testcases:
                logger.warning("No testcases found for assignment_id=%s", assignment_id)
                return {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "avg_time": 0.0,
                    "memory_usage": 0,
                }

            # ✅ Initialize CodeRunner and handle Render case
            runner = CodeRunner(language=language, timeout=self.timeout)

            if IS_RENDER or not runner.client:
                # Skip Docker-based runs on Render
                logger.warning(
                    "Render environment detected — skipping Docker testcase execution."
                )
                return {
                    "total": len(testcases),
                    "passed": 0,
                    "failed": len(testcases),
                    "avg_time": 0.0,
                    "memory_usage": 0,
                }

            # Run inside Docker (local only)
            runner.start_container(code)
            total, passed, failed = 0, 0, 0
            times, mems = [], []

            for t in testcases:
                total += 1
                tid = t["testcase_id"]
                input_data = t["input_data"] or ""
                expected = (t["expected_data"] or "").strip()

                start = time.perf_counter()
                stdout, stderr = runner.exec_testcase(input_data)
                exec_time = time.perf_counter() - start

                out_text = (stdout or "").strip()
                is_passed = out_text == expected and not stderr
                if is_passed:
                    passed += 1
                else:
                    failed += 1

                times.append(exec_time)
                mems.append(0)

                cursor.execute(
                    """INSERT INTO test_case_result
                       (submission_id, testcase_id, output, passed, execution_time)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (submission_id, tid, out_text or stderr, int(is_passed), exec_time),
                )

            avg_time = sum(times) / len(times) if times else 0.0
            avg_mem = sum(mems) / len(mems) if mems else 0

            conn.commit()
            cursor.close()
            conn.close()

            return {
                "total": total,
                "passed": passed,
                "failed": failed,
                "avg_time": avg_time,
                "memory_usage": avg_mem,
            }

        except Exception as e:
            logger.exception("Error running testcases: %s", e)
            # ✅ Render-safe fallback result
            if IS_RENDER:
                return {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "avg_time": 0.0,
                    "memory_usage": 0,
                }
            else:
                raise
        finally:
            try:
                if runner:
                    runner.stop_container()
            except Exception:
                pass

    # --------------------- Score Computation ---------------------
    def compute_final_score(self, test_res, plagiarism, quality_metrics):
        """Computes weighted final score and grade."""
        total, passed = test_res["total"], test_res["passed"]
        avg_time, memory_usage = test_res["avg_time"], test_res["memory_usage"]

        cyclomatic = int(quality_metrics.get("cyclomatic") or 0)
        length = int(quality_metrics.get("length") or 0)

        testcase_score = (passed / total) * 80 if total > 0 else 0
        plagiarism_score = max(0, (1 - plagiarism.get("score", 0.0))) * 5

        complexity_factor = min(1.0, 10 / (cyclomatic + 1))
        length_factor = min(1.0, 200 / (length + 1))
        time_factor = min(1.0, 1 / (avg_time + 0.1))
        mem_factor = min(1.0, 50000 / (memory_usage + 1))

        quality_factor = (
            complexity_factor + length_factor + time_factor + mem_factor
        ) / 4
        quality_score = quality_factor * 15

        score = testcase_score + plagiarism_score + quality_score if passed > 0 else 0

        if score >= 90:
            grade = "A"
        elif score >= 75:
            grade = "B"
        elif score >= 60:
            grade = "C"
        elif score >= 50:
            grade = "D"
        elif score >= 40:
            grade = "E"
        else:
            grade = "F"

        return grade, score

    # --------------------- Save Submission to DB ---------------------
    def save_to_db(
        self,
        assignment_id,
        student_id,
        code,
        plagiarism,
        feedback,
        quality_metrics,
        test_res,
        grade,
        score,
        language="python3",
    ):
        """Stores code submission and initial evaluation data."""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT COALESCE(MAX(version), 0) FROM code_submission WHERE user_id = %s AND assignment_id = %s",
                (student_id, assignment_id),
            )
            latest_version = cursor.fetchone()[0]
            new_version = latest_version + 1

            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            extension = {
                "python3": ".py",
                "c": ".c",
                "cpp": ".cpp",
                "java": ".java",
            }.get(language, ".txt")
            filename = (
                f"{student_id}_{assignment_id}_v{new_version}_{timestamp}{extension}"
            )

            # ✅ Try saving to Cloudinary, fallback to local
            rel_path = None
            try:
                file_bytes = io.BytesIO(code.encode("utf-8"))
                upload_result = cloudinary.uploader.upload(
                    file_bytes,
                    resource_type="raw",
                    folder="auto-eval/submitted_codes",
                    public_id=filename,
                    use_filename=True,
                    unique_filename=True,
                    overwrite=False,
                )
                rel_path = upload_result.get("secure_url")
                logger.info(f"Code uploaded to Cloudinary: {rel_path}")
            except Exception as e:
                logger.warning(f"Cloudinary upload failed: {e}")
                base_dir = os.path.join(
                    os.path.dirname(__file__), "..", "submitted_codes"
                )
                os.makedirs(base_dir, exist_ok=True)
                file_path = os.path.join(base_dir, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
                rel_path = f"/submitted_codes/{filename}"
                logger.info(f"Code saved locally at {rel_path}")

            cursor.execute(
                """INSERT INTO code_submission
                   (user_id, assignment_id, language, code_path, submitted_on, version)
                   VALUES (%s, %s, %s, %s, NOW(), %s)""",
                (student_id, assignment_id, language, rel_path, new_version),
            )
            submission_id = cursor.lastrowid

            cursor.execute(
                """INSERT INTO code_evaluation
                   (submission_id, feedback, grade, score, plagiarism_score,
                    has_syntax_error, code_quality_score, code_length,
                    cyclomatic_complexity, total_testcases, passed_testcases,
                    failed_testcases, average_execution_time, memory_usage)
                   VALUES (%s, %s, %s, %s, %s,
                           %s, %s, %s,
                           %s, %s, %s,
                           %s, %s, %s)""",
                (
                    submission_id,
                    feedback,
                    grade,
                    score,
                    plagiarism.get("score", 0.0),
                    True if quality_metrics.get("syntax_error") else False,
                    min(score, 100),
                    int(quality_metrics.get("length") or 0),
                    int(quality_metrics.get("cyclomatic") or 0),
                    test_res["total"],
                    test_res["passed"],
                    test_res["failed"],
                    test_res["avg_time"],
                    test_res["memory_usage"],
                ),
            )

            conn.commit()
            cursor.close()
            conn.close()
            logger.info("Results saved for submission_id=%s", submission_id)
            return submission_id

        except Exception as e:
            logger.error("Error saving to DB: %s", e)
            return None

    # --------------------- Full Evaluation Process ---------------------
    def evaluate(
        self, assignment_id: int, student_id: int, code: str, language: str = "python"
    ):
        """Main entry point for code evaluation workflow.

        Short-circuits immediately when syntax error is detected in quality check.
        """
        logger.info(
            f"Starting evaluation for student={student_id}, assignment={assignment_id}, Render={IS_RENDER}"
        )

        lang_map = {
            "python": "python3",
            "python3": "python3",
            "c": "c",
            "cpp": "cpp",
            "java": "java",
        }
        language = lang_map.get(language, language)

        # -------------------------
        # Step 1: Evaluate code quality (includes syntax check)
        # -------------------------
        quality_metrics = evaluate_quality(code, language)

        # Ensure evaluate_quality always returns a dict and sets "syntax_error" key when applicable.
        syntax_err = bool(quality_metrics.get("syntax_error"))
        if syntax_err:
            # -------------------------
            # If syntax error: save minimal submission and return early
            # -------------------------
            # Save an initial submission record marking syntax error (so students see their submission)
            initial_test_res = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "avg_time": 0.0,
                "memory_usage": 0,
            }

            submission_id = self.save_to_db(
                assignment_id=assignment_id,
                student_id=student_id,
                code=code,
                plagiarism={"score": 0.0, "matches": []},
                feedback=f"Syntax Error: {quality_metrics.get('syntax_error')}",
                quality_metrics=quality_metrics,
                test_res=initial_test_res,
                grade="F",
                score=0,
                language=language,
            )

            # Return a short-circuit response the frontend can use
            return {
                "status": "error",
                "stage": "syntax_check",
                "message": "Syntax error in submitted code",
                "details": quality_metrics.get("syntax_error"),
                "submission_id": submission_id,
                "assignment_id": assignment_id,
                "student_id": student_id,
            }

        # -------------------------
        # Normal path: no syntax error — continue evaluation
        # -------------------------

        # Step 2: Save initial record (pre-evaluation placeholder so we have a submission_id)
        submission_id = self.save_to_db(
            assignment_id,
            student_id,
            code,
            plagiarism={"score": 0.0, "matches": []},
            feedback="",
            quality_metrics=quality_metrics,
            test_res={
                "total": 0,
                "passed": 0,
                "failed": 0,
                "avg_time": 0.0,
                "memory_usage": 0,
            },
            grade="F",
            score=0,
            language=language,
        )

        # Step 3: Run plagiarism detection
        plagiarism = self.run_plagiarism_check(code, submission_id=submission_id)

        # Step 4: Run testcases (skipped on Render)
        test_res = self.run_testcases(assignment_id, submission_id, code, language)

        # Step 5: Compute grade and score
        grade, score = self.compute_final_score(test_res, plagiarism, quality_metrics)

        # Step 6: Generate AI feedback
        feedback = generate_feedback(
            code, {"tests": test_res}, quality_metrics, plagiarism
        )

        # Step 7: Save final evaluation info (update code_evaluation row)
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE code_evaluation
                SET feedback=%s, grade=%s, score=%s, plagiarism_score=%s,
                    has_syntax_error=%s, code_quality_score=%s,
                    code_length=%s, cyclomatic_complexity=%s,
                    total_testcases=%s, passed_testcases=%s, failed_testcases=%s,
                    average_execution_time=%s, memory_usage=%s
                WHERE submission_id=%s
                """,
                (
                    feedback,
                    grade,
                    score,
                    plagiarism.get("score", 0.0),
                    False,  # no syntax error here
                    min(score, 100),
                    int(quality_metrics.get("length") or 0),
                    int(quality_metrics.get("cyclomatic") or 0),
                    test_res["total"],
                    test_res["passed"],
                    test_res["failed"],
                    test_res["avg_time"],
                    test_res["memory_usage"],
                    submission_id,
                ),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.exception("Failed updating final evaluation row: %s", e)

        # Final response
        return {
            "status": "success",
            "stage": "evaluation_complete",
            "assignment_id": assignment_id,
            "student_id": student_id,
            "language": language,
            "test_results": test_res,
            "quality_metrics": quality_metrics,
            "plagiarism": plagiarism,
            "feedback": feedback,
            "score": score,
            "grade": grade,
            "submission_id": submission_id,
        }

    # --------------------- Plagiarism Check ---------------------
    def run_plagiarism_check(self, code: str, submission_id: int = None) -> dict:
        """Runs plagiarism check and normalizes results."""
        try:
            if submission_id is None:
                return {"score": 0.0, "matches": []}
            result = check_plagiarism(submission_id, code)
            return {
                "score": result.get("plagiarism_score", 0.0) / 100.0,
                "matches": [
                    m["matched_submission_id"]
                    for m in result.get("matches", [])
                    if m["is_plagiarism"]
                ],
            }
        except Exception as e:
            logger.error("Plagiarism check error: %s", e)
            return {"score": 0.0, "matches": []}
