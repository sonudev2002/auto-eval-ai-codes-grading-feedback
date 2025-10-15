"""
evaluation_pipeline.py
----------------------
Handles full automation of student code evaluation:
- Runs test cases in sandboxed containers (or simulates on Render)
- Checks plagiarism and code quality
- Generates AI feedback and grading
- Updates analytics, reports, and database records
"""

# ============================================================
# 🧩 Imports and Environment Setup
# ============================================================
import os
import subprocess
import logging
import uuid
import time
import psutil
import io
from datetime import datetime  # used for timestamps

# Internal module imports
from db import get_connection
from code_evaluation import evaluate_quality
from feedback_generate import generate_feedback
from plagiarism_detector import check_plagiarism

# Cloud integration
import cloudinary
import cloudinary.uploader
from config import Config

# ============================================================
# ☁️ Render Environment Detection
# ============================================================
# Render containers cannot run Docker; use simulation mode instead.
IS_RENDER = os.environ.get("RENDER") is not None

# ============================================================
# 🧾 Logger Setup
# ============================================================
logger = logging.getLogger("evaluation_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)


class EvaluationPipeline:
    """
    Orchestrates the full evaluation of a student's code submission.
    Includes:
      - Testcase execution (Docker or simulated on Render)
      - Scoring and grading
      - Cloud upload and database storage
    """

    def __init__(self, timeout: int = 100):
        """Initialize pipeline with an execution timeout (default 100s)."""
        self.timeout = timeout
        logger.info("EvaluationPipeline initialized")

    # --------------------- 🧪 Run Test Cases ---------------------
    def run_testcases(self, assignment_id, submission_id, code, language="python3"):
        """
        Run all test cases for a submission.
        On Render → simulation mode (no Docker).
        Locally → executes inside Docker via CodeRunner.
        """

        from code_submission import CodeRunner

        runner = None

        try:
            conn = get_connection()
            cursor = conn.cursor(dictionary=True)

            # Fetch testcases for assignment
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

            # Initialize CodeRunner
            runner = CodeRunner(language=language, timeout=self.timeout)

            # Simulate results if running on Render
            if IS_RENDER or not runner.client:

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

            # Execute testcases inside Docker
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

                # Store result in DB
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
            # Render-safe fallback result
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

    # --------------------- 🧮 Compute Final Score & Grade---------------------
    def compute_final_score(self, test_res, plagiarism, quality_metrics):
        """Calculate weighted score and assign grade (A–F)."""
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

        # Grade mapping
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

    # --------------------- 💾 Save Evaluation Data to Database ---------------------
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
        """
        Store submission, evaluation results, and code in DB.
        Handles:
          - Cloudinary upload (fallback to local)
          - Safe versioning and timestamp formatting
          - Inserts into code_submission + code_evaluation
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # ✅ Determine next version safely
            cursor.execute(
                "SELECT COALESCE(MAX(version), 0) FROM code_submission WHERE user_id = %s AND assignment_id = %s",
                (student_id, assignment_id),
            )
            latest_version = cursor.fetchone()[0] or 0
            new_version = latest_version + 1

            # ✅ Generate timestamp safely (prevents strftime() bug)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # ✅ Determine proper file extension
            extension = {
                "python3": ".py",
                "c": ".c",
                "cpp": ".cpp",
                "java": ".java",
            }.get(language, ".txt")

            filename = (
                f"{student_id}_{assignment_id}_v{new_version}_{timestamp}{extension}"
            )

            # ✅ Cloudinary upload (with fallback to local)
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
                logger.info(f"✅ Code uploaded to Cloudinary: {rel_path}")
            except Exception as e:
                logger.warning(
                    f"⚠️ Cloudinary upload failed: {e}. Saving locally instead."
                )
                base_dir = os.path.join(
                    os.path.dirname(__file__), "..", "submitted_codes"
                )
                os.makedirs(base_dir, exist_ok=True)
                file_path = os.path.join(base_dir, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
                rel_path = f"/submitted_codes/{filename}"
                logger.info(f"✅ Code saved locally at {rel_path}")

            # ✅ Insert into code_submission
            cursor.execute(
                """
                INSERT INTO code_submission
                (user_id, assignment_id, language, code_path, submitted_on, version)
                VALUES (%s, %s, %s, %s, NOW(), %s)
                """,
                (student_id, assignment_id, language, rel_path, new_version),
            )
            submission_id = cursor.lastrowid or 0  # ensure numeric safety

            # ✅ Insert into code_evaluation
            cursor.execute(
                """
                INSERT INTO code_evaluation
                (submission_id, feedback, grade, score, plagiarism_score,
                has_syntax_error, code_quality_score, code_length,
                cyclomatic_complexity, total_testcases, passed_testcases,
                failed_testcases, average_execution_time, memory_usage)
                VALUES (%s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s)
                """,
                (
                    submission_id,
                    feedback,
                    grade,
                    float(score or 0.0),
                    float(plagiarism.get("score", 0.0)),
                    bool(quality_metrics.get("syntax_error")),
                    float(min(score, 100.0)),
                    int(quality_metrics.get("length") or 0),
                    int(quality_metrics.get("cyclomatic") or 0),
                    int(test_res.get("total", 0)),
                    int(test_res.get("passed", 0)),
                    int(test_res.get("failed", 0)),
                    float(test_res.get("avg_time", 0.0)),
                    float(test_res.get("memory_usage", 0.0)),
                ),
            )

            conn.commit()
            cursor.close()
            conn.close()

            # ✅ Confirm success
            if submission_id > 0:
                logger.info(
                    "✅ Results saved successfully for submission_id=%s", submission_id
                )
            else:
                logger.error("⚠️ No submission_id returned after DB insert!")

            return submission_id

        except Exception as e:
            logger.exception("🔥 Error saving to DB: %s", e)
            # even if it fails, always return a valid int
            return 0

    # --------------------- Full Evaluation Process ---------------------
    def evaluate(
        self, assignment_id: int, student_id: int, code: str, language: str = "python"
    ):
        """Main entry point for full code evaluation workflow.

        Includes syntax check, plagiarism detection, test execution, grading,
        feedback generation, and real-time updates to analytics tables.
        """

        logger.info(
            f"Starting evaluation for student={student_id}, assignment={assignment_id}, Render={IS_RENDER}"
        )

        # 🔹 Normalize language aliases
        lang_map = {
            "python": "python3",
            "python3": "python3",
            "c": "c",
            "cpp": "cpp",
            "java": "java",
        }
        language = lang_map.get(language, language)

        # =======================================================
        # STEP 1: Evaluate code quality + syntax check
        # =======================================================
        quality_metrics = evaluate_quality(code, language)
        syntax_err = bool(quality_metrics.get("syntax_error"))

        # ---------------- Syntax Error: Short-circuit early ----------------
        if syntax_err:
            logger.warning(
                f"Syntax error detected for student={student_id}, assignment={assignment_id}"
            )

            # Minimal test result placeholder
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

            # Create basic feedback + assignment status even for syntax errors
            try:
                conn_se = get_connection()
                cur_se = conn_se.cursor()
                cur_se.execute(
                    """
                    INSERT INTO feedback_score (submission_id, feedback_score)
                    VALUES (%s, 1)
                    ON DUPLICATE KEY UPDATE feedback_score = VALUES(feedback_score)
                    """,
                    (submission_id,),
                )
                cur_se.execute(
                    """
                    INSERT INTO student_assignment_status (user_id, assignment_id, status, submitted_at, updated_at)
                    VALUES (%s, %s, 'submitted', NOW(), NOW())
                    ON DUPLICATE KEY UPDATE
                        status = VALUES(status),
                        updated_at = NOW()
                    """,
                    (student_id, assignment_id),
                )
                conn_se.commit()
                cur_se.close()
                conn_se.close()
            except Exception:
                logger.exception(
                    "⚠️ Failed to insert feedback/assignment status for syntax error"
                )

            # Return response
            return {
                "status": "error",
                "stage": "syntax_check",
                "message": "Syntax error in submitted code",
                "details": quality_metrics.get("syntax_error"),
                "submission_id": submission_id,
                "assignment_id": assignment_id,
                "student_id": student_id,
            }

        # =======================================================
        # STEP 2: Save initial placeholder submission
        # =======================================================
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

        # =======================================================
        # STEP 3: Run plagiarism detection
        # =======================================================
        plagiarism = self.run_plagiarism_check(code, submission_id=submission_id)

        # =======================================================
        # STEP 4: Execute testcases (simulated on Render)
        # =======================================================
        test_res = self.run_testcases(assignment_id, submission_id, code, language)

        # =======================================================
        # STEP 5: Compute final grade and score
        # =======================================================
        grade, score = self.compute_final_score(test_res, plagiarism, quality_metrics)

        # =======================================================
        # STEP 6: Generate AI feedback
        # =======================================================
        feedback = generate_feedback(
            code, {"tests": test_res}, quality_metrics, plagiarism
        )

        # =======================================================
        # STEP 7: Save final evaluation
        # =======================================================
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
            logger.exception("❌ Failed to update final evaluation: %s", e)

        # =======================================================
        # STEP 8: Post-Evaluation Housekeeping (feedback, status, analytics)
        # =======================================================
        import threading

        def post_evaluation_tasks():
            """Handles feedback_score, student_assignment_status, and analytics update."""
            try:
                if not submission_id or submission_id <= 0:
                    logger.error(
                        "⚠️ post_evaluation_tasks() called with invalid submission_id."
                    )
                    return

                conn_post = get_connection()
                cur_post = conn_post.cursor()

                # 1️⃣ Insert feedback_score
                fb_val = 3
                if isinstance(feedback, dict) and "score" in feedback:
                    try:
                        fb_val = int(round(float(feedback["score"])))
                    except Exception:
                        fb_val = 3
                fb_val = max(0, min(5, fb_val))  # ensure 0-5 range

                cur_post.execute(
                    """
                    INSERT INTO feedback_score (submission_id, feedback_score)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE feedback_score = VALUES(feedback_score)
                    """,
                    (submission_id, fb_val),
                )

                # 2️⃣ Update student_assignment_status → graded
                cur_post.execute(
                    "SELECT due_date FROM assignment WHERE assignment_id=%s",
                    (assignment_id,),
                )
                due_row = cur_post.fetchone()

                if not due_row or not due_row[0]:
                    status_value = "Pending Submission"
                else:
                    due_date = due_row[0]
                    if datetime.now() > due_date:
                        status_value = "Submitted Late"
                    else:
                        status_value = "Submitted On Time"

                cur_post.execute(
                    """
                    INSERT INTO student_assignment_status (user_id, assignment_id, status, submitted_at, updated_at)
                    VALUES (%s, %s, %s, NOW(), NOW())
                    ON DUPLICATE KEY UPDATE
                    status = VALUES(status),
                    updated_at = NOW()
                    """,
                    (student_id, assignment_id, status_value),
                )

                conn_post.commit()
                cur_post.close()
                conn_post.close()
                logger.info(
                    f"✅ Feedback & assignment status updated for submission {submission_id}"
                )

                # 3️⃣ Analytics refresh (instructor + student)
                try:
                    from analytics import (
                        StudentDifficultyAnalytics,
                        InstructorDifficultyAnalytics,
                        student_performance_analytics,
                        instructor_performance_analytics,
                    )

                    # Student updates
                    StudentDifficultyAnalytics().update_user_stats(student_id)
                    student_performance_analytics.update_user(student_id)

                    # Instructor updates
                    conn_ai = get_connection()
                    cur_ai = conn_ai.cursor(dictionary=True)
                    cur_ai.execute(
                        "SELECT instructor_id FROM assignment WHERE assignment_id=%s",
                        (assignment_id,),
                    )
                    row = cur_ai.fetchone()
                    cur_ai.close()
                    conn_ai.close()

                    if row and row.get("instructor_id"):
                        instr_id = row["instructor_id"]
                        InstructorDifficultyAnalytics().update_user_stats(instr_id)
                        instructor_performance_analytics.update_user(instr_id)
                        logger.info(
                            f"✅ Instructor analytics updated for instructor {instr_id}"
                        )

                except Exception:
                    logger.exception("⚠️ Analytics refresh failed (non-fatal)")

            except Exception:
                logger.exception("⚠️ Post-evaluation housekeeping failed")

        # Run background thread for housekeeping
        try:
            threading.Thread(target=post_evaluation_tasks, daemon=True).start()
        except Exception:
            logger.exception("⚠️ Failed to start background analytics thread")

        # =======================================================
        # STEP 9: Return final response
        # =======================================================
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
