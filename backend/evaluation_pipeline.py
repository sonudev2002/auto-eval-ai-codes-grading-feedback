import os
import subprocess
import logging
import uuid
import datetime
import time
import psutil

from db import get_connection
from code_evaluation import evaluate_quality
from feedback_generate import generate_feedback
from plagiarism_detector import check_plagiarism  # ✅ now actually used

logger = logging.getLogger("evaluation_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(ch)


class EvaluationPipeline:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        logger.info("EvaluationPipeline initialized")

    # --------------------- Run Testcases ---------------------
    def run_testcases(self, assignment_id, submission_id, code, language="python3"):
        """
        Fetch testcases from DB (Test_Cases),
        run code inside ONE Docker container for all testcases,
        store results in Test_Case_Result,
        return aggregated stats.
        """
        from code_submission import CodeRunner

        runner = None
        try:
            conn = get_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute(
                "SELECT testcase_id, input_data, expected_data FROM Test_Cases WHERE assignment_id = %s",
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

            runner = CodeRunner(language=language, timeout=self.timeout)
            runner.start_container(code)

            total, passed, failed = 0, 0, 0
            times, mems = [], []

            for t in testcases:
                total += 1
                tid = t["testcase_id"]
                input_data = t["input_data"] or ""
                expected = (t["expected_data"] or "").strip()

                start = time.perf_counter()
                stdout, stderr = runner.exec_testcase(input_data)  # unpack correctly
                exec_time = time.perf_counter() - start

                out_text = (stdout or "").strip()
                is_passed = False

                if stderr:
                    logger.info("Testcase %s: runtime error: %s", tid, stderr)
                    is_passed = False
                else:
                    if out_text == expected:
                        is_passed = True
                        passed += 1
                    else:
                        is_passed = False
                        failed += 1

                times.append(exec_time)
                mems.append(0)  # TODO: get real memory via docker stats

                output_to_store = out_text if out_text else (stderr or "")

                cursor.execute(
                    """INSERT INTO Test_Case_Result
                    (submission_id, testcase_id, output, passed, execution_time)
                    VALUES (%s, %s, %s, %s, %s)""",
                    (submission_id, tid, output_to_store, int(is_passed), exec_time),
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
            logger.exception("Error running testcases")
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "avg_time": 0.0,
                "memory_usage": 0,
            }
        finally:
            try:
                if runner:
                    runner.stop_container()
            except Exception:
                pass

    # --------------------- Score Computation ---------------------
    def compute_final_score(self, test_res, plagiarism, quality_metrics):
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
        if passed == 0:
            score = 0
        else:
            score = testcase_score + plagiarism_score + quality_score

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

    # --------------------- DB Save ---------------------
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
        try:
            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT COALESCE(MAX(version), 0) FROM Code_Submission WHERE user_id = %s AND assignment_id = %s",
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

            base_dir = os.path.join(os.path.dirname(__file__), "..", "submitted_codes")
            base_dir = os.path.abspath(base_dir)
            os.makedirs(base_dir, exist_ok=True)

            file_path = os.path.join(base_dir, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            rel_path = f"submitted_codes/{filename}".replace("\\", "/").lstrip("/")

            cursor.execute(
                """INSERT INTO Code_Submission
                (user_id, assignment_id, language, code_path, submitted_on, version)
                VALUES (%s, %s, %s, %s, NOW(), %s)""",
                (student_id, assignment_id, language, rel_path, new_version),
            )
            submission_id = cursor.lastrowid

            cursor.execute(
                """INSERT INTO Code_Evaluation
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

    # --------------------- Full Evaluation ---------------------
    def evaluate(
        self, assignment_id: int, student_id: int, code: str, language: str = "python"
    ):
        lang_map = {
            "python": "python3",
            "python3": "python3",
            "c": "c",
            "cpp": "cpp",
            "java": "java",
        }
        language = lang_map.get(language, language)

        quality_metrics = evaluate_quality(code, language)

        # First save submission with dummy values so submission_id exists
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

        # ✅ Now run plagiarism with real submission_id
        plagiarism = self.run_plagiarism_check(code, submission_id=submission_id)

        test_res = self.run_testcases(assignment_id, submission_id, code, language)

        grade, score = self.compute_final_score(test_res, plagiarism, quality_metrics)

        feedback = generate_feedback(
            code, {"tests": test_res}, quality_metrics, plagiarism
        )

        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE Code_Evaluation
                   SET feedback=%s, grade=%s, score=%s,
                       plagiarism_score=%s,
                       total_testcases=%s, passed_testcases=%s,
                       failed_testcases=%s, average_execution_time=%s, memory_usage=%s
                   WHERE submission_id=%s""",
                (
                    feedback,
                    grade,
                    score,
                    plagiarism.get("score", 0.0),
                    test_res["total"],
                    test_res["passed"],
                    test_res["failed"],
                    test_res["avg_time"],
                    test_res["memory_usage"],
                    submission_id,
                ),
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error("Error updating evaluation: %s", e)

        return {
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

    # --------------------- Plagiarism ---------------------
    def run_plagiarism_check(self, code: str, submission_id: int = None) -> dict:
        try:
            if submission_id is None:
                return {"score": 0.0, "matches": []}

            # ✅ Call Siamese model-based detector
            result = check_plagiarism(submission_id, code)

            return {
                "score": result.get("plagiarism_score", 0.0) / 100.0,  # normalize 0–1
                "matches": [
                    m["matched_submission_id"]
                    for m in result.get("matches", [])
                    if m["is_plagiarism"]
                ],
            }

        except Exception as e:
            logger.error("Plagiarism check error: %s", e)
            return {"score": 0.0, "matches": []}


if __name__ == "__main__":
    pipeline = EvaluationPipeline()
    test_code = "print('Hello from test code')"
    result = pipeline.evaluate(assignment_id=1, student_id=1, code=test_code)
    import json

    print(json.dumps(result, indent=2))
