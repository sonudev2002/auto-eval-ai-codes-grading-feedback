from evaluation_pipeline import EvaluationPipeline
from grade_distribution import GradeDistributionManager
from assignment_management import get_assignment_details
from db import get_connection
from markdown import markdown
import os
import uuid
import docker
import logging
import re
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
        self.language = language
        self.timeout = timeout
        self.client = docker.from_env()
        self.container = None
        self.tempfile = None

        self.language_images = {
            "python3": "python:3.10-slim",
            "c": "gcc:latest",
            "cpp": "gcc:latest",
            "java": "openjdk:11",
        }

    # ---------------- Utility ----------------
    def _write_tempfile(self, source_code: str, extension: str) -> str:
        base_dir = os.path.join(os.getcwd(), "docker_temp")
        os.makedirs(base_dir, exist_ok=True)
        filename = os.path.join(base_dir, f"{uuid.uuid4().hex}{extension}")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(source_code)
        return filename

    def _select_image_and_cmd(self, filename: str) -> Tuple[str, list]:
        container_file = f"/app/{os.path.basename(filename)}"
        if self.language == "python3":
            return self.language_images["python3"], ["python", container_file]
        elif self.language == "c":
            return self.language_images["c"], ["gcc", container_file, "-o", "/app/main"]
        elif self.language == "cpp":
            return self.language_images["cpp"], [
                "g++",
                container_file,
                "-o",
                "/app/main",
            ]
        elif self.language == "java":
            return self.language_images["java"], ["javac", container_file]
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    @staticmethod
    def _clean_output(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)  # strip control chars
        text = text.replace("jTraceback", "Traceback").replace("0    ", "    ")
        return text.strip()

    @staticmethod
    def _normalize_input(stdin_data: Any) -> str:
        if stdin_data is None:
            return ""
        if isinstance(stdin_data, str):
            return stdin_data
        if isinstance(stdin_data, (list, tuple, set)):
            return " ".join(map(str, stdin_data))
        return str(stdin_data)

    # ---------------- Persistent Container ----------------
    def start_container(self, source_code: str):
        lang_map = {
            "python": "python3",
            "python3": "python3",
            "c": "c",
            "cpp": "cpp",
            "java": "java",
        }
        self.language = lang_map.get(self.language, self.language)

        ext_map = {"python3": ".py", "c": ".c", "cpp": ".cpp", "java": ".java"}
        ext = ext_map.get(self.language, ".txt")
        self.tempfile = self._write_tempfile(source_code, ext)

        image, _ = self._select_image_and_cmd(self.tempfile)
        self.container = self.client.containers.run(
            image=image,
            command="sleep 60",
            volumes={os.path.dirname(self.tempfile): {"bind": "/app", "mode": "rw"}},
            working_dir="/app",
            stdin_open=True,
            stdout=True,
            stderr=True,
            detach=True,
        )

        # Compile if needed
        if self.language == "c":
            build_cmd = f"gcc /app/{os.path.basename(self.tempfile)} -o /app/main"
        elif self.language == "cpp":
            build_cmd = f"g++ /app/{os.path.basename(self.tempfile)} -o /app/main"
        elif self.language == "java":
            build_cmd = f"javac /app/{os.path.basename(self.tempfile)}"
        else:
            build_cmd = None

        if build_cmd:
            exit_code, logs = self.container.exec_run(build_cmd)
            if exit_code != 0:
                raise RuntimeError(
                    f"Compilation failed: {logs.decode(errors='ignore')}"
                )

    def exec_testcase(self, input_data: Any) -> Tuple[str, str]:
        if self.language == "python3":
            run_cmd = ["python", f"/app/{os.path.basename(self.tempfile)}"]
        elif self.language in ("c", "cpp"):
            run_cmd = ["/app/main"]
        elif self.language == "java":
            classname = os.path.basename(self.tempfile).replace(".java", "")
            run_cmd = ["java", "-cp", "/app", classname]
        else:
            raise ValueError("Unsupported language")

        exit_code, stdout, stderr = self._exec_with_input(
            self.container, run_cmd, stdin_data=input_data, timeout=self.timeout
        )

        stdout = self._clean_output(stdout)
        stderr = self._clean_output(stderr)

        if exit_code == 0:
            return stdout, ""
        else:
            return "", f"Error (exit {exit_code}): {stderr or stdout}"

    @staticmethod
    def _exec_with_input(
        container, cmd, stdin_data: Optional[Any] = None, timeout: int = 5
    ):
        try:
            exec_id = container.client.api.exec_create(
                container.id, cmd, stdin=True, tty=False
            )
            sock = container.client.api.exec_start(
                exec_id, detach=False, tty=False, stream=False, socket=True
            )

            if stdin_data:
                stdin_str = CodeRunner._normalize_input(stdin_data)
                try:
                    sock._sock.send(stdin_str.encode())
                    sock._sock.shutdown(1)
                except Exception:
                    pass

            chunks = []
            while True:
                try:
                    data = sock.recv(4096)
                    if not data:
                        break
                    chunks.append(data)
                except Exception:
                    break

            combined = b"".join(chunks).decode(errors="ignore")
            inspect = container.client.api.exec_inspect(exec_id)
            exit_code = inspect.get("ExitCode", 1)
            return exit_code, combined, ""
        except Exception as e:
            return 1, "", f"Exec failed: {e}"

    def run_multiple_inputs(self, inputs: List[Any]) -> List[Dict[str, str]]:
        results = []
        try:
            for inp in inputs:
                stdout, stderr = self.exec_testcase(inp)
                results.append({"stdout": stdout, "stderr": stderr})
            return results
        finally:
            self.stop_container()

    def stop_container(self):
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


def submit_code(assignment_id, student_id, source_code, language="python3") -> Dict:
    logger.info("submit_code called")
    pipeline = EvaluationPipeline(timeout=10)
    result = pipeline.evaluate(
        assignment_id=assignment_id,
        student_id=student_id,
        code=source_code,
        language=language,
    )

    # --- Grade Distribution Updates ---
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
        logger.error(f"Grade distribution update failed: {e}")

    # ✅ Plagiarism logging removed here (handled in evaluation_pipeline)
    return result


# -----------------------------------------------------------#


class SubmissionService:
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
                FROM Code_Submission cs
                JOIN Assignment a ON cs.assignment_id = a.assignment_id
                WHERE cs.submission_id = %s
                """,
                (submission_id,),
            )
            result["submission"] = self.cursor.fetchone()

            # 2. Code Evaluation
            self.cursor.execute(
                "SELECT ce.* FROM Code_Evaluation ce WHERE ce.submission_id = %s",
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
                FROM Plagiarism_match pm
                JOIN Code_Evaluation ce ON pm.evaluation_id = ce.code_evaluation_id
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
                FROM Test_Case_Result tr
                JOIN Test_Cases tc ON tr.testcase_id = tc.testcase_id
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
    return SubmissionService().get_submission_details(submission_id)


if __name__ == "__main__":
    full_result = get_submission_details(53)
    print(full_result)
