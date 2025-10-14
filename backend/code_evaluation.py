import ast
import logging
import time
import subprocess
import tempfile
import os
from radon.complexity import cc_visit, cc_rank
import requests
import ast

logger = logging.getLogger("code_evaluation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(ch)


class CodeQualityEvaluator:
    """
    Evaluates code quality metrics such as cyclomatic complexity, length, and syntax validity.
    """

    def __init__(self, source_code: str, language: str = "python"):
        self.source_code = source_code
        self.language = language.lower()
        self.metrics = {
            "cyclomatic": None,
            "cyclomatic_rank": None,
            "length": None,
            "syntax_error": None,
            "analysis_time": None,
        }

    def check_syntax(self):
        """
        Syntax check for all supported languages.
        If syntax error found, self.metrics["syntax_error"] will contain a
        human-readable error message and further evaluation should stop.
        """
        import shutil

        start_time = time.perf_counter()
        self.metrics["syntax_error"] = None
        tmp_dir = None
        file_path = None

        try:
            # ---------------- Python ---------------- #
            if self.language in ("python", "python3"):
                try:
                    ast.parse(self.source_code)
                    logger.info("✅ Python syntax check passed")
                except SyntaxError as e:
                    # format detailed syntax error
                    error_msg = (
                        f"SyntaxError: {e.msg} (line {e.lineno}, column {e.offset})"
                    )
                    if e.text:
                        error_msg += f"\n→ {e.text.strip()}"
                    self.metrics["syntax_error"] = error_msg
                    logger.warning("❌ Python syntax error: %s", error_msg)

            # ---------------- C ---------------- #
            elif self.language == "c":
                file_path, tmp_dir = self._write_temp_file(".c")
                result = subprocess.run(
                    ["gcc", "-fsyntax-only", file_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    self.metrics["syntax_error"] = (
                        result.stderr.strip() or "C syntax error"
                    )
                    logger.warning(
                        "❌ C syntax error: %s", self.metrics["syntax_error"]
                    )
                else:
                    logger.info("✅ C syntax check passed")

            # ---------------- C++ ---------------- #
            elif self.language == "cpp":
                file_path, tmp_dir = self._write_temp_file(".cpp")
                result = subprocess.run(
                    ["g++", "-fsyntax-only", file_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    self.metrics["syntax_error"] = (
                        result.stderr.strip() or "C++ syntax error"
                    )
                    logger.warning(
                        "❌ C++ syntax error: %s", self.metrics["syntax_error"]
                    )
                else:
                    logger.info("✅ C++ syntax check passed")

            # ---------------- Java ---------------- #
            elif self.language == "java":
                file_path, tmp_dir = self._write_temp_file(".java")
                result = subprocess.run(
                    ["javac", "-Xlint", file_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    self.metrics["syntax_error"] = (
                        result.stderr.strip() or "Java syntax error"
                    )
                    logger.warning(
                        "❌ Java syntax error: %s", self.metrics["syntax_error"]
                    )
                else:
                    logger.info("✅ Java syntax check passed")

            # ---------------- Unsupported ---------------- #
            else:
                msg = f"⚠️ Syntax check not supported for language: {self.language}"
                self.metrics["syntax_error"] = msg
                logger.warning(msg)

        except subprocess.TimeoutExpired:
            self.metrics["syntax_error"] = f"Syntax check timed out for {self.language}"
            logger.error(self.metrics["syntax_error"])

        except Exception as e:
            self.metrics["syntax_error"] = f"Unexpected error during syntax check: {e}"
            logger.exception(self.metrics["syntax_error"])

        finally:
            # Cleanup temp directory safely
            if tmp_dir and os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                except Exception as e:
                    logger.debug("Temp dir cleanup failed: %s", e)

            self.metrics["analysis_time"] = round(time.perf_counter() - start_time, 3)

    def _load_source_code(self):
        """
        Loads source code from Cloudinary if a URL is provided.
        """
        if isinstance(self.source_code, str) and self.source_code.startswith(
            "https://res.cloudinary.com"
        ):
            try:
                response = requests.get(self.source_code)
                if response.status_code == 200:
                    self.source_code = response.text
                    logger.info("Loaded source code from Cloudinary URL")
                else:
                    logger.warning(
                        f"Failed to fetch code from Cloudinary: {response.status_code}"
                    )
            except Exception as e:
                logger.error(f"Error fetching code from Cloudinary: {e}")

    def _write_temp_file(self, suffix: str):
        """Helper to write code to a temp file for external tools."""
        tmp_dir = tempfile.mkdtemp()
        file_path = os.path.join(tmp_dir, f"temp{suffix}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.source_code)
        return file_path, tmp_dir

    def calculate_length(self):
        """Counts the number of non-empty lines in the code."""
        lines = [line for line in self.source_code.splitlines() if line.strip()]
        self.metrics["length"] = len(lines)
        logger.info("Code length: %s", self.metrics["length"])

    def calculate_cyclomatic_complexity(self):
        """
        Cyclomatic complexity calculation.
        Python -> radon
        C/C++ -> lizard
        Java -> lizard (basic support) or CK tool
        """
        if self.language in ("python", "python3"):
            try:
                blocks = cc_visit(self.source_code)
                if blocks:
                    avg_complexity = sum(b.complexity for b in blocks) / len(blocks)
                    rank = cc_rank(avg_complexity)
                    self.metrics["cyclomatic"] = round(avg_complexity, 2)
                    self.metrics["cyclomatic_rank"] = rank
                    logger.info(
                        "Python cyclomatic complexity: %s (Rank: %s)",
                        self.metrics["cyclomatic"],
                        self.metrics["cyclomatic_rank"],
                    )
            except Exception as e:
                logger.error("Error calculating Python complexity: %s", e)

        elif self.language in ["c", "cpp", "java"]:
            suffix = (
                ".c"
                if self.language == "c"
                else ".cpp" if self.language == "cpp" else ".java"
            )
            file_path, tmp_dir = self._write_temp_file(suffix)
            try:
                result = subprocess.run(
                    ["lizard", file_path],
                    capture_output=True,
                    text=True,
                )
                output = result.stdout.strip()
                if "Average" in output:
                    # crude parse: lizard prints average CC in last line
                    for line in output.splitlines():
                        if "Average" in line:
                            self.metrics["cyclomatic"] = float(line.split()[-1])
                            self.metrics["cyclomatic_rank"] = "N/A"
                            break
                logger.info(
                    "%s cyclomatic complexity: %s",
                    self.language.upper(),
                    self.metrics["cyclomatic"],
                )
            except Exception as e:
                logger.error(
                    "Error calculating %s cyclomatic complexity: %s",
                    self.language,
                    e,
                )
        else:
            logger.warning(
                "Cyclomatic complexity not supported for language=%s", self.language
            )

    def evaluate(self) -> dict:
        """
        Main method to run all evaluations and return metrics.
        """
        logger.info("Starting code quality evaluation for language=%s", self.language)
        self._load_source_code()
        start = time.time()

        self.check_syntax()
        self.calculate_length()
        self.calculate_cyclomatic_complexity()

        self.metrics["analysis_time"] = round(time.time() - start, 4)
        logger.info(
            "Code quality evaluation completed in %s seconds",
            self.metrics["analysis_time"],
        )

        return self.metrics


# ----------------- Public API -----------------
def evaluate_quality(source_code: str, language: str = "python") -> dict:
    """
    Functional wrapper to keep compatibility with pipeline.
    Returns a dict with code quality metrics.
    """
    evaluator = CodeQualityEvaluator(source_code, language)
    return evaluator.evaluate()


# Example usage
if __name__ == "__main__":
    sample_code_py = "def add(a,b):\n    return a+b"
    print("PYTHON:", evaluate_quality(sample_code_py, "python"))

    sample_code_c = "#include <stdio.h>\nint main(){return 0;}"
    print("C:", evaluate_quality(sample_code_c, "c"))

    sample_code_cpp = "#include <iostream>\nint main(){std::cout<<1;}"
    print("C++:", evaluate_quality(sample_code_cpp, "cpp"))

    sample_code_java = (
        "public class Test{public static void main(String[]a){System.out.println(1);}}"
    )
    print("JAVA:", evaluate_quality(sample_code_java, "java"))
