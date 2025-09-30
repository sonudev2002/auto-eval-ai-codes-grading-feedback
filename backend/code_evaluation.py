"""
code_evaluation.py
===================

Responsible for evaluating code quality:
- Cyclomatic complexity
- Code length
- Syntax errors

Supports multiple languages:
- Python (AST + Radon)
- C (gcc + lizard)
- C++ (g++ + lizard/cppcheck)
- Java (javac + CK metrics)

Requires external tools:
- gcc, g++, javac
- lizard (pip install lizard)
- cppcheck (for C++)
- ck (Java metrics, optional)
"""

import ast
import logging
import time
import subprocess
import tempfile
import os
from radon.complexity import cc_visit, cc_rank

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

    def _write_temp_file(self, suffix: str):
        """Helper to write code to a temp file for external tools."""
        tmp_dir = tempfile.mkdtemp()
        file_path = os.path.join(tmp_dir, f"temp{suffix}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.source_code)
        return file_path, tmp_dir

    def check_syntax(self):
        """
        Syntax check for all supported languages.
        """
        if self.language in ("python", "python3"):
            try:
                ast.parse(self.source_code)
                self.metrics["syntax_error"] = None
                logger.info("Python syntax check passed")
            except SyntaxError as e:
                self.metrics["syntax_error"] = str(e)
                logger.warning("Python syntax error: %s", e)

        elif self.language == "c":
            file_path, tmp_dir = self._write_temp_file(".c")
            result = subprocess.run(
                ["gcc", "-fsyntax-only", file_path],
                capture_output=True,
                text=True,
            )
            self.metrics["syntax_error"] = (
                None if result.returncode == 0 else result.stderr.strip()
            )
            logger.info("C syntax check result: %s", self.metrics["syntax_error"])
            return

        elif self.language == "cpp":
            file_path, tmp_dir = self._write_temp_file(".cpp")
            result = subprocess.run(
                ["g++", "-fsyntax-only", file_path],
                capture_output=True,
                text=True,
            )
            self.metrics["syntax_error"] = (
                None if result.returncode == 0 else result.stderr.strip()
            )
            logger.info("C++ syntax check result: %s", self.metrics["syntax_error"])
            return

        elif self.language == "java":
            file_path, tmp_dir = self._write_temp_file(".java")
            result = subprocess.run(
                ["javac", "-Xlint", file_path],
                capture_output=True,
                text=True,
            )
            self.metrics["syntax_error"] = (
                None if result.returncode == 0 else result.stderr.strip()
            )
            logger.info("Java syntax check result: %s", self.metrics["syntax_error"])
            return

        else:
            logger.warning("Syntax check not supported for language=%s", self.language)

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
