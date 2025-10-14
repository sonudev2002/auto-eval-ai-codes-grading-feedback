import logging
import traceback
from typing import Dict, Any
from openai import OpenAI
import sys, os

# Append parent directory to system path for module imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Config

# Configure logger for feedback generation process
logger = logging.getLogger("feedback")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(ch)


class FeedbackGenerator:
    """
    Handles automated generation of feedback for code submissions.
    Can use ChatGPT API or fallback to a rule-based feedback system.
    """

    def __init__(self, use_chatgpt: bool = True):
        self.use_chatgpt = use_chatgpt
        self.client: OpenAI | None = None

        # Check and initialize OpenAI API client
        if self.use_chatgpt and not Config.OPENAI_API_KEY:
            logger.warning(
                "OPENAI_API_KEY not found. Falling back to rule-based feedback."
            )
            self.use_chatgpt = False

        if self.use_chatgpt:
            try:
                self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.use_chatgpt = False

    def _format_prompt(
        self,
        execution: Dict[str, Any],
        quality: Dict[str, Any],
        plagiarism: Dict[str, Any],
        source_code: str,
    ) -> str:
        """
        Formats input data into a structured prompt for ChatGPT feedback generation.
        """
        return f"""
You are a teaching assistant providing constructive feedback on student code.

Execution:
- Status: {execution.get("status")}
- Output: {execution.get("output")}
- Error: {execution.get("error")}
- Time: {execution.get("time")} sec

Code Quality:
- Cyclomatic Complexity: {quality.get("cyclomatic")}
- Rank: {quality.get("cyclomatic_rank")}
- Length: {quality.get("length")} lines
- Syntax Error: {quality.get("syntax_error")}

Plagiarism:
- Score: {plagiarism.get("score")}
- Matched IDs: {plagiarism.get("matched_ids")}

Student Code:
{source_code}

Provide concise, encouraging feedback covering:
1. Strengths (correctness, clarity, creativity)
2. Weaknesses (syntax, logic, performance)
3. Suggestions for improvement
4. If plagiarism is high, remind about originality.
""".strip()

    def _chatgpt_feedback(self, prompt: str) -> str | None:
        """
        Generates feedback using the ChatGPT API.
        Returns None if an API error occurs.
        """
        if not self.client:
            return None

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful teaching assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.8,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"ChatGPT API error: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _rule_based_feedback(
        self,
        execution: Dict[str, Any],
        quality: Dict[str, Any],
        plagiarism: Dict[str, Any],
    ) -> str:
        """
        Generates simple feedback based on predefined rules when ChatGPT is unavailable.
        """
        feedback_parts = []

        # Execution analysis
        if execution.get("status") != "success":
            feedback_parts.append(
                f"Code execution failed: {execution.get('error', 'Unknown error')}."
            )
        else:
            feedback_parts.append("Code executed successfully.")

        # Syntax validation
        if quality.get("syntax_error"):
            feedback_parts.append(f"Syntax issue detected: {quality['syntax_error']}.")
        else:
            feedback_parts.append("No syntax errors found.")

        # Complexity evaluation
        cyclomatic = quality.get("cyclomatic")
        if cyclomatic:
            if cyclomatic > 10:
                feedback_parts.append(
                    "Code complexity is high; consider breaking functions into smaller units."
                )
            else:
                feedback_parts.append(
                    f"Cyclomatic complexity is {cyclomatic} ({quality.get('cyclomatic_rank')})."
                )

        # Code length assessment
        if quality.get("length", 0) > 100:
            feedback_parts.append(
                "Code is lengthy; try modularizing into smaller reusable parts."
            )

        # Plagiarism detection
        if plagiarism.get("score", 0) > 0.7:
            feedback_parts.append(
                "High similarity detected; ensure your submission is original."
            )

        return " ".join(feedback_parts)

    def generate_feedback(
        self,
        execution: Dict[str, Any],
        quality: Dict[str, Any],
        plagiarism: Dict[str, Any],
        source_code: str,
    ) -> str:
        """
        Main method to generate feedback using ChatGPT or fallback rule-based logic.
        """
        if self.use_chatgpt:
            prompt = self._format_prompt(execution, quality, plagiarism, source_code)
            feedback_text = self._chatgpt_feedback(prompt)
            if feedback_text:
                return feedback_text

        # Default to rule-based feedback if ChatGPT unavailable or fails
        return self._rule_based_feedback(execution, quality, plagiarism)


# ---------------- Public API ----------------
def generate_feedback(
    source_code: str,
    execution: Dict[str, Any],
    quality: Dict[str, Any],
    plagiarism: Dict[str, Any],
) -> str:
    """
    Public API function to generate feedback for code submissions.
    """
    generator = FeedbackGenerator()
    return generator.generate_feedback(execution, quality, plagiarism, source_code)
