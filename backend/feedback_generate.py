"""
feedback_generator.py
---------------------
Automated feedback generation system for code submissions.

This module evaluates a student's code by combining:
- Execution results (success, runtime errors, etc.)
- Code quality metrics (complexity, syntax, length)
- Plagiarism analysis (similarity score and matched IDs)

Based on configuration, it:
1. Uses OpenAI’s ChatGPT API (GPT-4) to generate personalized feedback, OR
2. Falls back to a rule-based static feedback system when the API is unavailable.

Outputs clear, constructive, and encouraging feedback messages.

Public API:
    generate_feedback(source_code, execution, quality, plagiarism)
"""

import logging
import traceback
from typing import Dict, Any
from openai import OpenAI
import sys, os

# Add project root to sys.path for proper imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Config

# ---------------- Logger Setup ----------------
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
    Generates automated feedback for code submissions.
    Uses ChatGPT API if available, else falls back to rule-based logic.
    """

    def __init__(self, use_chatgpt: bool = True):
        self.use_chatgpt = use_chatgpt
        self.client: OpenAI | None = None

        # Validate OpenAI API key
        if self.use_chatgpt and not Config.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not found. Using rule-based feedback.")
            self.use_chatgpt = False

        # Initialize OpenAI client if API key is available
        if self.use_chatgpt:
            try:
                self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.use_chatgpt = False

    # ---------------- Prompt Builder ----------------
    def _format_prompt(
        self,
        execution: Dict[str, Any],
        quality: Dict[str, Any],
        plagiarism: Dict[str, Any],
        source_code: str,
    ) -> str:
        """Format code metrics and results into a ChatGPT-ready feedback prompt."""
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

Provide concise feedback including:
1. Strengths (correctness, clarity)
2. Weaknesses (logic, performance)
3. Suggestions for improvement
4. Reminder if plagiarism is high.
""".strip()

    # ---------------- ChatGPT Feedback ----------------
    def _chatgpt_feedback(self, prompt: str) -> str | None:
        """Generate feedback using ChatGPT API."""
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

    # ---------------- Rule-Based Feedback ----------------
    def _rule_based_feedback(
        self,
        execution: Dict[str, Any],
        quality: Dict[str, Any],
        plagiarism: Dict[str, Any],
    ) -> str:
        """Fallback static feedback when ChatGPT is disabled or unavailable."""
        feedback_parts = []

        # Execution results
        if execution.get("status") != "success":
            feedback_parts.append(
                f"Code execution failed: {execution.get('error', 'Unknown error')}."
            )
        else:
            feedback_parts.append("Code executed successfully.")

        # Syntax check
        if quality.get("syntax_error"):
            feedback_parts.append(f"Syntax issue: {quality['syntax_error']}.")
        else:
            feedback_parts.append("No syntax errors found.")

        # Complexity analysis
        cyclomatic = quality.get("cyclomatic")
        if cyclomatic:
            if cyclomatic > 10:
                feedback_parts.append(
                    "Code complexity is high; consider refactoring into smaller functions."
                )
            else:
                feedback_parts.append(
                    f"Cyclomatic complexity: {cyclomatic} ({quality.get('cyclomatic_rank')})."
                )

        # Code length
        if quality.get("length", 0) > 100:
            feedback_parts.append(
                "Code is lengthy; consider breaking into smaller modules."
            )

        # Plagiarism score
        if plagiarism.get("score", 0) > 0.7:
            feedback_parts.append(
                "High similarity detected; ensure originality in your submission."
            )

        return " ".join(feedback_parts)

    # ---------------- Feedback Generator ----------------
    def generate_feedback(
        self,
        execution: Dict[str, Any],
        quality: Dict[str, Any],
        plagiarism: Dict[str, Any],
        source_code: str,
    ) -> str:
        """Main feedback generator: ChatGPT first, fallback to rule-based."""
        if self.use_chatgpt:
            prompt = self._format_prompt(execution, quality, plagiarism, source_code)
            feedback_text = self._chatgpt_feedback(prompt)
            if feedback_text:
                return feedback_text

        # Fallback if ChatGPT fails or disabled
        return self._rule_based_feedback(execution, quality, plagiarism)


# ---------------- Public API ----------------
def generate_feedback(
    source_code: str,
    execution: Dict[str, Any],
    quality: Dict[str, Any],
    plagiarism: Dict[str, Any],
) -> str:
    """Public function to generate code feedback."""
    generator = FeedbackGenerator()
    return generator.generate_feedback(execution, quality, plagiarism, source_code)
