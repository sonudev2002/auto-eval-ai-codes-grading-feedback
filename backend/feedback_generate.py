import logging
import traceback
from typing import Dict, Any
from openai import OpenAI
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Config


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
    Generates feedback for code submissions.
    """

    def __init__(self, use_chatgpt: bool = True):
        self.use_chatgpt = use_chatgpt
        self.client: OpenAI | None = None

        if self.use_chatgpt and not Config.OPENAI_API_KEY:
            logger.warning(
                "OPENAI_API_KEY not found in config. Falling back to rule-based feedback."
            )
            self.use_chatgpt = False

        if self.use_chatgpt:
            try:
                # Initialize client with explicit key from config
                self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            except Exception as e:
                logger.error("Failed to initialize OpenAI client: %s", e)
                self.use_chatgpt = False

    def _format_prompt(
        self,
        execution: Dict[str, Any],
        quality: Dict[str, Any],
        plagiarism: Dict[str, Any],
        source_code: str,
    ) -> str:
        """
        Creates a structured prompt for ChatGPT.
        """

        prompt = f"""
                You are a teaching assistant giving constructive, supportive, and educational feedback on student code.
Analyze the submission based on the following details:

Execution:

Status: {execution.get("status")}

Output: {execution.get("output")}

Error: {execution.get("error")}

Time: {execution.get("time")} sec

Code Quality:

Cyclomatic Complexity: {quality.get("cyclomatic")}

Complexity Rank: {quality.get("cyclomatic_rank")}

Code Length (lines): {quality.get("length")}

Syntax Error: {quality.get("syntax_error")}

Plagiarism:

Score: {plagiarism.get("score")}

Matched IDs: {plagiarism.get("matched_ids")}

Student Code:
{source_code}

Now provide feedback to the student in a clear, friendly, and constructive tone:

Highlight what works well (correctness, logic, clarity, structure, creativity).

Identify issues or errors (syntax, runtime, inefficiency, unnecessary complexity).

Suggest actionable improvements (readability, modularity, efficiency, robustness).

If plagiarism score is high, include a gentle reminder about the importance of writing original code and learning from references responsibly.

Keep the feedback concise, encouraging, and focused on helping the student understand and improve.
"""

        return prompt.strip()

    def _chatgpt_feedback(self, prompt: str) -> str | None:
        """
        Calls OpenAI API to generate feedback (new API).
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
                max_tokens=100,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("ChatGPT API error: %s", e)
            logger.debug(traceback.format_exc())
            return None

    def _rule_based_feedback(
        self,
        execution: Dict[str, Any],
        quality: Dict[str, Any],
        plagiarism: Dict[str, Any],
    ) -> str:
        """
        Simple fallback feedback without ChatGPT.
        """
        parts = []

        # Execution feedback
        if execution.get("status") != "success":
            parts.append(
                f"Your code failed to execute correctly: {execution.get('error', 'Unknown error')}."
            )
        else:
            parts.append("Your code executed successfully.")

        # Syntax feedback
        if quality.get("syntax_error"):
            parts.append(f"Fix the syntax error: {quality['syntax_error']}.")
        else:
            parts.append("No syntax issues detected.")

        # Complexity feedback
        if quality.get("cyclomatic") and quality["cyclomatic"] > 10:
            parts.append(
                "Your code is complex. Consider breaking functions into smaller ones."
            )
        elif quality.get("cyclomatic"):
            parts.append(
                f"Cyclomatic complexity is {quality['cyclomatic']} ({quality.get('cyclomatic_rank')})."
            )

        # Length feedback
        if quality.get("length") and quality["length"] > 100:
            parts.append(
                "Your code is long. Try refactoring into smaller, reusable functions."
            )

        # Plagiarism feedback
        if plagiarism.get("score") and plagiarism["score"] > 0.7:
            parts.append(
                "Warning: Your code is very similar to others. Please submit original work."
            )

        return " ".join(parts)

    def generate_feedback(
        self,
        execution: Dict[str, Any],
        quality: Dict[str, Any],
        plagiarism: Dict[str, Any],
        source_code: str,
    ) -> str:
        """
        Main entry point: returns feedback as text.
        """

        if self.use_chatgpt:

            prompt = self._format_prompt(execution, quality, plagiarism, source_code)

            feedback_text = self._chatgpt_feedback(prompt)

            if feedback_text:
                return feedback_text

        # Fallback: rule-based feedback
        return self._rule_based_feedback(execution, quality, plagiarism)


# ---------------- Public API ----------------
def generate_feedback(
    source_code: str,
    execution: Dict[str, Any],
    quality: Dict[str, Any],
    plagiarism: Dict[str, Any],
) -> str:

    generator = FeedbackGenerator()

    result = generator.generate_feedback(
        execution=execution,
        quality=quality,
        plagiarism=plagiarism,
        source_code=source_code,
    )

    return result
