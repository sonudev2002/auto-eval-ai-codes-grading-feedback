from db import get_connection
from datetime import datetime
import logging
from flask import request
from contextlib import contextmanager
from backend.notification_system import NotificationSystem
import csv
from io import TextIOWrapper

# instantiate once
notifier = NotificationSystem(use_background=False)


logger = logging.getLogger(__name__)


# test case creation
def create_testcases(assignment_id, input_data, expected_data):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
                    INSERT INTO test_cases(assignment_id, input_data, expected_data)
                    VALUES (%s , %s, %s)
                    
                    """,
            (assignment_id, input_data, expected_data),
        )
        conn.commit()
        return True
    except Exception as e:
        print("Error in testcase creation: ", e)
        return None

    finally:
        if "conn" in locals():
            conn.close()


# example creation
def create_example(assignment_id, description):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
                    INSERT INTO example(assignment_id, description)
                    VALUES (%s , %s)
                    
                    """,
            (assignment_id, description),
        )
        conn.commit()
        return True
    except Exception as e:
        print("Error in example creation: ", e)
        return None

    finally:
        if "conn" in locals():
            conn.close()


#  assignemnt creation
def create_assignment(
    title, description, hint, instructor_id, difficulty_level, due_date, repository_id
):
    try:
        created_at = datetime.now()
        conn = get_connection()
        cursor = conn.cursor()
        # ✅ Remove RETURNING and just insert the data
        cursor.execute(
            """
            INSERT INTO assignment 
            (title, description, hint, instructor_id, difficulty_level, due_date, created_date, repository_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                title,
                description,
                hint,
                instructor_id,
                difficulty_level,
                due_date,
                created_at,
                repository_id,
            ),
        )
        # ✅ Get the inserted assignment_id
        assignment_id = cursor.lastrowid
        conn.commit()
        return assignment_id
    except Exception as e:
        print("Error in assignment creation: ", e)
        return None
    finally:
        if "conn" in locals():
            conn.close()


def upload_assignment(form_data, csv_file=None):
    try:
        # --- Step 1: Validate and extract repository ID ---
        repo_raw = form_data.get("repository_id", [None])[0]
        if not repo_raw or not repo_raw.isdigit():
            print("Upload Assignment Error: Invalid repository ID:", repo_raw)
            return False
        repo_id = int(repo_raw)

        # --- Step 2: Extract assignment core fields ---
        title = form_data.get("title", [""])[0].strip()
        description = form_data.get("description", [""])[0].strip()
        hint = form_data.get("hint", [""])[0].strip()
        difficulty = form_data.get("difficulty", [""])[0].strip()
        due_date = form_data.get("due_date", [""])[0].strip()
        instructor_id = int(form_data.get("instructor_id"))

        if not (repo_id and title and description and due_date):
            print("Upload Assignment Error: Missing required fields")
            return False

        # --- Step 3: Create assignment ---
        assignment_id = create_assignment(
            title, description, hint, instructor_id, difficulty, due_date, repo_id
        )
        if not assignment_id:
            return False

        # --- Step 4: Add examples ---
        for desc in form_data.get("examples[]", []):
            if desc.strip():
                create_example(assignment_id, desc.strip())

        # --- Step 5A: Manual test cases ---
        inputs = form_data.get("testcase_input[]", [])
        outputs = form_data.get("testcase_expected[]", [])
        for inp, exp in zip(inputs, outputs):
            if inp.strip() and exp.strip():
                create_testcases(assignment_id, inp.strip(), exp.strip())

        # --- Step 5B: CSV test cases with validation ---
        if csv_file:

            reader = csv.DictReader(TextIOWrapper(csv_file, encoding="utf-8"))

            # ✅ Validate required headers
            required_headers = {"input_data", "expected_data"}
            missing = required_headers - set((reader.fieldnames or []))
            if missing:
                return {
                    "success": False,
                    "message": f"CSV missing required headers: {', '.join(missing)}",
                }

            # ✅ Process rows
            for row in reader:
                inp = (row.get("input_data") or "").strip()
                exp = (row.get("expected_data") or "").strip()
                if inp and exp:
                    create_testcases(assignment_id, inp, exp)

        # --- Step 6: Notify ---
        notifier.notify_assignment_uploaded(
            assignment_id=int(assignment_id),
            title=title,
            due_date=due_date,
            instructor_id=instructor_id,
            recipient_keys=["students"],
        )
        return True

    except Exception as e:
        print("❌ Upload Assignment Error:", e)
        return False


def update_assignment_backend(form_data):
    """
    Update an assignment’s fields based on provided JSON data using a single DB transaction.

    form_data keys:
      - assignment_id (int)
      - description (str, optional)
      - hint (str, optional)
      - due_date (YYYY-MM-DD, optional)
      - examples (list[str], optional)
      - test_cases (list[dict], optional)
    Returns True on success, False on error.
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        assignment_id = int(form_data.get("assignment_id"))
        updated_fields = []

        # 1. Update description
        if desc := form_data.get("description", "").strip():
            cursor.execute(
                "UPDATE assignment SET description = %s WHERE assignment_id = %s",
                (desc, assignment_id),
            )
            updated_fields.append("description")

        # 2. Update hint
        if hint := form_data.get("hint", "").strip():
            cursor.execute(
                "UPDATE assignment SET hint = %s WHERE assignment_id = %s",
                (hint, assignment_id),
            )
            updated_fields.append("hint")

        # 3. Update due date
        if due := form_data.get("due_date", "").strip():
            cursor.execute(
                "UPDATE assignment SET due_date = %s WHERE assignment_id = %s",
                (due, assignment_id),
            )
            updated_fields.append("due_date")

        # 4. Replace examples in one transaction
        examples = form_data.get("examples") or []
        if examples:
            cursor.execute(
                "DELETE FROM example WHERE assignment_id = %s", (assignment_id,)
            )
            for ex in examples:
                ex = ex.strip()
                if ex:
                    cursor.execute(
                        "INSERT INTO example (assignment_id, description) VALUES (%s, %s)",
                        (assignment_id, ex),
                    )
            updated_fields.append("examples")

        # 5. Replace test cases in one transaction
        test_cases = form_data.get("test_cases") or []
        if test_cases:
            cursor.execute(
                "DELETE FROM test_cases WHERE assignment_id = %s", (assignment_id,)
            )
            for tc in test_cases:
                inp = tc.get("input", "").strip()
                exp = tc.get("expected", "").strip()
                if inp and exp:
                    cursor.execute(
                        "INSERT INTO test_cases (assignment_id, input_data, expected_data) VALUES (%s, %s, %s)",
                        (assignment_id, inp, exp),
                    )
            updated_fields.append("test_cases")

        # If nothing selected, nothing to update
        if not updated_fields:
            return False

        conn.commit()
        return True

    except Exception as e:
        if conn:
            conn.rollback()
        print("❌ Error updating assignment:", e)
        return False

    finally:
        if conn:
            conn.close()


def get_all_repositories():
    try:

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT repository_id, repo_title FROM assignment_repository")
        rows = cursor.fetchall()

        result = [{"repository_id": r[0], "repo_title": r[1]} for r in rows]

        return result

    except Exception as e:
        print("❌ Error fetching repositories:", e)
        return []

    finally:
        if "conn" in locals():
            conn.close()


def get_assignments_by_repo(repo_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT assignment_id, title FROM assignment
            WHERE repository_id = %s
        """,
            (repo_id,),
        )
        rows = cursor.fetchall()

        return [{"assignment_id": row[0], "title": row[1]} for row in rows]

    except Exception as e:
        print("Error fetching assignments:", e)
        return []

    finally:
        if "conn" in locals():
            conn.close()


def get_assignment_details(assignment_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Get main assignment details
        cursor.execute(
            """
            SELECT a.assignment_id, a.title, a.description, a.hint, a.due_date, a.repository_id, a.instructor_id
            FROM assignment a
            WHERE a.assignment_id = %s
        """,
            (assignment_id,),
        )
        row = cursor.fetchone()

        if not row:
            return {}

        # Get examples
        cursor.execute(
            "SELECT description FROM example WHERE assignment_id = %s", (assignment_id,)
        )
        examples = [r[0] for r in cursor.fetchall()]

        # Get test cases
        cursor.execute(
            "SELECT input_data, expected_data FROM test_cases WHERE assignment_id = %s",
            (assignment_id,),
        )
        test_cases = [{"input": r[0], "expected": r[1]} for r in cursor.fetchall()]

        return {
            "assignment_id": row[0],
            "title": row[1],
            "description": row[2],
            "hint": row[3],
            "due_date": row[4].strftime("%Y-%m-%d") if row[4] else "",
            "repository_id": row[5],
            "examples": examples,
            "test_cases": test_cases,
            "instructor_id": row[6],
        }

    except Exception as e:
        print("Error fetching assignment details:", e)
        return {}

    finally:
        if "conn" in locals():
            conn.close()


def delete_assignment(assignment_id):
    """
    Remove an assignment and all its examples & test cases.
    Returns True on success, False on any error.
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 1. Remove test cases
        cursor.execute(
            "DELETE FROM test_cases WHERE assignment_id = %s", (assignment_id,)
        )
        # 2. Remove examples
        cursor.execute("DELETE FROM example WHERE assignment_id = %s", (assignment_id,))
        # 3. Remove assignment
        cursor.execute(
            "DELETE FROM assignment WHERE assignment_id = %s", (assignment_id,)
        )

        conn.commit()
        return True

    except Exception as e:
        if conn:
            conn.rollback()
        print("❌ Error deleting assignment:", e)
        return False

    finally:
        if conn:
            conn.close()


def create_repository(repo_title, user_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Check for existing repository with same title (case-insensitive)
        cursor.execute(
            """
            SELECT 1 FROM assignment_repository
            WHERE LOWER(repo_title) = LOWER(%s)
        """,
            (repo_title.strip(),),
        )
        existing = cursor.fetchone()

        if existing:
            return False, "Repository with this title already exists."

        created_at = datetime.now()

        cursor.execute(
            """
            INSERT INTO assignment_repository (repo_title, created_by, created_date)
            VALUES (%s, %s, %s)
        """,
            (repo_title.strip(), user_id, created_at),
        )

        conn.commit()
        return True, "Repository created successfully."

    except Exception as e:
        print("❌ Repository creation error:", e)
        if conn:
            conn.rollback()
        return False, "Database error occurred."

    finally:
        if conn:
            conn.close()


def delete_repository_by_id(repo_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Step 1: Get all assignment IDs under this repository
        cursor.execute(
            "SELECT assignment_id FROM assignment WHERE repository_id = %s", (repo_id,)
        )
        assignment_ids = [row[0] for row in cursor.fetchall()]

        for assignment_id in assignment_ids:
            # Get all submission_ids for this assignment
            cursor.execute(
                "SELECT submission_id FROM code_submission WHERE assignment_id = %s",
                (assignment_id,),
            )
            submission_ids = [row[0] for row in cursor.fetchall()]

            for submission_id in submission_ids:
                # Delete test case results
                cursor.execute(
                    "DELETE FROM test_case_result WHERE submission_id = %s",
                    (submission_id,),
                )
                # Delete feedback score
                cursor.execute(
                    "DELETE FROM feedback_score WHERE submission_id = %s",
                    (submission_id,),
                )
                # Delete code evaluation
                cursor.execute(
                    "DELETE FROM code_evaluation WHERE submission_id = %s",
                    (submission_id,),
                )
                # Delete plagiarism matches
                cursor.execute(
                    "DELETE FROM plagiarism_match WHERE matched_submission_id = %s OR evaluation_id IN (SELECT code_evaluation_id FROM code_evaluation WHERE submission_id = %s)",
                    (submission_id, submission_id),
                )

            # Delete code submissions
            cursor.execute(
                "DELETE FROM code_submission WHERE assignment_id = %s", (assignment_id,)
            )
            # Delete test cases
            cursor.execute(
                "DELETE FROM test_cases WHERE assignment_id = %s", (assignment_id,)
            )
            # Delete examples
            cursor.execute(
                "DELETE FROM example WHERE assignment_id = %s", (assignment_id,)
            )
            # Delete analytics
            cursor.execute(
                "DELETE FROM assignment_analytics WHERE assignment_id = %s",
                (assignment_id,),
            )
            # Delete assignment itself
            cursor.execute(
                "DELETE FROM assignment WHERE assignment_id = %s", (assignment_id,)
            )

        # Finally, delete the repository
        cursor.execute(
            "DELETE FROM assignment_repository WHERE repository_id = %s", (repo_id,)
        )
        conn.commit()

        return True, "Repository and all related assignments and data deleted."

    except Exception as e:
        print("❌ Error during full repository delete:", e)
        if conn:
            conn.rollback()
        return False, "Error occurred while deleting repository."

    finally:
        if conn:
            conn.close()


class AssignmentsStudent:

    @staticmethod
    @contextmanager
    def _get_cursor():
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            yield cursor
        except Exception as e:
            logger.exception("Database error: %s", e)
            raise
        finally:
            if conn:
                conn.close()

    @staticmethod
    def _row_to_dict(row, keys):
        """Helper: Convert tuple row to dict with given keys"""
        return dict(zip(keys, row))

    @staticmethod
    def get_dashboard_assignment_detail():
        query = """
            SELECT 
                a.repository_id, 
                a.assignment_id, 
                a.title, 
                a.instructor_id, 
                u.first_name AS instructor_name, 
                d.difficulty_types AS difficulty_name,
                a.due_date
            FROM Assignment a
            JOIN User_Profile u ON a.instructor_id = u.user_id
            JOIN Difficulty_Level d ON a.difficulty_level = d.level_id
        """
        keys = [
            "repository_id",
            "assignment_id",
            "title",
            "instructor_id",
            "instructor_name",
            "difficulty_level",
            "due_date",
        ]
        try:
            with AssignmentsStudent._get_cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                return [AssignmentsStudent._row_to_dict(row, keys) for row in rows]

        except Exception:
            logger.error("Failed to fetch dashboard assignments")
            return []

    @staticmethod
    def get_repository_details():
        query = """
            SELECT 
                r.repository_id,
                r.repo_title,
                u.first_name AS created_by_name,
                COUNT(a.assignment_id) AS question_count
            FROM assignment_repository r
            JOIN user_profile u ON r.created_by = u.user_id
            LEFT JOIN assignment a ON r.repository_id = a.repository_id
            GROUP BY r.repository_id, r.repo_title, u.first_name
        """
        keys = ["repository_id", "title", "created_by_name", "question_count"]

        try:
            with AssignmentsStudent._get_cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                return [AssignmentsStudent._row_to_dict(row, keys) for row in rows]
        except Exception:
            logger.error("Failed to fetch repository details")
            return []

    @staticmethod
    def get_assignments_by_repo_detailed(repo_id):
        query = """
            SELECT 
                a.repository_id, 
                a.assignment_id, 
                a.title, 
                a.instructor_id, 
                u.first_name AS instructor_name, 
                d.difficulty_types AS difficulty_name, 
                a.due_date
            FROM Assignment a
            JOIN User_Profile u ON a.instructor_id = u.user_id
            JOIN Difficulty_Level d ON a.difficulty_level = d.level_id
            WHERE a.repository_id = %s
            ORDER BY a.due_date ASC
        """
        keys = [
            "repository_id",
            "assignment_id",
            "title",
            "instructor_id",
            "instructor_name",
            "difficulty_level",
            "due_date",
        ]

        try:
            with AssignmentsStudent._get_cursor() as cursor:
                cursor.execute(query, (repo_id,))
                rows = cursor.fetchall()
                return [AssignmentsStudent._row_to_dict(row, keys) for row in rows]
        except Exception:
            logger.error("Failed to fetch assignments for repo_id=%s", repo_id)
            return []


class Code_editor:

    @staticmethod
    @contextmanager
    def _get_cursor():
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            yield cursor
            # commit only if caller made changes (we keep it safe/read-only here)
            # conn.commit()
        except Exception as e:
            logger.exception("Database error: %s", e)
            raise
        finally:
            if conn:
                conn.close()

    @staticmethod
    def _rows_to_dict_list(cursor, rows):
        """
        Convert cursor.description + rows (tuples) to list of dicts.
        """
        cols = [col[0] for col in cursor.description] if cursor.description else []
        result = []
        for r in rows:
            # map each row tuple to dict keyed by column names
            result.append({cols[i]: r[i] for i in range(len(cols))})
        return result

    @staticmethod
    def assignment_detail_by_id(ass_id, user_id=None, role=None):
        """
        Return assignment details + past submissions.
        If role=student → only their submissions.
        If role=admin → all submissions.
        """
        try:
            with Code_editor._get_cursor() as cursor:
                # 1) Core assignment + difficulty + repository + instructor
                cursor.execute(
                    """
                    SELECT
                        a.assignment_id,
                        a.title,
                        a.description,
                        a.hint,
                        a.due_date,
                        u.user_id   AS instructor_user_id,
                        u.first_name AS instructor_first_name,
                        u.last_name  AS instructor_last_name,
                        dl.level_id  AS difficulty_level_id,
                        dl.difficulty_types,
                        dl.marks,
                        ar.repository_id,
                        ar.repo_title
                    FROM Assignment a
                    LEFT JOIN Difficulty_Level dl ON a.difficulty_level = dl.level_id
                    LEFT JOIN Assignment_Repository ar ON a.repository_id = ar.repository_id
                    LEFT JOIN User_Profile u ON a.instructor_id = u.user_id
                    WHERE a.assignment_id = %s
                    """,
                    (ass_id,),
                )
                row = cursor.fetchone()
                if not row:
                    logger.info(
                        "assignment_detail_by_id: assignment %s not found", ass_id
                    )
                    return None

                base = Code_editor._rows_to_dict_list(cursor, [row])[0]

                result = {
                    "title": base.get("title"),
                    "description": base.get("description"),
                    "hint": base.get("hint"),
                    "due_date": base.get("due_date"),
                    "instructor": None,
                    "difficulty": None,
                    "repository": None,
                    "examples": [],
                    "testcases": [],
                    "past_submissions": [],
                }

                # instructor
                if base.get("instructor_user_id") is not None:
                    first = base.get("instructor_first_name") or ""
                    last = base.get("instructor_last_name") or ""
                    result["instructor"] = {
                        "user_id": base.get("instructor_user_id"),
                        "first_name": first,
                        "last_name": last,
                        "full_name": (first + " " + last).strip(),
                    }

                # difficulty
                if base.get("difficulty_level_id") is not None:
                    result["difficulty"] = {
                        "level_id": base.get("difficulty_level_id"),
                        "difficulty_types": base.get("difficulty_types"),
                        "marks": base.get("marks"),
                    }

                # repository
                if base.get("repository_id") is not None:
                    result["repository"] = {
                        "repository_id": base.get("repository_id"),
                        "repo_title": base.get("repo_title"),
                    }

                # 2) Examples
                cursor.execute(
                    """
                    SELECT example_id, description
                    FROM Example
                    WHERE assignment_id = %s
                    ORDER BY example_id ASC
                    """,
                    (ass_id,),
                )
                examples_rows = cursor.fetchall()
                examples = []
                if examples_rows:
                    for example_id, raw_desc in examples_rows:
                        parts = [p.strip(" {}") for p in raw_desc.split("},")]
                        ex = {
                            "example_id": example_id,
                            "input": parts[0] if len(parts) > 0 else "",
                            "output": parts[1] if len(parts) > 1 else "",
                            "explanation": parts[2] if len(parts) > 2 else "",
                        }
                        examples.append(ex)
                result["examples"] = examples

                # 3) Test cases
                cursor.execute(
                    """
                    SELECT testcase_id, input_data, expected_data
                    FROM Test_Cases
                    WHERE assignment_id = %s
                    ORDER BY testcase_id ASC
                    """,
                    (ass_id,),
                )
                tc_rows = cursor.fetchall()
                result["testcases"] = (
                    Code_editor._rows_to_dict_list(cursor, tc_rows) if tc_rows else []
                )

                # 4) Past submissions
                if role in ("admin", "instructor"):

                    cursor.execute(
                        """
                        SELECT submission_id, user_id, submitted_on, language, code_path
                        FROM Code_Submission
                        WHERE assignment_id = %s
                        ORDER BY submitted_on DESC
                        """,
                        (ass_id,),
                    )
                else:  # student (or fallback)
                    cursor.execute(
                        """
                        SELECT submission_id, user_id, submitted_on, language, code_path
                        FROM Code_Submission
                        WHERE assignment_id = %s AND user_id = %s
                        ORDER BY submitted_on DESC
                        """,
                        (ass_id, user_id),
                    )

                sub_rows = cursor.fetchall()
                result["past_submissions"] = (
                    Code_editor._rows_to_dict_list(cursor, sub_rows) if sub_rows else []
                )

                if result["due_date"]:
                    result["due_date"] = result["due_date"].strftime("%Y-%m-%d")

                return result

        except Exception as exc:
            logger.exception(
                "Failed to fetch assignment details for id %s: %s", ass_id, exc
            )
            raise
