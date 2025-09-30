import os
import uuid
from typing import Dict, Any, List, cast
from werkzeug.utils import secure_filename
from backend.db import get_connection

# ----------------------------
# Config
# ----------------------------
UPLOAD_FOLDER = "uploads/screenshots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ----------------------------
# 1. Submit a New Issue
# ----------------------------
def submit_issue(
    user_id: int, issue_type: str, description: str, screenshot_files=None
) -> Dict[str, Any]:
    conn, cursor = None, None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Insert issue
        cursor.execute(
            """INSERT INTO Reported_Issue (user_id, issue_type, description)
               VALUES (%s, %s, %s)""",
            (user_id, issue_type, description),
        )
        issue_id = cursor.lastrowid

        # Save multiple screenshots
        if screenshot_files:
            for screenshot_file in screenshot_files:
                if screenshot_file and screenshot_file.filename.strip() != "":
                    if allowed_file(screenshot_file.filename):
                        filename = f"{uuid.uuid4().hex}_{secure_filename(screenshot_file.filename)}"
                        filepath = os.path.join(UPLOAD_FOLDER, filename)
                        screenshot_file.save(filepath)

                        cursor.execute(
                            """INSERT INTO Screenshots (issue_id, screenshot_path)
                               VALUES (%s, %s)""",
                            (issue_id, filepath),
                        )
                    else:
                        return {
                            "success": False,
                            "message": f"Invalid file type: {screenshot_file.filename}",
                        }

        conn.commit()
        return {"success": True, "issue_id": issue_id}

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[ERROR] submit_issue: {e}")
        return {"success": False, "message": str(e)}

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ----------------------------
# 2. Get Issues for a User (Student/Instructor)
# ----------------------------
def get_user_issues(user_id: int) -> List[Dict[str, Any]]:
    conn, cursor = None, None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """SELECT i.issue_id, i.issue_type, i.description, i.status, 
                      i.reported_at, i.resolved_at
               FROM Reported_Issue i
               WHERE i.user_id = %s
               ORDER BY i.reported_at DESC""",
            (user_id,),
        )

        issues = cast(List[Dict[str, Any]], cursor.fetchall())
        return issues

    except Exception as e:
        print(f"[ERROR] get_user_issues: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_screenshots(issue_id: int) -> List[str]:
    conn, cursor = None, None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """SELECT screenshot_path 
               FROM Screenshots
               WHERE issue_id = %s""",
            (issue_id,),
        )
        rows = cursor.fetchall()
        return [row[0] for row in rows]  # returns list of file paths

    except Exception as e:
        print(f"[ERROR] get_screenshots: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ----------------------------
# 3. Get All Issues (Admin View)
# ----------------------------
def get_all_issues() -> List[Dict[str, Any]]:
    conn, cursor = None, None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """SELECT i.issue_id, u.name AS reporter, i.issue_type, 
                      i.description, i.status, i.reported_at, i.resolved_at,
                      GROUP_CONCAT(s.screenshot_path) AS screenshots
               FROM Reported_Issue i
               JOIN User_Profile u ON i.user_id = u.user_id
               LEFT JOIN Screenshots s ON i.issue_id = s.issue_id
               GROUP BY i.issue_id
               ORDER BY i.reported_at DESC"""
        )

        issues = cast(List[Dict[str, Any]], cursor.fetchall())
        return issues

    except Exception as e:
        print(f"[ERROR] get_all_issues: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ----------------------------
# 4. Resolve an Issue (Admin)
# ----------------------------
def resolve_issue(issue_id: int) -> Dict[str, Any]:
    conn, cursor = None, None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """UPDATE Reported_Issue
               SET status = 'resolved', resolved_at = NOW()
               WHERE issue_id = %s""",
            (issue_id,),
        )
        conn.commit()
        return {"success": True}

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[ERROR] resolve_issue: {e}")
        return {"success": False, "message": str(e)}

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
