import os
import uuid
from typing import Dict, Any, List, cast
from werkzeug.utils import secure_filename
from backend.db import get_connection
import cloudinary
import cloudinary.uploader
from config import Config


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
            """INSERT INTO reported_issue (user_id, issue_type, description)
               VALUES (%s, %s, %s)""",
            (user_id, issue_type, description),
        )
        issue_id = cursor.lastrowid

        # Save multiple screenshots
        if screenshot_files:
            for screenshot_file in screenshot_files:
                if screenshot_file and screenshot_file.filename.strip() != "":
                    if not allowed_file(screenshot_file.filename):
                        return {
                            "success": False,
                            "message": f"Invalid file type: {screenshot_file.filename}",
                        }

                    url = None
                    filename = secure_filename(screenshot_file.filename)

                    # 1️⃣ Try Cloudinary first
                    if Config.CLOUDINARY_ENABLED:
                        try:
                            upload_result = cloudinary.uploader.upload(
                                screenshot_file,
                                folder="auto-eval/screenshots",
                                use_filename=True,
                                unique_filename=True,
                                overwrite=False,
                            )
                            url = upload_result.get("secure_url")
                        except Exception as e:
                            print(f"[WARN] Cloudinary upload failed: {e}")

                    # 2️⃣ Fallback to local
                    if not url:
                        try:
                            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                            unique_name = f"{uuid.uuid4().hex}_{filename}"
                            save_path = os.path.join(UPLOAD_FOLDER, unique_name)
                            screenshot_file.save(save_path)
                            # convert to web path (so Flask can serve)
                            url = f"/uploads/screenshots/{unique_name}".replace(
                                "\\", "/"
                            )
                        except Exception as e:
                            print(f"[ERROR] Local screenshot save failed: {e}")
                            return {
                                "success": False,
                                "message": f"Failed to save {filename}",
                            }

                    # 3️⃣ Save URL to DB
                    cursor.execute(
                        """INSERT INTO screenshots (issue_id, screenshot_path) VALUES (%s, %s)""",
                        (issue_id, url),
                    )

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
               FROM reported_issue i
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
               FROM screenshots
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
            """SELECT i.issue_id, 
       CONCAT(u.first_name, ' ', u.last_name) AS reporter, 
       i.issue_type, 
       i.description, 
       i.status, 
       i.reported_at, 
       i.resolved_at,
       GROUP_CONCAT(s.screenshot_path) AS screenshots
FROM reported_issue i
JOIN user_profile u ON i.user_id = u.user_id
LEFT JOIN screenshots s ON i.issue_id = s.issue_id
GROUP BY i.issue_id
ORDER BY i.reported_at DESC
"""
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
            """UPDATE reported_issue
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
