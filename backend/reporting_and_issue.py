"""
reporting_and_issue.py
----------------------
Handles issue reporting, screenshot uploads, retrieval, and administrative actions
for resolving and updating user-reported issues.
"""

import os
import uuid
from typing import Dict, Any, List, cast
from werkzeug.utils import secure_filename
from backend.db import get_connection
import cloudinary
import cloudinary.uploader
from config import Config

# ----------------------------
# Configuration
# ----------------------------
UPLOAD_FOLDER = "uploads/screenshots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has a valid extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ----------------------------
# 1. Submit a New Issue
# ----------------------------
def submit_issue(
    user_id: int, issue_type: str, description: str, screenshot_files=None
) -> Dict[str, Any]:
    """
    Create a new reported issue entry.
    Supports multiple optional screenshot uploads (Cloudinary or local fallback).
    """
    conn, cursor = None, None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Insert main issue record
        cursor.execute(
            """
            INSERT INTO reported_issue (user_id, issue_type, description)
            VALUES (%s, %s, %s)
            """,
            (user_id, issue_type, description),
        )
        issue_id = cursor.lastrowid

        # Process screenshots if any
        if screenshot_files:
            for screenshot_file in screenshot_files:
                if not screenshot_file or not screenshot_file.filename.strip():
                    continue

                if not allowed_file(screenshot_file.filename):
                    return {
                        "success": False,
                        "message": f"Invalid file type: {screenshot_file.filename}",
                    }

                filename = secure_filename(screenshot_file.filename)
                url = None

                # 1️⃣ Attempt Cloudinary upload first
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

                # 2️⃣ Fallback to local storage
                if not url:
                    try:
                        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                        unique_name = f"{uuid.uuid4().hex}_{filename}"
                        save_path = os.path.join(UPLOAD_FOLDER, unique_name)
                        screenshot_file.save(save_path)
                        url = f"/uploads/screenshots/{unique_name}".replace("\\", "/")
                    except Exception as e:
                        print(f"[ERROR] Local save failed: {e}")
                        return {
                            "success": False,
                            "message": f"Failed to save {filename}",
                        }

                # 3️⃣ Save screenshot URL in database
                cursor.execute(
                    """
                    INSERT INTO screenshots (issue_id, screenshot_path)
                    VALUES (%s, %s)
                    """,
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
# 2. Retrieve Issues (User-Specific)
# ----------------------------
def get_user_issues(user_id: int) -> List[Dict[str, Any]]:
    """Fetch all reported issues for a given user."""
    conn, cursor = None, None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT i.issue_id, i.issue_type, i.description, i.status,
                   i.reported_at, i.resolved_at
            FROM reported_issue i
            WHERE i.user_id = %s
            ORDER BY i.reported_at DESC
            """,
            (user_id,),
        )
        return cast(List[Dict[str, Any]], cursor.fetchall())

    except Exception as e:
        print(f"[ERROR] get_user_issues: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_screenshots(issue_id: int) -> List[str]:
    """Return all screenshot paths for a given issue ID."""
    conn, cursor = None, None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT screenshot_path
            FROM screenshots
            WHERE issue_id = %s
            """,
            (issue_id,),
        )
        rows = cursor.fetchall()
        return [row[0] for row in rows]

    except Exception as e:
        print(f"[ERROR] get_screenshots: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ----------------------------
# 3. Retrieve All Issues (Admin)
# ----------------------------
def get_all_issues() -> List[Dict[str, Any]]:
    """Fetch all reported issues with reporter details (admin view)."""
    conn, cursor = None, None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT i.issue_id,
                   CONCAT(u.first_name, ' ', u.last_name) AS reporter,
                   i.issue_type, i.description, i.status,
                   i.reported_at, i.resolved_at,
                   GROUP_CONCAT(s.screenshot_path) AS screenshots
            FROM reported_issue i
            JOIN user_profile u ON i.user_id = u.user_id
            LEFT JOIN screenshots s ON i.issue_id = s.issue_id
            GROUP BY i.issue_id
            ORDER BY i.reported_at DESC
            """
        )
        return cast(List[Dict[str, Any]], cursor.fetchall())

    except Exception as e:
        print(f"[ERROR] get_all_issues: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ----------------------------
# 4. Resolve Issue (Admin)
# ----------------------------
def resolve_issue(issue_id: int) -> Dict[str, Any]:
    """Mark an issue as resolved."""
    conn, cursor = None, None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE reported_issue
            SET status = 'resolved', resolved_at = NOW()
            WHERE issue_id = %s
            """,
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


# ----------------------------
# 5. Update Issue Status (Admin)
# ----------------------------
def update_issue_status(issue_id: int, new_status: str) -> Dict[str, Any]:
    """
    Update issue status in the admin panel.

    Automatically sets resolved_at for closing statuses.
    Sends a notification to the reporter via dashboard and email.
    """
    allowed_statuses = {
        "OPEN",
        "UNDER_REVIEW",
        "IN_PROGRESS",
        "ESCALATED",
        "AWAITING_USER_INFO",
        "DELAYED",
        "EXPECTED_7_DAYS",
        "CONTACT_PENDING",
        "RESOLVED",
        "CLOSED",
        "REJECTED",
    }

    if new_status not in allowed_statuses:
        return {"success": False, "message": "Invalid status"}

    conn, cursor = None, None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # --- Update issue status and handle resolved_at timestamp ---
        cursor.execute(
            """
            UPDATE reported_issue
            SET status = %s,
                resolved_at = CASE
                    WHEN %s IN ('RESOLVED', 'CLOSED', 'REJECTED')
                    THEN NOW()
                    ELSE NULL
                END
            WHERE issue_id = %s
            """,
            (new_status, new_status, issue_id),
        )
        conn.commit()

        # --- Fetch reporter ID ---
        reporter_id = None
        try:
            with conn.cursor(dictionary=True) as cur:
                cur.execute(
                    "SELECT user_id FROM reported_issue WHERE issue_id=%s",
                    (issue_id,),
                )
                row = cur.fetchone()
                reporter_id = row.get("user_id") if row else None
        except Exception as e:
            print(f"[WARN] Could not fetch reporter ID: {e}")

        # --- Send notification if reporter exists ---
        if reporter_id:
            try:
                from backend.notification_system import NotificationSystem

                notif = NotificationSystem()
                subject = "Issue Status Updated"
                message = f"Your reported issue #{issue_id} has been marked as '{new_status}'."

                notif.send_message(
                    sender_role="system",
                    sender_id=0,
                    message=message,
                    recipients=[reporter_id],
                    channels=["dashboard", "email"],
                    subject=subject,
                    notif_type="issue",
                )
                print(
                    f"[INFO] Notification sent for issue {issue_id} to user {reporter_id}"
                )
            except Exception as notify_err:
                print(f"[WARN] Notification sending failed: {notify_err}")

        return {"success": True}

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[ERROR] update_issue_status: {e}")
        return {"success": False, "message": str(e)}

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
