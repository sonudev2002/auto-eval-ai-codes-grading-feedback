import os
import uuid
import smtplib
import requests
import random
import logging
from contextlib import contextmanager

from flask import request, session, jsonify, redirect, url_for, current_app, Response
from email.message import EmailMessage
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import time as systime  # system time for OTP expiry (unix timestamps)
import time  # used for time.tzset()
from datetime import date, datetime, time as datetime_time, timedelta

# timezone support
import pytz

from backend.db import get_connection
from user_agents import parse
from config import Config
import traceback
from typing import Any, Dict, Optional, TypedDict
from decimal import Decimal
from analytics import DBConnection, normalize_row
import cloudinary
import cloudinary.uploader

# -------------------- setup logging --------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    # simple console handler if not already configured by app
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# -------------------- timezone (IST) --------------------
# Prefer DB NOW() for SQL timestamps, but set process TZ for logging consistency.
try:
    os.environ["TZ"] = "Asia/Kolkata"
    time.tzset()  # may not exist on Windows; wrapped in try/except
    IST = pytz.timezone("Asia/Kolkata")
    logger.info("Timezone set to Asia/Kolkata for process")
except Exception:
    # On Windows, time.tzset may not be available; keep using pytz where needed.
    IST = pytz.timezone("Asia/Kolkata")
    logger.warning(
        "Could not call time.tzset(); using pytz aware timestamps where required"
    )


# -------------------- DB helper --------------------
def safe_commit(conn):
    """
    Commit a transaction, with rollback on error.
    Use within try/except blocks where conn exists.
    """
    try:
        conn.commit()
    except Exception as e:
        try:
            conn.rollback()
        except Exception as rb_err:
            logger.error("Rollback failed: %s", rb_err)
        logger.error("Commit failed, rolled back: %s", e)
        raise


# Keep upload folder constant
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_PIC_BYTES = int(os.getenv("MAX_PIC_BYTES", 2 * 1024 * 1024))  # 2 MB default
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def normalize_row(row: dict | None) -> dict | None:
    """Convert DB row values into JSON/template-safe types."""
    if row is None:
        return None

    clean: dict = {}
    for key, value in row.items():
        if value is None:
            clean[key] = None
        elif isinstance(value, Decimal):
            # Convert Decimal to float
            clean[key] = float(value)
        elif isinstance(value, datetime):
            # MySQL DATETIME is naive; localize to IST (makes it timezone-aware)
            if value.tzinfo is None:
                value = IST.localize(value)
            clean[key] = value.isoformat()
        elif isinstance(value, date):
            clean[key] = value.isoformat()
        elif isinstance(value, datetime_time):
            clean[key] = value.strftime("%H:%M:%S")
        elif isinstance(value, timedelta):
            clean[key] = str(value)
        else:
            clean[key] = value
    return clean


def now_ist() -> datetime:
    """Return timezone-aware IST datetime for logging or python-side use."""
    return datetime.now(IST)


USERS = []  # For checking uniqueness
OTP_STORE = {}


class UserRow(TypedDict):
    user_id: int
    email: str
    password: str
    role: str
    first_name: str
    middle_name: Optional[str]
    last_name: Optional[str]
    profile_picture_path: Optional[str]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_email_registered(email):
    email = (email or "").strip().lower()
    conn = get_connection()
    print("user management inside is_email_registerd")
    try:

        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM user_profile WHERE email = %s", (email,))
            yes = cur.fetchone()
            print(yes)
            return yes is not None
    finally:
        conn.close()


def save_profile_picture(file) -> str | None:
    """
    Saves uploaded profile picture.
    Priority:
     1) Cloudinary (if configured)
     2) Local uploads folder (fallback)
    Returns stored URL/path or None on failure.
    """
    if not file or not getattr(file, "filename", ""):
        return None

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return None

    # Check file size
    try:
        file.stream.seek(0, os.SEEK_END)
        size = file.stream.tell()
        file.stream.seek(0)
    except Exception:
        size = None

    if size is not None and size > MAX_PIC_BYTES:
        current_app.logger.warning("Profile picture too large (>2MB)")
        return None

    # 1️⃣ Try Cloudinary first
    if Config.CLOUDINARY_ENABLED:
        try:
            upload_result = cloudinary.uploader.upload(
                file,
                folder="auto-eval/profile_pictures",
                use_filename=True,
                unique_filename=True,
                overwrite=False,
            )
            url = upload_result.get("secure_url")
            if url:
                return url  # ✅ return CDN URL
        except Exception as e:
            current_app.logger.warning(f"Cloudinary upload failed: {e}")

    # 2️⃣ Fallback to Local Uploads
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        dest_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(dest_path)
        web_path = f"/uploads/{unique_name}".replace("\\", "/")
        return web_path
    except Exception as e:
        current_app.logger.error(f"Local save failed: {e}")
        return None


def generate_otp():
    print("user management inside generate_otp()")
    return f"{random.randint(100000, 999999)}"


def send_email_otp(to_email: str, otp: str) -> bool:
    print("user management inside send email otp")
    smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp-relay.brevo.com")
    smtp_port = int(os.getenv("EMAIL_SMTP_PORT", 587))
    smtp_user = os.getenv("EMAIL_SMTP_LOGIN", "98a206004@smtp-brevo.com")
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD_SENDER")

    if not all([smtp_user, password, sender, to_email]):
        current_app.logger.error("Missing SMTP credentials or recipient")
        return False

    msg = EmailMessage()
    msg["Subject"] = "Your OTP Code"
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(f"Your OTP code is: {otp}, Do not share with anyone.")

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(smtp_user, password)
            smtp.send_message(msg)
        return True
    except Exception as e:
        current_app.logger.error(f"Email sending error: {e}")
        return False


def send_sms_otp(mobile: str, otp: str) -> bool:
    print("user management inside send sms otp")
    url = os.getenv("FAST2SMS_URL", "https://www.fast2sms.com/dev/bulkV2")
    payload = {
        "authorization": os.getenv(
            "FAST2SMS_API_KEY", getattr(Config, "FAST2SMS_API_KEY", None)
        ),
        "message": f"Your OTP is {otp}",
        "language": "english",
        "route": "q",
        "numbers": mobile,
    }
    headers = {"cache-control": "no-cache", "authorization": payload["authorization"]}

    try:
        response = requests.post(url, data=payload, headers=headers, timeout=10)
        return response.status_code == 200 and "true" in (response.text or "").lower()
    except Exception as e:
        current_app.logger.error("SMS sending error: %s", e)
        return False


def send_otp_email():
    print("user management inside send_otp_email")
    data = request.get_json()
    email = data.get("email")
    if not email:
        return jsonify({"success": False, "message": "Email required"}), 400

    if not is_email_registered(email):
        return jsonify({"success": False, "message": "Email not found"}), 404
    otp = generate_otp()

    OTP_STORE[email] = {
        "otp": otp,
        "expires": time.time() + 300,
        "verified": False,
    }
    email_sent = send_email_otp(email, otp)
    if email_sent:
        return jsonify({"success": True, "message": "OTP sent successfully"}), 200
    else:
        return jsonify({"success": False, "message": "Failed to send OTP"}), 500


def send_otp():
    data = request.get_json()
    email = data.get("email")
    mobile = data.get("mobile")

    if not email or not mobile:
        return jsonify({"success": False, "message": "Email and mobile required"}), 400

    # if not is_email_registered(email):
    #     return jsonify({"success": False, "message": "Email already registered"}), 409

    otp = generate_otp()
    OTP_STORE[email] = {
        "otp": otp,
        "mobile": mobile,
        "expires": systime.time() + 300,
        "verified": False,
    }

    email_sent = send_email_otp(email, otp)
    sms_sent = send_sms_otp(mobile, otp)

    if email_sent or sms_sent:
        return jsonify({"success": True, "message": "OTP sent successfully"}), 200
    else:
        return jsonify({"success": False, "message": "Failed to send OTP"}), 500


def verify_otp():
    print("user management inside verify_otp()")
    data = request.get_json()
    email = data.get("email")
    otp = data.get("otp")
    record = OTP_STORE.get(email)

    if not record:
        return jsonify({"valid": False, "message": "No OTP request found"}), 400
    if systime.time() > record["expires"]:
        del OTP_STORE[email]
        return jsonify({"valid": False, "message": "OTP expired"}), 400
    if record["otp"] != otp:
        return jsonify({"valid": False, "message": "Invalid OTP"}), 401

    record["verified"] = True
    return jsonify({"valid": True, "message": "OTP verified"}), 200


def register_user():
    print("user management inside register_userU()")
    data = request.form
    email = data.get("email")
    otp_record = OTP_STORE.get(email)

    # Detect if this request came from admin
    from_admin = request.path.startswith("/admin/")

    if not from_admin:
        otp_record = OTP_STORE.get(email)
        if not otp_record or not otp_record.get("verified"):
            return jsonify({"error": "OTP verification required"}), 403

    # For normal users — remove verified OTP record
    if not from_admin:
        OTP_STORE.pop(email, None)

    if is_email_registered(email):
        return jsonify({"error": "Email already registered"}), 409
    print("afer is email register")
    mobile = data.get("mobile")
    if not mobile or not mobile.isdigit() or len(mobile) != 10:
        return jsonify({"error": "Invalid mobile number"}), 400

    file = request.files.get("profile_picture")
    profile_path = save_profile_picture(file) if file else None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Insert address (use provided fields, allow NULLs)
        cursor.execute(
            """
            INSERT INTO address (country_name, state_name, district_name, local_address, pincode, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            """,
            (
                data.get("country"),
                data.get("state", ""),
                data.get("district", ""),
                data.get("local_address", ""),
                data.get("pincode", ""),
            ),
        )

        address_id = cursor.lastrowid

        # Hash password and insert user
        password_hash = generate_password_hash(data.get("password") or "")
        cursor.execute(
            """
            INSERT INTO user_profile 
            (first_name, middle_name, last_name, email, mobile_number, password, role, profile_picture_path, address_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """,
            (
                data.get("first_name"),
                data.get("middle_name", ""),
                data.get("last_name"),
                email,
                mobile,
                password_hash,
                data.get("role", "student"),
                profile_path,
                address_id,
            ),
        )

        conn.commit()
        OTP_STORE.pop(email, None)
        return (
            jsonify(
                {"success": True, "message": "Signup successful", "redirect": True}
            ),
            201,
        )
    except Exception as e:
        current_app.logger.exception("DB insert failed")
        return jsonify({"error": "Registration failed", "details": str(e)}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass


def check_email():
    email = request.args.get("email")
    exists = is_email_registered(email)
    print("user management inside check_email()")
    return jsonify({"exists": exists})


def user_verify() -> Response:
    email: str = request.form["email"]
    password: str = request.form["password"]

    try:
        conn = get_connection()
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(
                """
                SELECT user_id, email, password, role, first_name, middle_name, last_name, profile_picture_path
                FROM user_profile
                WHERE email = %s
            """,
                (email,),
            )
            user: Optional[UserRow] = cursor.fetchone()  # type: ignore

        if (
            user
            and isinstance(user["password"], str)
            and check_password_hash(user["password"], password)
        ):
            ip: str = str(request.remote_addr or "unknown")
            user_agent_str: str = request.headers.get("User-Agent", "") or ""
            user_agent = parse(user_agent_str)

            # Extract details
            os_info: str = str(user_agent.os.family)
            browser_info: str = str(user_agent.browser.family)
            device_type: str = (
                "Mobile"
                if user_agent.is_mobile
                else (
                    "Tablet"
                    if user_agent.is_tablet
                    else "PC" if user_agent.is_pc else "Other"
                )
            )

            with conn.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO login_log 
                    (user_id, ip_address, login_time, device_info, os, browser, type)
                    VALUES (%s, %s, NOW(), %s, %s, %s, %s)""",
                    (
                        int(user["user_id"]),
                        ip,
                        user_agent_str,
                        os_info,
                        browser_info,
                        device_type,
                    ),
                )
                log_id: int = int(cursor.lastrowid or 0)
                conn.commit()

            # Build full name safely
            full_name_parts = [
                user.get("first_name") or "",
                user.get("middle_name") or "",
                user.get("last_name") or "",
            ]
            full_name = " ".join(part for part in full_name_parts if part).strip()

            # Handle profile picture path
            profile_pic_path = user.get("profile_picture_path")
            if (
                profile_pic_path
                and isinstance(profile_pic_path, str)
                and profile_pic_path.strip()
            ):
                profile_pic = profile_pic_path.replace("\\", "/")
            else:
                profile_pic = url_for("static", filename="images/default_avatar.png")

            session["user"] = {
                "user_id": int(user["user_id"]),
                "email": str(user["email"]),
                "role": str(user["role"]),
                "log_id": log_id,
                "full_name": full_name,
                # ✅ store Cloudinary or local path under correct key name
                "profile_picture_path": str(profile_pic),
            }

            return jsonify({"success": True, "redirect": f"/{user['role']}_dashboard"})
        else:
            return jsonify({"success": False, "message": "Invalid email or password"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
    finally:
        conn.close()


def user_logout(session):
    log_id = session.get("user", {}).get("log_id")
    if log_id:
        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE login_log SET logout_time =NOW() WHERE log_id= %s",
                    (log_id,),
                )
                conn.commit()
        except Exception as e:
            print("Error updating logout_time", e)
        finally:
            conn.close()
    session.clear()
    return redirect(url_for("index"))


def change_password():
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    email = data.get("email")
    role = data.get("role")
    password = data.get("password")

    # 1. Validation
    if not email or not role or not password:
        return jsonify({"success": False, "error": "Missing required fields"}), 400

    # 2. Check if email exists
    if not is_email_registered(email):
        return jsonify({"success": False, "error": "Email not found"}), 404

    print("user management inside change_password()")

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 3. Hash the new password
        hashed_password = generate_password_hash(password)

        # 4. Update query
        sql = """UPDATE user_profile 
                 SET password = %s 
                 WHERE email = %s AND role = %s"""
        cursor.execute(sql, (hashed_password, email, role))
        conn.commit()

        # 5. Check if any row was updated
        if cursor.rowcount == 0:
            print("user management inside change_password() no match found")
            return jsonify({"success": False, "error": "No matching user found"}), 404

        return (
            jsonify({"success": True, "message": "Password updated successfully"}),
            200,
        )

    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        print("Exception details:", str(e))
        return (
            jsonify(
                {"success": False, "error": "Change Password failed", "details": str(e)}
            ),
            500,
        )

    finally:
        if conn:
            conn.close()


# --------------------
# Read-only: StudentProfileData (split: basic vs performance)
# --------------------
class StudentProfileData:
    """Read-only accessors for a student's profile and analytics (split)."""

    def __init__(self, user_id: int) -> None:
        self.user_id = user_id

    # --- Public: profile page data (lightweight) ---
    def get_basic_profile(self) -> Dict[str, Any]:
        """Return only the data required by profile.html"""
        return {
            "user": self.get_user_info(),
            "profile": self.get_address(),
        }

    # --- Public: performance page data (analytics + completed assignments) ---
    def get_performance_data(self) -> Dict[str, Any]:
        """Return the analytics & completed assignments required by performance_history.html"""
        return {
            "report": self.get_performance(),
            "completed_assignments": self.get_completed_assignments(),
        }

    # --- Internal helpers ---
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM user_profile WHERE user_id = %s"
        with DBConnection() as cur:
            cur.execute(query, (self.user_id,))
            row = cur.fetchone()
        return normalize_row(row) if row else None

    def get_address(self) -> Optional[Dict[str, Any]]:
        query = """
            SELECT a.address_id, a.country_name, a.state_name, a.district_name,
                   a.local_address, a.pincode
            FROM user_profile u
            LEFT JOIN address a ON u.address_id = a.address_id
            WHERE u.user_id = %s
        """
        with DBConnection() as cur:
            cur.execute(query, (self.user_id,))
            row = cur.fetchone()
        return normalize_row(row) if row else None

    def get_performance(self) -> Dict[str, Any]:
        perf_query = """
            SELECT average_score, completion_rate, pass_rate,
                plagiarism_incidents, performance_band, total_assignments
            FROM student_performance_analytics
            WHERE user_id = %s
        """
        stats_query = """
            SELECT d.difficulty_types, s.assignment_count, s.average_score, s.average_pass_rate
            FROM student_difficulty_stats s
            JOIN difficulty_level d ON s.difficulty_level = d.level_id
            WHERE s.user_id = %s
        """

        # ---- Fix: ensure full fetch and close before next cursor ----
        with DBConnection() as cur:
            cur.execute(perf_query, (self.user_id,))
            perf_row = cur.fetchone()
            if cur._connection.unread_result:
                cur.fetchall()  # consume if any pending
        perf = normalize_row(perf_row) if perf_row else {}

        with DBConnection() as cur:
            cur.execute(stats_query, (self.user_id,))
            stats_rows = cur.fetchall() or []

        perf["assignment_counts"] = [s.get("assignment_count", 0) for s in stats_rows]
        perf["avg_scores"] = [s.get("average_score", 0) for s in stats_rows]
        perf["pass_rates"] = [s.get("average_pass_rate", 0) for s in stats_rows]
        perf["difficulty_labels"] = [
            s.get("difficulty_types", "Unknown") for s in stats_rows
        ]
        return perf

    def get_completed_assignments(self) -> Dict[str, list]:
        # include submission_id so the UI can link to specific submission reports
        query = """
            SELECT DISTINCT a.assignment_id, a.title, d.difficulty_types, c.submission_id
            FROM code_submission c
            JOIN assignment a ON c.assignment_id = a.assignment_id
            JOIN difficulty_level d ON a.difficulty_level = d.level_id
            WHERE c.user_id = %s
            ORDER BY a.assignment_id DESC
        """
        with DBConnection() as cur:
            cur.execute(query, (self.user_id,))
            rows = cur.fetchall() or []

        grouped: Dict[str, list] = {}
        for r in rows:
            nr = normalize_row(r) if r else {}
            key = nr.get("difficulty_types") or "Unknown"
            grouped.setdefault(key, []).append(nr)
        return grouped

    # keep a no-op close for compatibility (no persistent conn stored)
    def close(self) -> None:
        return None


# --------------------
# Read-only: InstructorProfileData (split + backward-compatible)
# --------------------
class InstructorProfileData:
    """Fetch profile + performance analytics for instructors (split)."""

    def __init__(self, user_id: int) -> None:
        self.user_id = user_id

    # Backwards-compatible method: returns both basic + performance (if any code still calls it)
    def get_profile_data(self) -> dict[str, Any]:
        result = self.get_basic_profile()
        perf = self.get_performance_data()
        result.update(perf)
        return result

    # Basic profile (for profile.html)
    def get_basic_profile(self) -> dict[str, Any]:
        return {
            "user": self.get_user_info(),
            "profile": self.get_address(),
        }

    # Performance data (for instructor-specific performance page)
    def get_performance_data(self) -> dict[str, Any]:
        return {
            "report": self.get_performance(),
            "managed_assignments": self.get_managed_assignments(),
        }

    # --- Internal helpers ---
    def get_user_info(self) -> Optional[dict[str, Any]]:
        query = "SELECT * FROM user_profile WHERE user_id = %s"
        with DBConnection() as cur:
            cur.execute(query, (self.user_id,))
            row = cur.fetchone()
        return normalize_row(row) if row else None

    def get_address(self) -> Optional[dict[str, Any]]:
        query = """
            SELECT a.address_id, a.country_name, a.state_name, a.district_name,
                   a.local_address, a.pincode
            FROM user_profile u
            LEFT JOIN address a ON u.address_id = a.address_id
            WHERE u.user_id = %s
        """
        with DBConnection() as cur:
            cur.execute(query, (self.user_id,))
            row = cur.fetchone()
        return normalize_row(row) if row else None

    def get_performance(self) -> dict[str, Any]:
        perf_query = """
            SELECT total_assignments_created, total_submissions_received,
                overall_avg_score, avg_pass_rate,
                plagiarism_rate, feedback_score_avg,
                responsiveness_score, consistency_score
            FROM instructor_performance_analytics
            WHERE user_id = %s
        """
        stats_query = """
            SELECT d.difficulty_types, s.assignment_count, s.average_score,
                s.average_pass_rate, s.average_feedback_score
            FROM instructor_difficulty_stats s
            JOIN difficulty_level d ON s.difficulty_level = d.level_id
            WHERE s.user_id = %s
        """

        with DBConnection() as cur:
            cur.execute(perf_query, (self.user_id,))
            perf_row = cur.fetchone() or {}

        with DBConnection() as cur:
            cur.execute(stats_query, (self.user_id,))
            stats_rows = cur.fetchall() or []

        perf: dict[str, Any] = normalize_row(perf_row) if perf_row else {}
        perf["assignment_counts"] = [s.get("assignment_count", 0) for s in stats_rows]
        perf["avg_scores"] = [s.get("average_score", 0) for s in stats_rows]
        perf["pass_rates"] = [s.get("average_pass_rate", 0) for s in stats_rows]
        perf["feedback_scores"] = [
            s.get("average_feedback_score", 0) for s in stats_rows
        ]
        perf["difficulty_labels"] = [
            s.get("difficulty_types", "Unknown") for s in stats_rows
        ]
        return perf

    def get_managed_assignments(self) -> dict[str, list[dict[str, Any]]]:
        query = """
            SELECT a.assignment_id, a.title, d.difficulty_types,
                   COUNT(DISTINCT c.user_id) AS distinct_students
            FROM assignment a
            LEFT JOIN code_submission c ON a.assignment_id = c.assignment_id
            JOIN difficulty_level d ON a.difficulty_level = d.level_id
            WHERE a.instructor_id = %s
            GROUP BY a.assignment_id, d.difficulty_types
            ORDER BY a.assignment_id DESC
        """
        with DBConnection() as cur:
            cur.execute(query, (self.user_id,))
            rows = cur.fetchall() or []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            nr = normalize_row(r) if r else {}
            key: str = nr.get("difficulty_types") or "Unknown"
            grouped.setdefault(key, []).append(nr)
        return grouped

    def close(self) -> None:
        """Compatibility no-op for unified interface with StudentProfileData."""
        return None


# --------------------
# Read-only: AdminProfileData
# --------------------
class AdminProfileData:
    """Fetch profile + address for admins."""

    def __init__(self, user_id: int) -> None:
        self.user_id = user_id
        self.conn = get_connection()

    def get_profile_data(self) -> Dict[str, Any]:
        return {
            "user": self.get_user_info(),
            "profile": self.get_address(),
        }

    def get_user_info(self) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM user_profile WHERE user_id = %s"
        with self.conn.cursor(dictionary=True, buffered=True) as cur:
            cur.execute(query, (self.user_id,))
            return normalize_row(cur.fetchone())

    def get_address(self) -> Optional[Dict[str, Any]]:
        query = """
            SELECT a.address_id, a.country_name, a.state_name, a.district_name,
                   a.local_address, a.pincode
            FROM user_profile u
            LEFT JOIN address a ON u.address_id = a.address_id
            WHERE u.user_id = %s
        """
        with self.conn.cursor(dictionary=True, buffered=True) as cur:
            cur.execute(query, (self.user_id,))
            return normalize_row(cur.fetchone())

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


# --------------------
# Updater: UpdateProfileData
# --------------------


class UpdateProfileData(AdminProfileData):
    """
    Extends ProfileData with update capabilities.
    Includes:
      - update_picture()
      - update_info()
      - update_password()
    """

    # ----------------------------
    # Profile Picture
    # ----------------------------
    def update_picture(self, file=None, delete_picture: bool = False) -> Dict[str, Any]:
        try:
            current_user = self.get_user_info() or {}
            if not current_user:
                return {"status": "error", "message": "User not found."}

            old_pic_path = current_user.get("profile_picture_path")
            new_pic_path = None

            # Helper: delete old local file
            def delete_local(path: str):
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    current_app.logger.warning(
                        f"Failed to delete local file {path}: {e}"
                    )

            # Delete existing picture
            if delete_picture:
                if old_pic_path:
                    delete_local(old_pic_path)
                new_pic_path = None

            # Upload new picture flow
            elif file and hasattr(file, "filename") and file.filename:
                if not allowed_file(file.filename):
                    return {"status": "error", "message": "File extension not allowed."}

                try:
                    file.stream.seek(0, os.SEEK_END)
                    size = file.stream.tell()
                    file.stream.seek(0)
                except Exception:
                    size = None

                if size is not None and size > MAX_PIC_BYTES:
                    return {"status": "error", "message": "File too large (max 2MB)."}

                mimetype = getattr(file, "mimetype", "") or ""
                if not mimetype.startswith("image/"):
                    return {
                        "status": "error",
                        "message": "Uploaded file is not an image.",
                    }

                try:
                    new_pic_path = save_profile_picture(file)
                    if not new_pic_path:
                        return {
                            "status": "error",
                            "message": "Failed to save profile picture.",
                        }
                except Exception as e:
                    current_app.logger.error(f"Error saving profile picture: {e}")
                    return {
                        "status": "error",
                        "message": "Failed to save profile picture.",
                    }

                # Delete old local picture after successful upload
                if old_pic_path:
                    try:
                        delete_local(old_pic_path)
                    except Exception as e:
                        current_app.logger.warning(
                            f"Failed to delete old picture after upload: {e}"
                        )

            # Persist to DB
            query = "UPDATE user_profile SET profile_picture_path=%s WHERE user_id=%s"
            with self.conn.cursor() as cursor:
                cursor.execute(query, (new_pic_path, self.user_id))
            self.conn.commit()

            if "user" in session:
                session["user"]["profile_picture_path"] = new_pic_path or session[
                    "user"
                ].get("profile_picture_path", "")

            return {
                "status": "success",
                "message": "Profile picture updated.",
                "profile_picture": new_pic_path,
            }

        except Exception as e:
            try:
                self.conn.rollback()
            except Exception:
                pass
            current_app.logger.error(f"update_picture failed: {traceback.format_exc()}")
            return {"status": "error", "message": str(e)}

    # ----------------------------
    # Update Info
    # ----------------------------
    def update_info(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handles basic info (name, email, mobile, address)."""
        try:
            current_user = self.get_user_info() or {}
            current_address = self.get_address() or {}
            if not current_user:
                return {"status": "error", "message": "User not found."}

            first_name = form_data.get("first_name", "").strip() or current_user.get(
                "first_name"
            )
            email = form_data.get("email", "").strip() or current_user.get("email")

            if not first_name or not email:
                return {
                    "status": "error",
                    "message": "First name and email cannot be empty.",
                }

            def normalize_optional(value, old_val=None):
                value = (value or "").strip()
                return None if value == "" else value or old_val

            middle_name = normalize_optional(
                form_data.get("middle_name"), current_user.get("middle_name")
            )
            last_name = normalize_optional(
                form_data.get("last_name"), current_user.get("last_name")
            )
            mobile_number = normalize_optional(
                form_data.get("mobile_number"), current_user.get("mobile_number")
            )

            with self.conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE user_profile
                    SET first_name=%s, middle_name=%s, last_name=%s,
                        email=%s, mobile_number=%s
                    WHERE user_id=%s
                    """,
                    (
                        first_name,
                        middle_name,
                        last_name,
                        email,
                        mobile_number,
                        self.user_id,
                    ),
                )

                address_id = current_user.get("address_id")
                if not address_id:
                    cursor.execute(
                        "INSERT INTO address (country_name, state_name, district_name, local_address, pincode) VALUES (NULL,NULL,NULL,NULL,NULL)"
                    )
                    self.conn.commit()
                    address_id = cursor.lastrowid
                    cursor.execute(
                        "UPDATE user_profile SET address_id=%s WHERE user_id=%s",
                        (address_id, self.user_id),
                    )
                    self.conn.commit()
                    current_address = {"address_id": address_id}

                country = normalize_optional(
                    form_data.get("country_name"), current_address.get("country_name")
                )
                state = normalize_optional(
                    form_data.get("state_name"), current_address.get("state_name")
                )
                district = normalize_optional(
                    form_data.get("district_name"), current_address.get("district_name")
                )
                local_address = normalize_optional(
                    form_data.get("local_address"), current_address.get("local_address")
                )
                pincode = normalize_optional(
                    form_data.get("pincode"), current_address.get("pincode")
                )

                cursor.execute(
                    """
                    UPDATE address
                    SET country_name=%s, state_name=%s, district_name=%s,
                        local_address=%s, pincode=%s
                    WHERE address_id=%s
                    """,
                    (country, state, district, local_address, pincode, address_id),
                )

            self.conn.commit()
            return {
                "status": "success",
                "message": "Profile info updated successfully.",
            }

        except Exception as e:
            try:
                self.conn.rollback()
            except Exception:
                pass
            current_app.logger.error(f"update_info failed: {traceback.format_exc()}")
            return {"status": "error", "message": str(e)}

    # ----------------------------
    # Update Password
    # ----------------------------
    def update_password(
        self, current_password: str, new_password: str, confirm_password: str
    ) -> Dict[str, Any]:
        """Securely update user's password."""
        try:
            current_user = self.get_user_info()
            if not current_user:
                return {"status": "error", "message": "User not found."}

            if not check_password_hash(
                current_user.get("password"), current_password or ""
            ):
                return {"status": "error", "message": "Current password is incorrect."}

            if not new_password or new_password != confirm_password:
                return {
                    "status": "error",
                    "message": "New password and confirm password do not match.",
                }

            password_hash = generate_password_hash(new_password)
            query = "UPDATE user_profile SET password=%s WHERE user_id=%s"
            with self.conn.cursor() as cursor:
                cursor.execute(query, (password_hash, self.user_id))
            self.conn.commit()

            return {"status": "success", "message": "Password updated successfully."}

        except Exception as e:
            try:
                self.conn.rollback()
            except Exception:
                pass
            current_app.logger.error(
                f"Password update failed for user {self.user_id}: {traceback.format_exc()}"
            )
            return {"status": "error", "message": "Failed to update password."}
