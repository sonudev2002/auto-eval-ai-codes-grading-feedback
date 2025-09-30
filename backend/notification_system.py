# backend/notification_system.py
import logging
import time
import os
import textwrap
from typing import List, Iterable, Optional, Set, Union, Dict, Any, Tuple, cast
from concurrent.futures import ThreadPoolExecutor
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests

from backend.db import get_connection
from config import Config

# -----------------------------
# Logger setup
# -----------------------------
logger = logging.getLogger("notification_system")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -----------------------------
# Retry decorator
# -----------------------------
def retry(times: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry wrapper for unstable operations (e.g., email/SMS APIs)."""

    def decorator(fn):
        def wrapper(*args, **kwargs):
            _delay = delay
            last_exc = None
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    logger.warning(
                        "%s failed attempt %d/%d: %s",
                        fn.__name__,
                        attempt + 1,
                        times,
                        e,
                    )
                    time.sleep(_delay)
                    _delay *= backoff
            logger.error("%s failed after %d attempts", fn.__name__, times)
            raise last_exc

        return wrapper

    return decorator


# ============================================================
# Repositories (DB Layer) - all DB work stays in this module
# ============================================================
class UserRepository:
    """Handles user data fetching."""

    @staticmethod
    def _fetch_single_col(query: str, params: tuple = ()) -> List[Any]:
        conn = get_connection()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(query, params)
            rows = cur.fetchall()
            return rows
        finally:
            conn.close()

    @staticmethod
    def get_user_ids_by_role(role: str) -> List[int]:
        rows = UserRepository._fetch_single_col(
            "SELECT user_id FROM User_Profile WHERE role = %s", (role,)
        )
        return [r["user_id"] for r in rows] if rows else []

    @staticmethod
    def get_all_students_and_instructors_ids() -> List[int]:
        rows = UserRepository._fetch_single_col(
            "SELECT user_id FROM User_Profile WHERE role IN ('student','instructor')"
        )
        return [r["user_id"] for r in rows] if rows else []

    @staticmethod
    def get_all_user_ids() -> List[int]:
        rows = UserRepository._fetch_single_col("SELECT user_id FROM User_Profile")
        return [r["user_id"] for r in rows] if rows else []

    @staticmethod
    def get_user_email(user_id: int) -> Optional[str]:
        rows = UserRepository._fetch_single_col(
            "SELECT email FROM User_Profile WHERE user_id=%s", (user_id,)
        )
        if rows and rows[0].get("email"):
            return rows[0]["email"]
        return None

    @staticmethod
    def get_user_mobile_number(user_id: int) -> Optional[str]:
        rows = UserRepository._fetch_single_col(
            "SELECT mobile_number FROM User_Profile WHERE user_id=%s", (user_id,)
        )
        if rows and rows[0].get("mobile_number"):
            return rows[0]["mobile_number"]
        return None


class NotificationRepository:
    """Handles persistence of notifications and broadcasts."""

    def create_notification(
        self,
        user_id: int,
        message: str,
        notif_type: str = "info",
        mode: str = "dashboard",
        subject: Optional[str] = None,
    ) -> Optional[int]:
        """Insert a personal/system notification (Notification table)."""
        if not user_id or not message:
            return None
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO Notification (user_id, message, notification_mode, type, subject, status, created_at)
                VALUES (%s, %s, %s, %s, %s, 'unread', NOW())
                """,
                (user_id, message, mode, notif_type, subject),
            )
            conn.commit()
            return getattr(cursor, "lastrowid", None)
        finally:
            conn.close()

    def fetch_user_notifications(
        self,
        user_id: int,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Fetch personal notifications for a user including the notification_mode so frontend can show channels.
        Returns list of dicts containing notification_id, message, type, status, notification_mode, created_at, subject.
        """
        conn = get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            if status:
                cursor.execute(
                    """
                    SELECT notification_id, message, type, status, notification_mode, created_at, subject 
                    FROM Notification
                    WHERE user_id=%s AND notification_mode IS NOT NULL AND status=%s
                    ORDER BY created_at DESC 
                    LIMIT %s OFFSET %s
                    """,
                    (user_id, status, limit, offset),
                )
            else:
                cursor.execute(
                    """
                    SELECT notification_id, message, type, status, notification_mode, created_at, subject 
                    FROM Notification
                    WHERE user_id=%s AND notification_mode IS NOT NULL
                    ORDER BY created_at DESC 
                    LIMIT %s OFFSET %s
                    """,
                    (user_id, limit, offset),
                )
            rows = cursor.fetchall() or []
            # ensure created_at are datetime objects (DB driver usually does this)
            return rows
        finally:
            conn.close()

    def mark_as_read(self, notification_id: int):
        """Mark a notification as read."""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE Notification SET status='read' WHERE notification_id=%s",
                (notification_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def create_broadcast(
        self,
        message: str,
        btype: str = "general",
        mode: str = "system",
        subject: Optional[str] = None,
        notif_type: str = "broadcast",
    ) -> Optional[int]:
        """
        Insert a broadcast row (one per broadcast).
        We store btype in broadcast_type and mode string in broadcast_mode (comma-separated channels).
        """
        if not message:
            return None
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO BroadCast_Notification (broadcast_type, broadcast_mode, message, created_at)
                VALUES (%s, %s, %s, NOW())
                """,
                (btype, mode, message),
            )
            conn.commit()
            return getattr(cursor, "lastrowid", None)
        finally:
            conn.close()

    def fetch_broadcasts_for_types(
        self, types: Iterable[str], limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Fetch broadcast rows whose broadcast_type is in the provided types list.
        Always returns a list of dicts.
        """
        types_list = list(types)
        if not types_list:
            return []

        conn = get_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            placeholders = ",".join(["%s"] * len(types_list))
            sql = f"""
                SELECT broadcast_id, broadcast_type, broadcast_mode, message, created_at
                FROM BroadCast_Notification
                WHERE broadcast_type IN ({placeholders})
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            params = types_list + [limit, offset]

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            return cast(List[Dict[str, Any]], rows or [])

        except Exception as e:
            logger.exception("Error in fetch_broadcasts_for_types: %s", e)
            return []
        finally:
            conn.close()


# ============================================================
# Delivery Strategies
# ============================================================
class InAppDelivery:
    def __init__(self, repo: NotificationRepository):
        self.repo = repo

    def send(
        self,
        user_id: int,
        message: str,
        notif_type="info",
        subject: Optional[str] = None,
    ):
        # In-app delivery = create Notification row (dashboard)
        return self.repo.create_notification(
            user_id, message, notif_type, mode="dashboard", subject=subject
        )


class EmailDelivery:
    def __init__(self):
        self.smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
        self.sender_email = os.getenv(
            "EMAIL_SENDER", getattr(Config, "EMAIL_SENDER", None)
        )
        self.sender_password = os.getenv(
            "EMAIL_PASSWORD", getattr(Config, "EMAIL_PASSWORD_SENDER", None)
        )

    @retry(times=3, delay=1.0, backoff=2.0)
    def send(self, to_email: str, subject: str, body: str, timeout: int = 10):
        if not to_email or not self.sender_email:
            logger.debug(
                "EmailDelivery: missing recipient or sender config, skipping email send."
            )
            return
        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = to_email
        msg["Subject"] = subject or "Notification"
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=timeout) as server:
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, [to_email], msg.as_string())


class SMSDelivery:
    def __init__(self):
        self.api_key = os.getenv(
            "FAST2SMS_API_KEY", getattr(Config, "FAST2SMS_API_KEY", None)
        )
        self.sender_id = os.getenv(
            "FAST2SMS_SENDER_ID", getattr(Config, "FAST2SMS_SENDER_ID", None)
        )
        self.endpoint = os.getenv(
            "FAST2SMS_ENDPOINT",
            getattr(Config, "FAST2SMS_ENDPOINT", "https://www.fast2sms.com/dev/bulkV2"),
        )

    @retry(times=2, delay=1.0, backoff=2.0)
    def send(self, to_mobile: str, message: str):
        if not to_mobile or not self.api_key:
            logger.debug("SMSDelivery: missing mobile or api key, skipping SMS send.")
            return
        payload = {
            "sender_id": self.sender_id or "FSTSMS",
            "message": message,
            "language": "english",
            "route": "v3",
            "numbers": to_mobile,
        }
        headers = {
            "authorization": self.api_key,
            "Content-Type": "application/json",
        }
        resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()


# ============================================================
# Orchestrator - high-level logic lives here
# ============================================================
class NotificationSystem:
    """Main facade for sending/broadcasting notifications."""

    def __init__(self, use_background: bool = False, max_workers: int = 20):
        self.users = UserRepository()
        self.repo = NotificationRepository()
        self.inapp = InAppDelivery(self.repo)
        self.email = EmailDelivery()
        self.sms = SMSDelivery()
        self.use_background = use_background
        self.executor = (
            ThreadPoolExecutor(max_workers=max_workers) if use_background else None
        )

    # -----------------------------
    # Helpers
    # -----------------------------
    def _resolve_recipients(self, recipients: Iterable[Union[str, int]]) -> Set[int]:
        """Expand recipient groups like 'all', 'students', 'instructors' into user ids."""
        resolved: Set[int] = set()
        for r in recipients:
            if isinstance(r, str):
                key = r.strip().lower()
                if key == "students":
                    resolved.update(self.users.get_user_ids_by_role("student"))
                elif key == "instructors":
                    resolved.update(self.users.get_user_ids_by_role("instructor"))
                elif key in ("students_and_instructors", "students+instructors"):
                    resolved.update(self.users.get_all_students_and_instructors_ids())
                elif key == "all" or key == "all_users":
                    resolved.update(self.users.get_all_user_ids())
                else:
                    # allow a comma-separated list of numeric ids in a single string
                    if "," in key:
                        parts = [p.strip() for p in key.split(",") if p.strip()]
                        for p in parts:
                            if p.isdigit():
                                resolved.add(int(p))
            elif isinstance(r, int):
                resolved.add(r)
        return resolved

    def _send_single(
        self,
        user_id: int,
        message: str,
        channels: Iterable[str],
        subject=None,
        notif_type="info",
    ):
        """Send a notification to one user across multiple channels.

        Important: this method is used for targeted notifications (personal).
        Broadcasts are handled by broadcast() which does not create per-user Notification rows.
        """
        try:
            for ch in channels:
                ch_key = ch.strip().lower()
                if ch_key == "dashboard":
                    # dashboard => create Notification row
                    self.inapp.send(user_id, message, notif_type, subject=subject)
                elif ch_key == "email":
                    email = self.users.get_user_email(user_id)
                    if email:
                        try:
                            self.email.send(
                                email,
                                subject or f"[{notif_type.upper()}] Notification",
                                message,
                            )
                        except Exception as e:
                            logger.exception(
                                "Email send failed for user %s: %s", user_id, e
                            )
                elif ch_key == "sms":
                    mobile = self.users.get_user_mobile_number(user_id)
                    if mobile:
                        try:
                            self.sms.send(mobile, message)
                        except Exception as e:
                            logger.exception(
                                "SMS send failed for user %s: %s", user_id, e
                            )
                else:
                    logger.debug("Unknown channel '%s' requested - skipping", ch_key)
        except Exception:
            logger.exception("_send_single encountered an unexpected error")

    # -----------------------------
    # Public APIs
    # -----------------------------
    def send_message(
        self,
        sender_role: str,
        sender_id: int,
        message: str,
        recipients: Iterable[Union[str, int]],
        channels: Iterable[str] = ("dashboard",),
        subject: Optional[str] = None,
        notif_type: str = "info",
    ):
        """
        Send to specific user(s) or groups → saved in Notification table (for dashboard) and optionally emailed/SMSed.
        recipients: can be list of ints or group keys like 'students', 'instructors', 'all', or comma-separated ids string.
        channels: any combination of 'dashboard', 'email', 'sms'
        """
        if not message:
            logger.warning("send_message called with empty message")
            return

        # Resolve recipients -> deduplicated set
        resolved_ids = self._resolve_recipients(recipients)
        if not resolved_ids:
            logger.warning("No recipients resolved for message: %s", message)
            return

        # normalize channels
        channels_list = [c.strip().lower() for c in channels if c]
        if not channels_list:
            channels_list = ["dashboard"]

        for uid in resolved_ids:
            if self.use_background and self.executor:
                self.executor.submit(
                    self._send_single, uid, message, channels_list, subject, notif_type
                )
            else:
                self._send_single(uid, message, channels_list, subject, notif_type)

    def broadcast(
        self,
        message: str,
        channels: Iterable[str] = ("dashboard",),
        subject: Optional[str] = None,
        notif_type: str = "broadcast",
        btype: str = "general",
        mode: str = "system",
    ):
        """
        Broadcast -> create one broadcast row in BroadCast_Notification (no per-user notification rows).
        For non-dashboard channels (email/sms) we still attempt delivery, but we DON'T create Notification rows
        so broadcast records live only in broadcast table (per your spec).
        """
        if not message:
            logger.warning("broadcast called with empty message")
            return None

        # Normalize mode / channels: we'll store the channel string on broadcast_mode
        channels_list = [c.strip().lower() for c in channels if c]
        mode_string = "+".join(channels_list) or mode or "system"

        try:
            broadcast_id = self.repo.create_broadcast(
                message=message,
                btype=btype,
                mode=mode_string,
                subject=subject,
                notif_type=notif_type,
            )
        except Exception:
            logger.exception("Failed to insert broadcast row")
            broadcast_id = None

        # If channels include email or sms, send them out to all users (no Notification row created)
        all_user_ids = set(self.users.get_all_user_ids())
        if not all_user_ids:
            logger.info("No user ids found to send broadcast to")
            return broadcast_id

        # Delivery function for broadcast: only deliver via email/sms (dashboard is represented by broadcast row)
        def _deliver_broadcast_to_user(uid: int):
            try:
                if "email" in channels_list:
                    email = self.users.get_user_email(uid)
                    if email:
                        try:
                            self.email.send(
                                email, subject or f"[BROADCAST] {btype}", message
                            )
                        except Exception as e:
                            logger.exception(
                                "Broadcast email send failed for %s: %s", uid, e
                            )
                if "sms" in channels_list:
                    mobile = self.users.get_user_mobile_number(uid)
                    if mobile:
                        try:
                            self.sms.send(mobile, message)
                        except Exception as e:
                            logger.exception(
                                "Broadcast sms send failed for %s: %s", uid, e
                            )
            except Exception:
                logger.exception(
                    "Unexpected error delivering broadcast to user %s", uid
                )

        # Dispatch deliveries (background if enabled)
        for uid in all_user_ids:
            if self.use_background and self.executor:
                self.executor.submit(_deliver_broadcast_to_user, uid)
            else:
                _deliver_broadcast_to_user(uid)

        logger.info(
            "Broadcast %s (%s/%s) accepted. channels=%s delivered_attempted=%d",
            broadcast_id,
            btype,
            mode_string,
            channels_list,
            len(all_user_ids),
        )
        return broadcast_id

    # -----------------------------
    # Wrappers for system events
    # -----------------------------
    def notify_signup(self, user_id: int):
        msg = getattr(Config, "SIGNUP_WELCOME_MESSAGE", "Welcome to the platform!")
        self.send_message(
            "system",
            0,
            msg,
            [user_id],
            channels=("dashboard", "email"),
            notif_type="welcome",
            subject="Welcome!",
        )

    def notify_assignment_uploaded(
        self,
        assignment_id: int,
        title: str,
        due_date: Union[str, datetime.date, datetime.datetime],
        instructor_id: int,
        recipient_keys: Iterable[Union[str, int]] = ("students",),
    ):
        """Notify students when a new assignment is uploaded.

        For system requirement: assignments may produce a broadcast OR notifications.
        Here we create a broadcast row AND (optionally) send dashboard notifications to recipients if
        the caller requested recipient_keys that resolve to specific users (keeps API flexible).
        By default we will create a broadcast targeted at 'students' so students see it in their feed via broadcast table.
        """
        if isinstance(due_date, (datetime.date, datetime.datetime)):
            due_date_str = due_date.isoformat()
        else:
            due_date_str = str(due_date).strip()

        message = textwrap.dedent(
            f"""\ 
            🚀 New assignment: "{title}"
            📅 Due Date: {due_date_str}
            🆔 Assignment ID: {assignment_id}
            Please check the assignment and submit before the deadline.
            """
        ).strip()

        subject = f"New Assignment: {title}"

        # Per your spec: store in broadcast table for 'students'
        btype = "all_students"
        mode = "dashboard"  # broadcast_mode will reflect channel(s)
        self.broadcast(
            message=message,
            channels=("dashboard",),
            subject=subject,
            notif_type="assignment",
            btype=btype,
            mode=mode,
        )

        # Additionally: allow immediate targeted notifications if explicit recipient ids were passed (flexible).
        # If recipient_keys contains ints or resolvable user ids, send dashboard notifications to those users as well.
        resolved = self._resolve_recipients(recipient_keys)
        if resolved:
            # send targeted notification rows (dashboard) to resolved ids
            self.send_message(
                sender_role="instructor",
                sender_id=instructor_id,
                message=message,
                recipients=resolved,
                channels=("dashboard",),
                subject=subject,
                notif_type="assignment",
            )

        logger.info(
            "notify_assignment_uploaded: assignment_id=%s title=%s broadcasted_to=%s resolved_target_count=%d",
            assignment_id,
            title,
            btype,
            len(resolved),
        )

    def notify_profile_updated(
        self, user_id: int, changed_fields: Optional[List[str]] = None
    ):
        changed = ", ".join(changed_fields) if changed_fields else "your profile"
        msg = f"Your {changed} has been updated successfully."
        self.send_message(
            sender_role="system",
            sender_id=0,
            message=msg,
            recipients=[user_id],
            channels=("dashboard", "email"),
            subject="Profile Updated",
            notif_type="profile",
        )

    def notify_password_reset(self, user_id: int):
        msg = "Your password was changed successfully. If you did not perform this action, contact support immediately."
        self.send_message(
            sender_role="system",
            sender_id=0,
            message=msg,
            recipients=[user_id],
            channels=("dashboard", "email"),
            subject="Password Changed",
            notif_type="security",
        )

    def notify_issue_resolved(
        self, issue_id: int, user_id: int, resolution_note: Optional[str] = None
    ):
        msg = f"Your reported issue (ID: {issue_id}) has been marked resolved."
        if resolution_note:
            msg += f" Note: {resolution_note}"
        self.send_message(
            sender_role="system",
            sender_id=0,
            message=msg,
            recipients=[user_id],
            channels=("dashboard", "email"),
            subject="Issue Resolved",
            notif_type="issue",
        )
        # Also create a broadcast for admins/instructors if needed (not by default)

    # -----------------------------
    # User-facing fetch APIs
    # -----------------------------
    def fetch_user_notifications(
        self,
        user_id: int,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return personal notifications for a user (Notification table)."""
        return self.repo.fetch_user_notifications(user_id, status, limit, offset)

    def fetch_broadcasts_for_role(
        self, role: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Return broadcast rows relevant to role (e.g., student -> all_students & all_users)."""
        role_map = {
            "student": ["all_students", "all_users"],
            "instructor": ["all_instructors", "all_users"],
            "admin": ["all_admins", "all_users"],
            "user": ["all_users"],
        }
        broadcast_types = role_map.get(role, ["all_users"])
        result = self.repo.fetch_broadcasts_for_types(broadcast_types, limit, offset)

        return result

    def mark_notification_read(self, notification_id: int):
        try:
            self.repo.mark_as_read(notification_id)
        except Exception:
            logger.exception("mark_notification_read failed for %s", notification_id)
