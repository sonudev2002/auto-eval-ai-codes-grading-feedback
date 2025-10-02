from backend.db import get_connection


class AdminControlPanel:
    """Handles fetching data for Admin Dashboard (Login Logs, Notifications, Reported Issues)."""

    @staticmethod
    def get_user_id_by_email(email: str) -> int | None:
        conn = get_connection()
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT user_id FROM user_profile WHERE email=%s", (email,))
            row = cursor.fetchone()
            return row["user_id"] if row else None

    @staticmethod
    def get_login_logs(limit: int = 20):
        conn = get_connection()
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(
                """
                SELECT l.*, u.first_name, u.last_name
                FROM login_log l
                JOIN user_profile u ON l.user_id = u.user_id
                ORDER BY l.login_time DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cursor.fetchall()

    @staticmethod
    def get_notifications(limit: int = 20):
        conn = get_connection()
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(
                """
                SELECT n.*, u.first_name, u.last_name
                FROM notification n
                JOIN user_profile u ON n.user_id = u.user_id
                ORDER BY n.created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cursor.fetchall()

    @staticmethod
    def get_broadcasts(limit: int = 20):
        conn = get_connection()
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(
                """
                SELECT b.*
                FROM broadcast_notification b
                ORDER BY b.created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cursor.fetchall()

    @staticmethod
    def get_reported_issues(limit: int = 50, status_group: str | None = None):
        """
        Fetch reported issues.
        status_group options:
          - None      → All issues (default)
          - "open"    → Only OPEN issues
          - "other"   → Not OPEN and not resolved/closed/rejected
          - "closed"  → RESOLVED / CLOSED / REJECTED
        """
        conn = get_connection()
        with conn.cursor(dictionary=True) as cursor:
            where_clause = ""

            if status_group == "open":
                where_clause = "WHERE r.status = 'OPEN'"
            elif status_group == "other":
                where_clause = (
                    "WHERE r.status NOT IN ('OPEN','RESOLVED','CLOSED','REJECTED')"
                )
            elif status_group == "closed":
                where_clause = "WHERE r.status IN ('RESOLVED','CLOSED','REJECTED')"

            query = f"""
                SELECT r.*, u.first_name, u.last_name
                FROM reported_issue r
                JOIN user_profile u ON r.user_id = u.user_id
                {where_clause}
                ORDER BY r.reported_at DESC
                LIMIT %s
            """
            cursor.execute(query, (limit,))
            return cursor.fetchall()

    @staticmethod
    def get_notifications_by_user(user_id: int, limit: int = 50):
        """Fetch notifications for a specific user."""
        conn = get_connection()
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(
                """
                SELECT n.*, u.first_name, u.last_name
                FROM notification n
                JOIN user_profile u ON n.user_id = u.user_id
                WHERE n.user_id = %s
                ORDER BY n.created_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            return cursor.fetchall()

    @staticmethod
    def get_reported_issues_by_user(user_id: int, limit: int = 50):
        """Fetch reported issues for a specific user."""
        conn = get_connection()
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(
                """
                SELECT r.*, u.first_name, u.last_name
                FROM reported_issue r
                JOIN user_profile u ON r.user_id = u.user_id
                WHERE r.user_id = %s
                ORDER BY r.reported_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            return cursor.fetchall()
