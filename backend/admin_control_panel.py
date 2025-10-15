"""
admin_control_panel.py
----------------------
Provides backend logic for the admin dashboard and control panel.
Includes:
- User management and deletion with audit tracking
- Login logs, notifications, and reported issue retrieval
- Broadcast and system-wide message handling
- Role-based data visibility for admin users
"""

from backend.db import get_connection


class AdminControlPanel:
    """Provides data access and management functions for admin operations."""

    # ============================================================
    # 🔍 User Management Utilities
    # ============================================================
    @staticmethod
    def get_user_id_by_email(email: str) -> int | None:
        """Return user_id for a given email, or None if not found."""
        conn = get_connection()
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT user_id FROM user_profile WHERE email=%s", (email,))
            row = cursor.fetchone()
            return row["user_id"] if row else None

    # ============================================================
    # 🕓 Login & Notification Logs
    # ============================================================
    @staticmethod
    def get_login_logs(limit: int = 20):
        """Fetch recent login activity for all users."""
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
        """Fetch latest system notifications with user info."""
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
        """Fetch recent broadcast messages."""
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

    # ============================================================
    # ⚠️ Reported Issues
    # ============================================================
    @staticmethod
    def get_reported_issues(limit: int = 50, status_group: str | None = None):
        """
        Fetch reported issues filtered by status.
        status_group:
            - None → All issues
            - "open" → Only OPEN issues
            - "other" → In progress / pending
            - "closed" → RESOLVED / CLOSED / REJECTED
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
        """Fetch all notifications for a specific user."""
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
        """Fetch all reported issues submitted by a specific user."""
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

    # ============================================================
    # 🗑️ User Deletion (Admin Only)
    # ============================================================
    @staticmethod
    def delete_user_and_data(
        user_id: int, acting_admin_id: int = 1
    ) -> tuple[bool, str]:
        """
        Delete user and dependent data safely.
        Rules:
          - Prevent self-deletion of main admin
          - Reassign instructor assignments to acting admin
          - Log action in admin_audit_log (if exists)
          - Remove orphaned address rows
        Returns: (success, message)
        """
        if int(user_id) == int(acting_admin_id):
            return False, "Cannot delete primary admin user"

        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # 1️⃣ Check user existence and role
                cursor.execute(
                    "SELECT role, address_id FROM user_profile WHERE user_id = %s",
                    (user_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return False, "User not found"

                # Support tuple or dict cursor output
                role = row[0] if isinstance(row, (list, tuple)) else row.get("role")
                address_id = (
                    row[1]
                    if isinstance(row, (list, tuple))
                    else row.get("address_id", None)
                )

                # 2️⃣ If instructor → reassign their assignments
                if str(role).lower() == "instructor":
                    cursor.execute(
                        "UPDATE assignment SET instructor_id = %s WHERE instructor_id = %s",
                        (acting_admin_id, user_id),
                    )

                # 3️⃣ Log deletion in admin_audit_log
                try:
                    cursor.execute(
                        """
                        INSERT INTO admin_audit_log (admin_id, action, target_user_id, details, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        """,
                        (
                            acting_admin_id,
                            "delete_user",
                            user_id,
                            f"reassigned_instructor={role=='instructor'}",
                        ),
                    )
                except Exception:
                    # Table may not exist — safe to ignore
                    pass

                # 4️⃣ Delete user (cascade removes related data)
                cursor.execute(
                    "DELETE FROM user_profile WHERE user_id = %s", (user_id,)
                )

                # 5️⃣ Delete orphaned address if no references
                if address_id:
                    try:
                        cursor.execute(
                            "SELECT COUNT(*) FROM user_profile WHERE address_id = %s",
                            (address_id,),
                        )
                        cnt = cursor.fetchone()
                        count_val = (
                            cnt[0]
                            if isinstance(cnt, (list, tuple))
                            else next(iter(cnt.values()))
                        )
                        if int(count_val) == 0:
                            cursor.execute(
                                "DELETE FROM address WHERE address_id = %s",
                                (address_id,),
                            )
                    except Exception:
                        pass

                conn.commit()
                return True, "User deleted successfully and related data cleaned up"
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            return False, f"Error deleting user: {str(e)}"
        finally:
            conn.close()
