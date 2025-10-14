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

    @staticmethod
    def delete_user_and_data(
        user_id: int, acting_admin_id: int = 1
    ) -> tuple[bool, str]:
        """
        Safely delete a user:
         - Prevent deleting primary admin (user_id == acting_admin_id)
         - If the user is an instructor, reassign assignments to acting_admin_id
         - Insert one audit log record
         - Delete the user row (DB ON DELETE CASCADE should remove dependent rows)
        Returns (success, message)
        """
        # Protect main admin
        if int(user_id) == int(acting_admin_id):
            return False, "Cannot delete primary admin user"

        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # 1. Verify existence and role
                cursor.execute(
                    "SELECT role, address_id FROM user_profile WHERE user_id = %s",
                    (user_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return False, "User not found"

                role = (
                    row[0] if isinstance(row, (list, tuple)) else row.get("role")
                )  # handle dict/tuple
                address_id = None
                if isinstance(row, (list, tuple)):
                    # if cursor returns tuple, guess address_id position (fallback)
                    try:
                        address_id = row[1]
                    except Exception:
                        address_id = None
                else:
                    address_id = row.get("address_id")

                # 2. If instructor, reassign their assignments to admin (acting_admin_id)
                if str(role).lower() == "instructor":
                    cursor.execute(
                        "UPDATE assignment SET instructor_id = %s WHERE instructor_id = %s",
                        (acting_admin_id, user_id),
                    )

                # 3. Audit log (create table admin_audit_log if not exists)
                try:
                    cursor.execute(
                        "INSERT INTO admin_audit_log (admin_id, action, target_user_id, details, created_at) "
                        "VALUES (%s, %s, %s, %s, NOW())",
                        (
                            acting_admin_id,
                            "delete_user",
                            user_id,
                            f"reassigned_instructor={role=='instructor'}",
                        ),
                    )
                except Exception:
                    # swallow — table might not exist; we don't want to block deletion
                    pass

                # 4. Delete user (let DB cascades remove dependent rows)
                cursor.execute(
                    "DELETE FROM user_profile WHERE user_id = %s", (user_id,)
                )

                # 5. (optional) remove address row if orphaned
                if address_id:
                    try:
                        cursor.execute(
                            "SELECT COUNT(*) FROM user_profile WHERE address_id = %s",
                            (address_id,),
                        )
                        cnt = cursor.fetchone()
                        # supports tuple/dict
                        count_val = (
                            cnt[0]
                            if isinstance(cnt, (list, tuple))
                            else (
                                cnt.get("COUNT(*)")
                                if isinstance(cnt, dict)
                                else list(cnt.values())[0]
                            )
                        )
                        if int(count_val) == 0:
                            cursor.execute(
                                "DELETE FROM address WHERE address_id = %s",
                                (address_id,),
                            )
                    except Exception:
                        pass

                conn.commit()
                return True, "User deleted and related data cleaned up"
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            return False, f"Error deleting user: {str(e)}"
        finally:
            conn.close()
