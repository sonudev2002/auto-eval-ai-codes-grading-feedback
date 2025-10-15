# backend/grade_distribution.py
"""
Grade Distribution Manager
--------------------------
Manages grade distribution data across users and assignments.
Handles:
- Grade count storage per user (insert/update/reset)
- Aggregation and trend analysis
- Chart-ready data for visualizations
"""

from collections import defaultdict
from datetime import datetime, date
from backend.db import get_connection
from typing import Dict


class GradeDistributionManager:
    """
    Central manager for grade distribution.
    Responsibilities:
    - Insert, update, and read grade distributions
    - Provide chart-ready data for visualizations
    - Analyze grade trends over time from submissions
    """

    def __init__(self):
        # Map of letter grades to corresponding database columns
        self.grade_map = {
            "A": "grade_a",
            "B": "grade_b",
            "C": "grade_c",
            "D": "grade_d",
            "E": "grade_e",
            "F": "grade_f",
        }

    # ===========================================================
    # 🔹 INTERNAL UTILITIES — Manage database connections safely
    # ===========================================================
    def _fetchone(self, query, params=None):
        """
        Execute a query and return a single record.
        Raises an exception on DB error.
        """
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute(query, params or [])
                row = cur.fetchone()
                cur.close()
                return row
        except Exception as e:
            print(f"[DB ERROR] _fetchone failed: {e}")
            raise

    def _fetchall(self, query, params=None):
        """
        Execute a query and return all rows as a list.
        Raises an exception on DB error.
        """
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute(query, params or [])
                rows = cur.fetchall()
                cur.close()
                return rows
        except Exception as e:
            print(f"[DB ERROR] _fetchall failed: {e}")
            raise

    def _execute(self, query, params=None):
        """
        Execute a write query (INSERT, UPDATE, DELETE).
        Commits automatically. Raises on error.
        """
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute(query, params or [])
                conn.commit()
                cur.close()
        except Exception as e:
            print(f"[DB ERROR] _execute failed: {e}")
            raise

    def _ensure_distribution_row(self, related_id: int):
        """
        Ensure that a grade_distribution record exists for the given user.
        If missing, insert a zeroed-out distribution row.
        """
        row = self._fetchone(
            "SELECT distribution_id FROM grade_distribution WHERE related_id = %s",
            (related_id,),
        )
        if not row:
            self._execute(
                """
                INSERT INTO grade_distribution (related_id, grade_a, grade_b, grade_c, grade_d, grade_e, grade_f)
                VALUES (%s, 0, 0, 0, 0, 0, 0)
                """,
                (related_id,),
            )

    # ===========================================================
    # 🔹 INSERT / UPDATE OPERATIONS
    # ===========================================================
    def update_distribution(self, related_id: int, grade: str):
        """
        Increment the count for a specific grade belonging to a user.

        Args:
            related_id (int): ID of the student/instructor.
            grade (str): Letter grade (A–F) to increment.
        """
        if grade not in self.grade_map:
            return  # Ignore invalid grades

        # Ensure the record exists before updating
        self._ensure_distribution_row(related_id)
        column = self.grade_map[grade]

        # Increment selected grade column
        self._execute(
            f"UPDATE grade_distribution SET {column} = {column} + 1 WHERE related_id = %s",
            (related_id,),
        )

    def reset_distribution(self, related_id: int):
        """
        Reset all grade counts for a specific user to zero.

        Args:
            related_id (int): ID of the student/instructor.
        """
        self._ensure_distribution_row(related_id)

        # Reset all grade columns to 0
        self._execute(
            """
            UPDATE grade_distribution
            SET grade_a=0, grade_b=0, grade_c=0,
                grade_d=0, grade_e=0, grade_f=0
            WHERE related_id = %s
            """,
            (related_id,),
        )

    # ===========================================================
    # 🔹 READ OPERATIONS
    # ===========================================================
    def get_distribution(self, related_id: int):
        """
        Retrieve the grade distribution for a specific user.
        Args: related_id (int): User ID (student or instructor).
        Returns: dict: { 'A': count, 'B': count, ..., 'F': count }
        """
        row = self._fetchone(
            """
            SELECT grade_a, grade_b, grade_c, grade_d, grade_e, grade_f
            FROM grade_distribution
            WHERE related_id = %s
            """,
            (related_id,),
        )

        # If user has no record, return zeroed distribution
        if not row:
            return {g: 0 for g in self.grade_map.keys()}

        return dict(zip(self.grade_map.keys(), row))

    def get_all_distributions(self):
        """
        Retrieve grade distributions for all users.

        Returns:
            dict[user_id] = { 'A': count, ..., 'F': count }
        """
        rows = self._fetchall(
            """
            SELECT related_id, grade_a, grade_b, grade_c, grade_d, grade_e, grade_f
            FROM grade_distribution
            """
        )

        result = {}
        for related_id, a, b, c, d, e, f in rows:
            result[related_id] = {
                "A": a,
                "B": b,
                "C": c,
                "D": d,
                "E": e,
                "F": f,
            }
        return result

    def get_overall_distribution(self):
        """
        Retrieve the total grade distribution across all users.

        Returns:
            dict: { 'A': total_A, 'B': total_B, ..., 'F': total_F }
        """
        row = self._fetchone(
            """
            SELECT SUM(grade_a), SUM(grade_b), SUM(grade_c),
                   SUM(grade_d), SUM(grade_e), SUM(grade_f)
            FROM grade_distribution
            """
        )

        # Return zeros if no data found
        if not row:
            return {g: 0 for g in self.grade_map.keys()}

        return dict(zip(self.grade_map.keys(), row))

    # -------------------------------
    # 🔹 Trend Analysis
    # -------------------------------
    def get_trend_over_time(self, interval="day"):
        """
        Returns grade counts over time from code_submission table.
        Interval: day / week / month
        """
        date_format = {"day": "%Y-%m-%d", "week": "%Y-%u", "month": "%Y-%m"}.get(
            interval, "%Y-%m-%d"
        )

        rows = self._fetchall(
            f"""
            SELECT DATE_FORMAT(cs.submitted_on, %s) AS period,
       ce.grade
FROM code_submission cs
JOIN code_evaluation ce ON cs.submission_id = ce.submission_id
WHERE ce.grade IS NOT NULL
ORDER BY cs.submitted_on

        """,
            (date_format,),
        )

        trend = {}
        for period, grade in rows:
            if period not in trend:
                trend[period] = {g: 0 for g in self.grade_map.keys()}
            if grade in trend[period]:
                trend[period][grade] += 1

        return trend

    # -------------------------------
    # 🔹 Chart Outputs
    # -------------------------------
    def get_chart_data_distribution(self, related_id: int):
        dist = self.get_distribution(related_id)
        return {"labels": list(dist.keys()), "values": list(dist.values())}

    def get_chart_data_overall(self):
        dist = self.get_overall_distribution()
        return {"labels": list(dist.keys()), "values": list(dist.values())}

    def get_chart_data_trend(self, interval="month"):
        date_format = {"day": "%Y-%m-%d", "week": "%Y-%u", "month": "%Y-%m"}.get(
            interval, "%Y-%m"
        )

        rows = self._fetchall(
            f"""
            SELECT DATE_FORMAT(cs.submitted_on, %s) as period,
                ce.grade, COUNT(*) as count
            FROM code_evaluation ce
            JOIN code_submission cs ON ce.submission_id = cs.submission_id
            WHERE ce.grade IS NOT NULL
            GROUP BY period, ce.grade
            ORDER BY period
        """,
            (date_format,),
        )

        trend = {}
        for period, grade, count in rows:
            if period not in trend:
                trend[period] = {g: 0 for g in self.grade_map.keys()}
            if grade in trend[period]:
                trend[period][grade] += int(count)

        labels = sorted(trend.keys())
        datasets = {g: [trend[t][g] for t in labels] for g in self.grade_map.keys()}

        return {"labels": labels, "datasets": datasets}

    # -------------------------------
    # 🔹 Generic Group Chart Helper
    # -------------------------------
    def get_group_distribution_charts(self, role: str):
        """
        Return grade distributions grouped by user (students or instructors).
        Only users that have grade_distribution rows will be included.
        """
        if role not in ("student", "instructor"):
            raise ValueError("Role must be either 'student' or 'instructor'")

        rows = self._fetchall(
            """
            SELECT gd.related_id,
                gd.grade_a, gd.grade_b, gd.grade_c, gd.grade_d, gd.grade_e, gd.grade_f
            FROM grade_distribution gd
            JOIN user_profile u ON gd.related_id = u.user_id
            WHERE u.role = %s
            """,
            (role,),
        )

        result = {}
        for user_id, a, b, c, d, e, f in rows:
            result[user_id] = {
                "labels": list(self.grade_map.keys()),
                "values": [
                    int(a or 0),
                    int(b or 0),
                    int(c or 0),
                    int(d or 0),
                    int(e or 0),
                    int(f or 0),
                ],
            }
        return result

    def recalculate_all_from_evaluations(self) -> bool:
        """
        Recalculate grade distribution for every user from Code_Evaluation scores.
        This will upsert rows in grade_distribution to reflect the current state.
        """
        try:
            cur = self.conn.cursor(dictionary=True)
            cur.execute(
                """
                SELECT u.user_id AS user_id, ce.score
                FROM code_submission cs
                JOIN code_evaluation ce ON cs.submission_id = ce.submission_id
                JOIN user_profile u ON cs.user_id = u.user_id
                """
            )
            rows = cur.fetchall()

            per_user = {}
            for r in rows:
                uid = r["user_id"]
                score = r.get("score")
                if uid not in per_user:
                    per_user[uid] = {g: 0 for g in self.grade_map.keys()}
                if score is None:
                    continue
                if score >= 90:
                    per_user[uid]["A"] += 1
                elif score >= 80:
                    per_user[uid]["B"] += 1
                elif score >= 70:
                    per_user[uid]["C"] += 1
                elif score >= 60:
                    per_user[uid]["D"] += 1
                elif score >= 50:
                    per_user[uid]["E"] += 1
                else:
                    per_user[uid]["F"] += 1

            for uid, counts in per_user.items():
                self._execute(
                    """
                    INSERT INTO grade_distribution
                    (related_id, grade_a, grade_b, grade_c, grade_d, grade_e, grade_f)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                    grade_a=VALUES(grade_a),
                    grade_b=VALUES(grade_b),
                    grade_c=VALUES(grade_c),
                    grade_d=VALUES(grade_d),
                    grade_e=VALUES(grade_e),
                    grade_f=VALUES(grade_f)
                    """,
                    (
                        uid,
                        counts["A"],
                        counts["B"],
                        counts["C"],
                        counts["D"],
                        counts["E"],
                        counts["F"],
                    ),
                )
            return True
        except Exception as e:
            print("recalculate_all_from_evaluations failed", e)
            return False

    def search_distribution(self, role: str, identifier: str):
        """
        Fetch grade distribution for a specific user (student or instructor)
        by user_id or email.
        """

        if role not in ("student", "instructor"):
            raise ValueError("Role must be 'student' or 'instructor'")
        # Detect if identifier is numeric user_id or email
        query = (
            "SELECT gd.grade_a, gd.grade_b, gd.grade_c, gd.grade_d, gd.grade_e, gd.grade_f "
            "FROM grade_distribution gd "
            "JOIN user_profile u ON gd.related_id = u.user_id "
            "WHERE u.role = %s AND (u.user_id = %s OR u.email = %s)"
        )
        row = self._fetchone(query, (role, identifier, identifier))

        if not row:
            return None

        a, b, c, d, e, f = row
        return {
            "A": a or 0,
            "B": b or 0,
            "C": c or 0,
            "D": d or 0,
            "E": e or 0,
            "F": f or 0,
        }

    def get_aggregated_distribution(self, role: str):
        """
        Aggregated grade distribution for all users of a given role.
        Example: role="student" -> totals across all students
        """
        if role not in ("student", "instructor"):
            raise ValueError("Role must be 'student' or 'instructor'")

        row = self._fetchone(
            """
            SELECT COALESCE(SUM(grade_a),0),
                COALESCE(SUM(grade_b),0),
                COALESCE(SUM(grade_c),0),
                COALESCE(SUM(grade_d),0),
                COALESCE(SUM(grade_e),0),
                COALESCE(SUM(grade_f),0)
            FROM grade_distribution gd
            JOIN user_profile u ON gd.related_id = u.user_id
            WHERE u.role = %s
            """,
            (role,),
        )
        if not row:
            return {g: 0 for g in self.grade_map.keys()}
        a, b, c, d, e, f = row
        return {
            "A": int(a),
            "B": int(b),
            "C": int(c),
            "D": int(d),
            "E": int(e),
            "F": int(f),
        }

    def get_assignment_distribution(self, assignment_id: int):
        """
        Return grade distribution for a particular assignment.
        Uses best score per student → derives grade like admin dashboard.
        """
        rows = self._fetchall(
            """
            SELECT u.user_id, MAX(ce.score) as best_score
            FROM code_submission cs
            JOIN code_evaluation ce ON cs.submission_id = ce.submission_id
            JOIN user_profile u ON cs.user_id = u.user_id
            WHERE cs.assignment_id = %s
            GROUP BY u.user_id
            """,
            (assignment_id,),
        )

        dist = {g: 0 for g in self.grade_map.keys()}

        for _, score in rows:
            if score is None:
                continue
            if score >= 90:
                dist["A"] += 1
            elif score >= 80:
                dist["B"] += 1
            elif score >= 70:
                dist["C"] += 1
            elif score >= 60:
                dist["D"] += 1
            elif score >= 50:
                dist["E"] += 1
            else:
                dist["F"] += 1

        return dist

    def get_chart_data_assignment(self, assignment_id: int):
        dist = self.get_assignment_distribution(assignment_id)
        return {"labels": list(dist.keys()), "values": list(dist.values())}
