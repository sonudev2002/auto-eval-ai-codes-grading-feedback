import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, cast
from decimal import Decimal
from datetime import date, datetime, timedelta, time
import time as time_module
from typing import Any, Mapping, Sequence
from db import get_connection
from apscheduler.schedulers.background import BackgroundScheduler
from grade_distribution import GradeDistributionManager

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def normalize_row(row: dict | None) -> dict | None:
    """Convert DB row values into JSON/template-safe types."""
    if row is None:
        return None

    clean: dict = {}
    for key, value in row.items():
        if isinstance(value, Decimal):
            clean[key] = float(value)
        elif isinstance(value, (date, datetime)):
            clean[key] = value.isoformat()
        elif isinstance(value, time):  # datetime.time
            clean[key] = value.strftime("%H:%M:%S")
        elif isinstance(value, timedelta):
            clean[key] = str(value)
        else:
            clean[key] = value
    return clean


def safe_float(x: Any) -> float:
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, (date, datetime)):
        return float(x.toordinal())
    if isinstance(x, timedelta):
        return x.total_seconds()
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------
# DB transaction helper
# ---------------------------


class DBConnection:
    """Context manager for MySQL DB connection with dictionary cursor."""

    def __init__(self) -> None:
        self.conn = None
        self.cursor = None

    def __enter__(self):
        from db import get_connection

        self.conn = get_connection()
        # ✅ Force dictionary cursor
        self.cursor = self.conn.cursor(dictionary=True)
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
        except Exception:
            pass


# ---------------------------
# Base analytics utilities
# ---------------------------
class BaseAnalytics:
    """Reusable helpers (safe wrapper using DBConnection context manager)."""

    def fetch_one(self, query: str, params: tuple = (), default: Any = None) -> Any:
        try:
            with DBConnection() as cursor:
                cursor.execute(query, params or ())
                row = cursor.fetchone()
                return row if row else default
        except Exception:
            logger.exception("fetch_one failed")
            return default

    def fetch_all(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        try:
            with DBConnection() as cursor:
                cursor.execute(query, params or ())
                return cursor.fetchall()
        except Exception:
            logger.exception("fetch_all failed")
            return []

    def execute(self, query: str, params: tuple = ()) -> bool:
        try:
            with DBConnection() as cursor:
                cursor.execute(query, params or ())
                cursor.connection.commit()
            return True
        except Exception:
            logger.exception("execute failed")
            try:
                if cursor and cursor.connection:
                    cursor.connection.rollback()
            except Exception:
                pass
            return False

    def close(self):
        # Nothing to do, connections are auto-closed by DBConnection
        pass


# ---------------------------
# System Analytics
# ---------------------------
class SystemAnalytics(BaseAnalytics):
    """Collects and saves system-wide snapshot."""

    @staticmethod
    def _grade_distribution_from_scores(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        dist = {g: 0 for g in ["A", "B", "C", "D", "E", "F"]}
        for r in rows:
            s = r.get("score")
            if s is None:
                continue
            if s >= 90:
                dist["A"] += 1
            elif s >= 80:
                dist["B"] += 1
            elif s >= 70:
                dist["C"] += 1
            elif s >= 60:
                dist["D"] += 1
            elif s >= 50:
                dist["E"] += 1
            else:
                dist["F"] += 1
        return dist

    def collect_data(self) -> Dict[str, Any]:
        """Gather live system metrics (safe to call on demand)."""
        total_students = self.fetch_one(
            "SELECT COUNT(*) AS cnt FROM user_profile WHERE role='student'",
            (),
            {"cnt": 0},
        )["cnt"]
        total_instructors = self.fetch_one(
            "SELECT COUNT(*) AS cnt FROM user_profile WHERE role='instructor'",
            (),
            {"cnt": 0},
        )["cnt"]
        total_assignments = self.fetch_one(
            "SELECT COUNT(*) AS cnt FROM assignment", (), {"cnt": 0}
        )["cnt"]
        active_users_today = self.fetch_one(
            "SELECT COUNT(DISTINCT user_id) AS cnt FROM login_log WHERE DATE(login_time)=CURDATE()",
            (),
            {"cnt": 0},
        )["cnt"]
        avg_score_row = self.fetch_one(
            "SELECT AVG(score) AS avg FROM code_evaluation", (), {"avg": 0.0}
        )
        avg_score = round((avg_score_row["avg"] or 0.0), 2)
        total_submissions = self.fetch_one(
            "SELECT COUNT(*) AS cnt FROM code_submission", (), {"cnt": 0}
        )["cnt"]
        new_users_last_week = self.fetch_one(
            "SELECT COUNT(*) AS cnt FROM user_profile WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)",
            (),
            {"cnt": 0},
        )["cnt"]

        scores = self.fetch_all("SELECT score FROM code_evaluation")
        grade_distribution = self._grade_distribution_from_scores(scores)

        return {
            "total_students": total_students,
            "total_instructors": total_instructors,
            "total_assignments": total_assignments,
            "active_users_today": active_users_today,
            "average_score": avg_score,
            "total_submissions": total_submissions,
            "new_users_last_week": new_users_last_week,
            "grade_distribution": grade_distribution,
        }

    def save_snapshot(self, data: Dict[str, Any]) -> bool:
        try:
            with DBConnection() as cur:
                cur.execute(
                    """
                    INSERT INTO system_statistics
                    (total_students, total_instructors, total_assignments, active_users_today,
                     average_score, total_submissions, new_users_last_week, grade_distribution)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        data["total_students"],
                        data["total_instructors"],
                        data["total_assignments"],
                        data["active_users_today"],
                        data["average_score"],
                        data["total_submissions"],
                        data["new_users_last_week"],
                        json.dumps(data["grade_distribution"]),
                    ),
                )
            logger.info("System snapshot saved")
            return True
        except Exception:
            logger.exception("save_snapshot failed")
            return False

    def fetch_latest_snapshot(self) -> dict | None:
        """Fetch the most recent system statistics snapshot from DB."""
        try:
            with DBConnection() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM system_statistics
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                if not row:
                    return None
                # convert DB row to plain dict
                result = dict(row)
                result = cast(Dict[str, Any], result)
                # Ensure JSON safe
                grade_raw = result.get("grade_distribution")
                if isinstance(grade_raw, str):
                    try:
                        result["grade_distribution"] = json.loads(grade_raw)
                    except Exception:
                        result["grade_distribution"] = {}
                elif isinstance(grade_raw, dict):
                    result["grade_distribution"] = grade_raw
                else:
                    result["grade_distribution"] = {}
                return result
        except Exception:
            logger.exception("fetch_latest_snapshot failed")
            return None

    def fetch_all_snapshots(self) -> list[dict]:
        """Fetch all system snapshots ordered by time ascending."""
        rows = self.fetch_all(
            """
                SELECT *
                FROM system_statistics
                ORDER BY timestamp ASC
                """
        )
        for row in rows:
            if isinstance(row.get("grade_distribution"), str):
                try:
                    row["grade_distribution"] = json.loads(row["grade_distribution"])
                except Exception:
                    row["grade_distribution"] = {}
        return rows


# ---------------------------
# Assignment Analytics
# ---------------------------
class AssignmentAnalyticsService:
    """Calculate and maintain per-assignment analytics."""

    CACHE_FILE = "assignment_analytics_cache.json"

    def __init__(self):
        # simple cache file storing last-updated timestamps
        self._ensure_cache_file()

    def _ensure_cache_file(self):
        if not os.path.exists(self.CACHE_FILE):
            with open(self.CACHE_FILE, "w") as f:
                json.dump({}, f)

    def _load_cache(self) -> Dict[str, str]:
        with open(self.CACHE_FILE, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return {}

    def _save_cache(self, data: Dict[str, str]):
        with open(self.CACHE_FILE, "w") as f:
            json.dump(data, f)

    def calculate_for(self, assignment_id: int) -> Optional[Dict[str, Any]]:
        """Calculate analytics for a single assignment. Returns None if no submissions."""
        with DBConnection() as cur:
            cur.execute(
                "SELECT COUNT(*) AS total FROM code_submission WHERE assignment_id=%s",
                (assignment_id,),
            )
            total_row = cur.fetchone()
            total = total_row["total"] if total_row else 0
            if total == 0:
                return None

            cur.execute(
                """
                SELECT
                  AVG(ce.score) AS avg_score,
                  SUM(CASE WHEN ce.score >= 40 THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0) * 100 AS pass_percent,
                  SUM(CASE WHEN ce.plagiarism_score > 50 THEN 1 ELSE 0 END) AS plagiarism_cases,
                  AVG(ce.average_execution_time) AS avg_time
                FROM code_evaluation ce
                JOIN code_submission cs ON ce.submission_id = cs.submission_id
                WHERE cs.assignment_id = %s
                """,
                (assignment_id,),
            )
            agg = cur.fetchone() or {}
            # most common feedback / error (if any)
            cur.execute(
                """
                SELECT feedback, COUNT(*) AS cnt
                FROM code_evaluation ce
                JOIN code_submission cs ON ce.submission_id = cs.submission_id
                WHERE cs.assignment_id = %s AND ce.feedback IS NOT NULL
                GROUP BY feedback ORDER BY cnt DESC LIMIT 1
                """,
                (assignment_id,),
            )
            common = cur.fetchone()
            most_common_error = common["feedback"] if common else None

            return {
                "assignment_id": assignment_id,
                "total_submission": int(total),
                "average_score": float(agg.get("avg_score") or 0.0),
                "pass_percentage": float(agg.get("pass_percent") or 0.0),
                "plagiarism_cases": int(agg.get("plagiarism_cases") or 0),
                "average_time": float(agg.get("avg_time") or 0.0),
                "most_common_error": most_common_error or "N/A",
            }

    def upsert(self, payload: Dict[str, Any]) -> bool:
        """Insert or update assignment analytics row."""
        with DBConnection() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO assignment_analytics
                    (assignment_id, total_submission, average_score, plagiarism_cases,
                     pass_percentage, average_time, most_common_error)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                      total_submission=VALUES(total_submission),
                      average_score=VALUES(average_score),
                      plagiarism_cases=VALUES(plagiarism_cases),
                      pass_percentage=VALUES(pass_percentage),
                      average_time=VALUES(average_time),
                      most_common_error=VALUES(most_common_error)
                    """,
                    (
                        payload["assignment_id"],
                        payload["total_submission"],
                        payload["average_score"],
                        payload["plagiarism_cases"],
                        payload["pass_percentage"],
                        payload["average_time"],
                        payload["most_common_error"],
                    ),
                )
                # update cache
                cache = self._load_cache()
                cache[str(payload["assignment_id"])] = time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self._save_cache(cache)
                return True
            except Exception:
                logger.exception("AssignmentAnalytics.upsert failed")
                return False

    def update_assignment(self, assignment_id: int) -> bool:
        stats = self.calculate_for(assignment_id)
        if not stats:
            # ensure row exists? We'll keep it absent if no submissions.
            logger.info("No submissions for assignment %s", assignment_id)
            return False
        return self.upsert(stats)

    def fetch_assignment(self, assignment_id: int) -> Optional[Dict[str, Any]]:
        with DBConnection() as cur:
            cur.execute(
                "SELECT * FROM assignment_analytics WHERE assignment_id=%s",
                (assignment_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        cache = self._load_cache()
        row["last_updated"] = cache.get(str(assignment_id), "Never")
        return row

    def update_all(self) -> Dict[int, bool]:
        result = {}
        with DBConnection() as cur:
            cur.execute("SELECT assignment_id FROM assignment")
            assignments = cur.fetchall()
        for a in assignments:
            aid = a["assignment_id"]
            ok = self.update_assignment(aid)
            result[aid] = ok
        return result


# ---------------------------
# Grade Distribution Analytics
# ---------------------------
class GradeDistributionAnalytics:
    """
    Wrapper to preserve compatibility. Internally delegates to GradeDistributionManager.
    """

    def __init__(self):
        self.manager = GradeDistributionManager()

    def get_user_distribution(self, user_id: int):
        return self.manager.get_distribution(user_id)

    def get_all_distributions(self):
        return self.manager.get_all_distributions()

    def get_overall_distribution(self):
        return self.manager.get_overall_distribution()

    def increment_user_grade(self, user_id: int, grade_col: str):
        # Map back from col to grade letter
        reverse_map = {v: k for k, v in self.manager.grade_map.items()}
        grade = reverse_map.get(grade_col.upper())
        if not grade:
            return False
        self.manager.update_distribution(user_id, grade)
        return True

    def recalculate_all_from_evaluations(self):
        return self.manager.recalculate_all_from_evaluations()


# ---------------------------
# Difficulty Analytics (Student / Instructor)
# ---------------------------
class DifficultyAnalyticsBase:
    """
    Shared implementation for Student_Difficulty_Stats and Instructor_Difficulty_Stats.
    The concrete classes will call with the right table name.
    """

    def __init__(self, table_name: str):
        self.table = table_name

    def get_user_level(self, user_id: int, level_id: int) -> Optional[Dict[str, Any]]:
        with DBConnection() as cur:
            cur.execute(
                f"SELECT * FROM {self.table} WHERE user_id=%s AND difficulty_level=%s",
                (user_id, level_id),
            )
            return cur.fetchone()

    def get_user_all_levels(self, user_id: int) -> List[Dict[str, Any]]:
        with DBConnection() as cur:
            cur.execute(
                f"""
                SELECT dl.difficulty_types, s.*
                FROM {self.table} s
                JOIN difficulty_level dl ON s.difficulty_level = dl.level_id
                WHERE s.user_id=%s
                """,
                (user_id,),
            )
            return cur.fetchall()

    def get_all_users_at_level(self, level_id: int) -> List[Dict[str, Any]]:
        with DBConnection() as cur:
            cur.execute(
                f"""
                SELECT u.user_id, u.first_name, u.last_name, s.*
                FROM {self.table} s
                JOIN user_profile u ON s.user_id=u.user_id
                WHERE s.difficulty_level = %s
                """,
                (level_id,),
            )
            return cur.fetchall()

    def get_all_users_all_levels(self) -> List[Dict[str, Any]]:
        with DBConnection() as cur:
            cur.execute(
                f"""
                SELECT u.user_id, u.first_name, u.last_name, dl.difficulty_types, s.*
                FROM {self.table} s
                JOIN user_profile u ON s.user_id=u.user_id
                JOIN Difficulty_Level dl ON s.difficulty_level=dl.level_id
                """
            )
            return cur.fetchall()

    def _calculate_level_metrics_for_user(
        self, user_id: int, level_id: int
    ) -> Dict[str, Any]:
        """
        Returns a dict with assignment_count, average_score, average_pass_rate, average_feedback_score
        based on assignments of that difficulty for the given user.
        """
        with DBConnection() as cur:
            cur.execute(
                """
                SELECT
                  COUNT(DISTINCT a.assignment_id) AS assignment_count,
                  AVG(ce.score) AS average_score,
                  SUM(CASE WHEN ce.score >= 40 THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0) * 100 AS average_pass_rate,
                  AVG(fs.feedback_score) AS average_feedback_score
                FROM assignment a
                JOIN code_submission cs ON a.assignment_id = cs.assignment_id
                LEFT JOIN code_evaluation ce ON cs.submission_id = ce.submission_id
                LEFT JOIN feedback_score fs ON cs.submission_id = fs.submission_id
                WHERE cs.user_id=%s AND a.difficulty_level=%s
                """,
                (user_id, level_id),
            )
            row = cur.fetchone() or {}
            return {
                "assignment_count": int(row.get("assignment_count") or 0),
                "average_score": float(row.get("average_score") or 0.0),
                "average_pass_rate": float(row.get("average_pass_rate") or 0.0),
                "average_feedback_score": float(
                    row.get("average_feedback_score") or 0.0
                ),
            }

    def update_user_stats(self, user_id: int) -> bool:
        """Recalculate all difficulty levels for a single user and upsert rows."""
        try:
            # fetch all levels
            with DBConnection() as cur:
                cur.execute("SELECT level_id FROM difficulty_level")
                levels = cur.fetchall()

            with DBConnection() as cur:
                for lvl in levels:
                    lid = lvl["level_id"]
                    metrics = self._calculate_level_metrics_for_user(user_id, lid)
                    # upsert
                    cur.execute(
                        f"""
                        INSERT INTO {self.table}
                        (user_id, difficulty_level, assignment_count, average_score, average_pass_rate, average_feedback_score)
                        VALUES (%s,%s,%s,%s,%s,%s)
                        ON DUPLICATE KEY UPDATE
                          assignment_count=VALUES(assignment_count),
                          average_score=VALUES(average_score),
                          average_pass_rate=VALUES(average_pass_rate),
                          average_feedback_score=VALUES(average_feedback_score)
                        """,
                        (
                            user_id,
                            lid,
                            metrics["assignment_count"],
                            metrics["average_score"],
                            metrics["average_pass_rate"],
                            metrics["average_feedback_score"],
                        ),
                    )
            return True
        except Exception:
            logger.exception(
                "update_user_stats failed for %s on table %s", user_id, self.table
            )
            return False

    def update_all_users(self) -> bool:
        """Recalculate difficulty stats for all students/instructors (rows inserted/updated)."""
        try:
            with DBConnection() as cur:
                cur.execute(
                    "SELECT user_id FROM user_profile WHERE role IN ('student','instructor')"
                )
                users = cur.fetchall()
            for u in users:
                self.update_user_stats(u["user_id"])
            return True
        except Exception:
            logger.exception("update_all_users failed for table %s", self.table)
            return False


class StudentDifficultyAnalytics(DifficultyAnalyticsBase):
    def __init__(self):
        super().__init__("student_difficulty_stats")


class InstructorDifficultyAnalytics(DifficultyAnalyticsBase):
    def __init__(self):
        super().__init__("instructor_difficulty_stats")


# ---------------------------
# Performance Analytics (Student / Instructor)
# ---------------------------
class PerformanceAnalyticsBase:
    """
    Handles Student_Performance_Analytics and Instructor_Performance_Analytics.
    Student: avg score, completion rate, pass rate, plagiarism_incidents, performance band, total_assignments
    Instructor: total_assignments_created, total_submissions_received, overall_avg_score, avg_pass_rate, plagiarism_rate, feedback_score_avg
    """

    def __init__(self, table_name: str):
        self.table = table_name

    def get_user_performance(self, user_id: int) -> Optional[Dict[str, Any]]:
        with DBConnection() as cur:
            cur.execute(f"SELECT * FROM {self.table} WHERE user_id=%s", (user_id,))
            return cur.fetchone()

    def get_all_users(self) -> List[Dict[str, Any]]:
        with DBConnection() as cur:
            cur.execute(f"SELECT * FROM {self.table}")
            return cur.fetchall()

    def get_filtered(
        self, where_clause: str, params: tuple = ()
    ) -> List[Dict[str, Any]]:
        with DBConnection() as cur:
            query = f"SELECT * FROM {self.table} WHERE {where_clause}"
            cur.execute(query, params)
            return cur.fetchall()

    def update_user(self, user_id: int) -> bool:
        """To be implemented by subclasses."""
        raise NotImplementedError

    def update_all(self) -> bool:
        try:
            with DBConnection() as cur:
                cur.execute(
                    "SELECT user_id FROM user_profile WHERE role IN ('student','instructor')"
                )
                users = cur.fetchall()
            for u in users:
                self.update_user(u["user_id"])
            return True
        except Exception:
            logger.exception("update_all failed for %s", self.table)
            return False


class StudentPerformanceAnalytics(PerformanceAnalyticsBase):
    def __init__(self):
        super().__init__("student_performance_analytics")

    def update_user(self, user_id: int) -> bool:
        try:
            with DBConnection() as cur:
                # get aggregate metrics for the student across all their submissions
                cur.execute(
                    """
                    SELECT
                      AVG(ce.score) AS avg_score,
                      SUM(CASE WHEN ce.score >= 40 THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0) * 100 AS pass_rate,
                      COUNT(DISTINCT cs.assignment_id) AS assignments_submitted,
                      SUM(CASE WHEN ce.plagiarism_score > 50 THEN 1 ELSE 0 END) AS plagiarism_incidents,
                      AVG(fs.feedback_score) AS avg_feedback
                    FROM code_submission cs
                    LEFT JOIN code_evaluation ce ON cs.submission_id=ce.submission_id
                    LEFT JOIN feedback_score fs ON cs.submission_id=fs.submission_id
                    WHERE cs.user_id=%s
                    """,
                    (user_id,),
                )
                agg = cur.fetchone() or {}

                avg_score = float(agg.get("avg_score") or 0.0)
                pass_rate = float(agg.get("pass_rate") or 0.0)
                assignments_submitted = int(agg.get("assignments_submitted") or 0)
                plagiarism_incidents = int(agg.get("plagiarism_incidents") or 0)
                avg_feedback = float(agg.get("avg_feedback") or 0.0)

                # completion_rate: (distinct assignments submitted) / (total assignments) * 100
                cur.execute("SELECT COUNT(*) AS total_assignments FROM assignment")
                total_assignments = cur.fetchone().get("total_assignments") or 0
                completion_rate = (
                    (assignments_submitted / total_assignments * 100.0)
                    if total_assignments > 0
                    else 0.0
                )

                # performance band
                if avg_score >= 80:
                    band = "Excellent"
                elif avg_score >= 60:
                    band = "Good"
                elif avg_score >= 40:
                    band = "Average"
                else:
                    band = "Poor"

                # performance level (arbitrary buckets)
                if avg_score < 40:
                    perf_level = "L1"
                elif avg_score < 70:
                    perf_level = "L2"
                else:
                    perf_level = "L3"

                # upsert into student performance table
                cur.execute(
                    """
                    INSERT INTO student_performance_analytics
                    (user_id, average_score, completion_rate, pass_rate,
                     plagiarism_incidents, performance_band, total_assignments, performance_level, last_updated)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                    ON DUPLICATE KEY UPDATE
                      average_score=VALUES(average_score),
                      completion_rate=VALUES(completion_rate),
                      pass_rate=VALUES(pass_rate),
                      plagiarism_incidents=VALUES(plagiarism_incidents),
                      performance_band=VALUES(performance_band),
                      total_assignments=VALUES(total_assignments),
                      performance_level=VALUES(performance_level),
                      last_updated=NOW()
                    """,
                    (
                        user_id,
                        avg_score,
                        completion_rate,
                        pass_rate,
                        plagiarism_incidents,
                        band,
                        assignments_submitted,
                        perf_level,
                    ),
                )
            return True
        except Exception:
            logger.exception(
                "studentPerformanceAnalytics.update_user failed for %s", user_id
            )
            return False


class InstructorPerformanceAnalytics(PerformanceAnalyticsBase):
    def __init__(self):
        super().__init__("instructor_performance_analytics")

    def update_user(self, user_id: int) -> bool:
        try:
            with DBConnection() as cur:
                # aggregate metrics for assignments created by instructor
                cur.execute(
                    """
                    SELECT
                      COUNT(DISTINCT a.assignment_id) AS total_assignments,
                      COUNT(cs.submission_id) AS total_submissions,
                      AVG(ce.score) AS avg_score,
                      SUM(CASE WHEN ce.score >= 40 THEN 1 ELSE 0 END) / NULLIF(COUNT(ce.score),0) * 100 AS avg_pass_rate,
                      SUM(CASE WHEN ce.plagiarism_score > 50 THEN 1 ELSE 0 END) / NULLIF(COUNT(ce.score),0) * 100 AS plagiarism_rate,
                      AVG(fs.feedback_score) AS avg_feedback
                    FROM assignment a
                    LEFT JOIN code_submission cs ON a.assignment_id = cs.assignment_id
                    LEFT JOIN code_evaluation ce ON cs.submission_id = ce.submission_id
                    LEFT JOIN feedback_score fs ON cs.submission_id = fs.submission_id
                    WHERE a.instructor_id=%s
                    """,
                    (user_id,),
                )
                agg = cur.fetchone() or {}

                total_assignments = int(agg.get("total_assignments") or 0)
                total_submissions = int(agg.get("total_submissions") or 0)
                avg_score = float(agg.get("avg_score") or 0.0)
                avg_pass_rate = float(agg.get("avg_pass_rate") or 0.0)
                plagiarism_rate = float(agg.get("plagiarism_rate") or 0.0)
                avg_feedback = float(agg.get("avg_feedback") or 0.0)

                # placeholders for responsiveness/consistency can be derived from activity logs; set default if not available
                responsiveness_score = 80.0
                consistency_score = 90.0

                cur.execute(
                    """
                    INSERT INTO instructor_performance_analytics
                    (user_id, total_assignments_created, total_submissions_received, overall_avg_score,
                     avg_pass_rate, plagiarism_rate, feedback_score_avg, responsiveness_score, consistency_score, last_updated)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                    ON DUPLICATE KEY UPDATE
                      total_assignments_created=VALUES(total_assignments_created),
                      total_submissions_received=VALUES(total_submissions_received),
                      overall_avg_score=VALUES(overall_avg_score),
                      avg_pass_rate=VALUES(avg_pass_rate),
                      plagiarism_rate=VALUES(plagiarism_rate),
                      feedback_score_avg=VALUES(feedback_score_avg),
                      responsiveness_score=VALUES(responsiveness_score),
                      consistency_score=VALUES(consistency_score),
                      last_updated=NOW()
                    """,
                    (
                        user_id,
                        total_assignments,
                        total_submissions,
                        avg_score,
                        avg_pass_rate,
                        plagiarism_rate,
                        avg_feedback,
                        responsiveness_score,
                        consistency_score,
                    ),
                )
            return True
        except Exception:
            logger.exception(
                "InstructorPerformanceAnalytics.update_user failed for %s", user_id
            )
            return False


# ---------------------------
# Instructor Analytics (Dashboard queries only)
# ---------------------------
class InstructorAnalytics(BaseAnalytics):
    """Fetch-only queries for instructor dashboard tables and charts."""

    def get_assignment_analytics(self, instructor_id: int):
        result = self.fetch_all(
            """
            SELECT a.assignment_id,
       a.title,
       r.repository_id,
       r.repo_title,
       COALESCE(SUM(aa.total_submission), 0) AS total_submission,
       COALESCE(AVG(aa.average_score), 0)   AS average_score,
       COALESCE(AVG(aa.pass_percentage), 0) AS pass_percentage,
       COALESCE(SUM(aa.plagiarism_cases), 0) AS plagiarism_cases,
       COALESCE(AVG(aa.average_time), 0)    AS average_time
FROM assignment a
JOIN assignment_repository r 
  ON a.repository_id = r.repository_id
LEFT JOIN assignment_analytics aa 
  ON a.assignment_id = aa.assignment_id
WHERE a.instructor_id = %s
GROUP BY a.assignment_id, a.title, r.repository_id, r.repo_title
ORDER BY a.assignment_id DESC;

            """,
            (instructor_id,),
        )
        return result

    def get_student_performance_summary(self, instructor_id: int):
        return self.fetch_one(
            """
            SELECT AVG(spa.completion_rate) AS avg_completion,
                   AVG(spa.pass_rate) AS avg_pass,
                   SUM(spa.plagiarism_incidents) AS total_plagiarism
            FROM student_performance_analytics spa
            WHERE spa.user_id IN (
                SELECT DISTINCT cs.user_id
                FROM code_submission cs
                JOIN assignment a ON cs.assignment_id = a.assignment_id
                WHERE a.instructor_id = %s
            )
            """,
            (instructor_id,),
            {"avg_completion": 0, "avg_pass": 0, "total_plagiarism": 0},
        )

    def get_feedback_chart_data(self, instructor_id: int):
        rows = self.fetch_all(
            """
            SELECT a.title, AVG(fs.feedback_score) AS avg_feedback
            FROM feedback_score fs
            JOIN code_submission cs ON fs.submission_id = cs.submission_id
            JOIN assignment a ON cs.assignment_id = a.assignment_id
            WHERE a.instructor_id = %s
            GROUP BY a.assignment_id
            """,
            (instructor_id,),
        )
        return {
            "labels": [r["title"] for r in rows],
            "values": [round(r["avg_feedback"] or 0, 2) for r in rows],
        }

    def get_grade_distribution(self, instructor_id: int):
        scores = self.fetch_all(
            """
            SELECT ce.score
            FROM code_evaluation ce
            JOIN code_submission cs ON ce.submission_id = cs.submission_id
            JOIN assignment a ON cs.assignment_id = a.assignment_id
            WHERE a.instructor_id = %s
            """,
            (instructor_id,),
        )
        return SystemAnalytics._grade_distribution_from_scores(scores)

    def get_score_trend(self, instructor_id: int):
        rows = self.fetch_all(
            """
            SELECT DATE(cs.submitted_on) AS date, AVG(ce.score) AS avg_score
            FROM code_submission cs
            JOIN code_evaluation ce ON cs.submission_id = ce.submission_id
            JOIN assignment a ON cs.assignment_id = a.assignment_id
            WHERE a.instructor_id = %s
            GROUP BY DATE(cs.submitted_on)
            ORDER BY DATE(cs.submitted_on)
            """,
            (instructor_id,),
        )
        return {
            "labels": [str(r["date"]) for r in rows],
            "values": [round(r["avg_score"] or 0, 2) for r in rows],
        }

    def get_student_performance_bands(self, instructor_id: int):
        rows = self.fetch_all(
            """
            SELECT spa.performance_band, COUNT(*) AS count
            FROM student_performance_analytics spa
            WHERE spa.user_id IN (
                SELECT DISTINCT cs.user_id
                FROM code_submission cs
                JOIN assignment a ON cs.assignment_id = a.assignment_id
                WHERE a.instructor_id = %s
            )
            GROUP BY spa.performance_band
            """,
            (instructor_id,),
        )
        bands = {"Excellent": 0, "Good": 0, "Average": 0, "Poor": 0}
        for r in rows:
            if r["performance_band"] in bands:
                bands[r["performance_band"]] = r["count"]
        return bands

    def get_difficulty_chart_data(self, instructor_id: int):
        """
        Average score of assignments grouped by difficulty level.
        """
        rows = self.fetch_all(
            """
            SELECT dl.difficulty_types, AVG(ce.score) AS avg_score
            FROM assignment a
            JOIN difficulty_level dl ON a.difficulty_level = dl.level_id
            LEFT JOIN code_submission cs ON a.assignment_id = cs.assignment_id
            LEFT JOIN code_evaluation ce ON cs.submission_id = ce.submission_id
            WHERE a.instructor_id = %s
            GROUP BY dl.difficulty_types
            """,
            (instructor_id,),
        )
        return {
            "labels": [r["difficulty_types"] for r in rows],
            "values": [round(r["avg_score"] or 0, 2) for r in rows],
        }

    def get_popularity_chart_data(self, instructor_id: int):
        """
        Count submissions per assignment (popularity).
        """
        rows = self.fetch_all(
            """
            SELECT a.title, COUNT(cs.submission_id) AS submissions
            FROM assignment a
            LEFT JOIN code_submission cs ON a.assignment_id = cs.assignment_id
            WHERE a.instructor_id = %s
            GROUP BY a.assignment_id
            """,
            (instructor_id,),
        )
        return {
            "labels": [r["title"] for r in rows],
            "values": [r["submissions"] for r in rows],
        }

    def get_plagiarism_trend(self, instructor_id: int):
        """
        Daily plagiarism case counts for this instructor's assignments.
        """
        rows = self.fetch_all(
            """
            SELECT DATE(cs.submitted_on) AS date,
                   SUM(CASE WHEN ce.plagiarism_score > 50 THEN 1 ELSE 0 END) AS cases
            FROM code_submission cs
            JOIN code_evaluation ce ON cs.submission_id = ce.submission_id
            JOIN assignment a ON cs.assignment_id = a.assignment_id
            WHERE a.instructor_id = %s
            GROUP BY DATE(cs.submitted_on)
            ORDER BY DATE(cs.submitted_on)
            """,
            (instructor_id,),
        )
        return {
            "labels": [str(r["date"]) for r in rows],
            "values": [r["cases"] for r in rows],
        }

    def get_feedback_trend(self, instructor_id: int):
        """
        Daily average feedback score for this instructor's assignments.
        """
        rows = self.fetch_all(
            """
            SELECT DATE(cs.submitted_on) AS date, AVG(fs.feedback_score) AS avg_feedback
            FROM feedback_score fs
            JOIN code_submission cs ON fs.submission_id = cs.submission_id
            JOIN assignment a ON cs.assignment_id = a.assignment_id
            WHERE a.instructor_id = %s
            GROUP BY DATE(cs.submitted_on)
            ORDER BY DATE(cs.submitted_on)
            """,
            (instructor_id,),
        )
        return {
            "labels": [str(r["date"]) for r in rows],
            "values": [round(r["avg_feedback"] or 0, 2) for r in rows],
        }

    def get_submission_trend(self, instructor_id: int, interval: str = "day"):
        """
        Returns submission counts per assignment over time.
        interval = "day" | "week" | "month"
        """
        date_format = {"day": "%Y-%m-%d", "week": "%Y-%u", "month": "%Y-%m"}.get(
            interval, "%Y-%m-%d"
        )

        rows = self.fetch_all(
            f"""
            SELECT a.title,
                   DATE_FORMAT(cs.submitted_on, %s) AS period,
                   COUNT(*) AS submissions
            FROM assignment a
            LEFT JOIN code_submission cs ON a.assignment_id = cs.assignment_id
            WHERE a.instructor_id = %s
            GROUP BY a.assignment_id, period
            ORDER BY period
            """,
            (date_format, instructor_id),
        )
        return rows


class SubmissionAnalytics:
    """Handles fetching submission list with filters/search."""

    @staticmethod
    def list(query: str | None = None, limit: int = 100) -> list[dict]:
        conn = get_connection()
        with conn.cursor(dictionary=True) as cur:
            sql = """
                SELECT cs.submission_id, cs.user_id, cs.assignment_id,
                       cs.language, cs.submitted_on, cs.version, cs.code_path
                FROM code_submission cs
            """
            params = []
            filters = []

            if query:
                if query.isdigit():
                    filters.append(
                        "(cs.submission_id=%s OR cs.user_id=%s OR cs.assignment_id=%s OR cs.version=%s)"
                    )
                    params += [query, query, query, query]
                else:
                    filters.append("cs.version LIKE %s")
                    params.append(f"%{query}%")

            if filters:
                sql += " WHERE " + " OR ".join(filters)

            sql += " ORDER BY cs.submitted_on DESC LIMIT %s"
            params.append(limit)

            cur.execute(sql, tuple(params))
            return cur.fetchall()


class AssignmentAnalytics:
    """Handles fetching assignments + statistics."""

    @staticmethod
    def list(
        sort: str | None = None,
        limit: int = 100,
        repo_id: int | None = None,
        assignment_id: int | None = None,
    ) -> list[dict]:
        conn = get_connection()
        with conn.cursor(dictionary=True) as cur:
            sql = """
                SELECT a.assignment_id, a.title,
                       r.repository_id, r.repo_title,
                       COALESCE(sa.total_submissions, 0) AS total_submissions,
                       COALESCE(sa.average_score, 0) AS average_score,
                       COALESCE(sa.plagiarism_cases, 0) AS plagiarism_cases,
                       COALESCE(sa.pass_percentage, 0) AS pass_percentage,
                       COALESCE(sa.average_time, '-') AS average_time,
                       COALESCE(sa.most_common_error, '-') AS most_common_error
                FROM assignment a
                JOIN assignment_repository r 
                     ON a.repository_id = r.repository_id
                LEFT JOIN (
                    SELECT assignment_id,
                           MAX(total_submission)     AS total_submissions,
                           AVG(average_score)        AS average_score,
                           MAX(plagiarism_cases)     AS plagiarism_cases,
                           AVG(pass_percentage)      AS pass_percentage,
                           MAX(average_time)         AS average_time,
                           MAX(most_common_error)    AS most_common_error
                    FROM assignment_analytics
                    GROUP BY assignment_id
                ) sa ON a.assignment_id = sa.assignment_id
                WHERE 1=1
            """

            params = []

            # 🔎 Case 1: Filter by assignment_id (highest priority)
            if assignment_id:
                sql += " AND a.assignment_id = %s"
                params.append(assignment_id)

            # 🔎 Case 2: Filter by repo_id
            elif repo_id:
                sql += " AND a.repository_id = %s"
                params.append(repo_id)

            # 🔃 Sorting
            if sort == "repository_id":
                sql += " ORDER BY r.repository_id"
            else:
                sql += " ORDER BY a.assignment_id DESC"

            # ⚡ Limit only when not searching a specific assignment
            if not assignment_id:
                sql += " LIMIT %s"
                params.append(limit)

            cur.execute(sql, tuple(params))
            return cur.fetchall()


class UserAnalytics:
    """Handles fetching instructor/student lists with search + sort."""

    @staticmethod
    def list(
        role: str,
        search: str | None = None,
        sort: str | None = None,
        score: float | None = None,  # ✅ new filter
        limit: int = 100,
    ) -> list[dict]:
        conn = get_connection()
        with conn.cursor(dictionary=True) as cur:
            # Role-specific join
            if role == "instructor":
                join_sql = "LEFT JOIN instructor_performance_analytics pa ON u.user_id = pa.user_id"
                feedback_field = "pa.feedback_score_avg"
            elif role == "student":
                join_sql = "LEFT JOIN student_performance_analytics pa ON u.user_id = pa.user_id"
                # Student table has performance_band not numeric score
                feedback_field = "pa.performance_band"
            else:
                join_sql = ""
                feedback_field = "NULL"

            sql = f"""
                SELECT DISTINCT u.user_id,
                       CONCAT(u.first_name,' ',u.last_name) AS name,
                       u.email,
                       {feedback_field} AS feedback_score
                FROM user_profile u
                {join_sql}
                WHERE u.role=%s
            """
            params = [role]

            # 🔎 Search filter
            if search:
                if search.isdigit():
                    sql += " AND u.user_id=%s"
                    params.append(int(search))
                else:
                    sql += " AND u.email LIKE %s"
                    params.append(f"%{search}%")

            # 🎯 Score filter (exact match for instructor feedback_score_avg)
            if score is not None and role == "instructor":
                sql += f" AND {feedback_field} = %s"
                params.append(score)

            # 🔃 Sorting
            if sort == "feedback_score":
                sql += f" ORDER BY {feedback_field} DESC"
            else:
                sql += " ORDER BY u.user_id"

            sql += " LIMIT %s"
            params.append(limit)

            cur.execute(sql, tuple(params))
            return cur.fetchall()


class FeedbackAnalytics:
    """Handles student feedback for submissions."""

    @staticmethod
    def save_feedback(submission_id: int, score: float) -> bool:
        """
        Save feedback for a submission into feedback_score table.
        Returns True if success, False otherwise.
        """
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO feedback_score (submission_id, feedback_score)
                VALUES (%s, %s)
                """,
                (submission_id, score),
            )
            conn.commit()
            cur.close()
            conn.close()
            logger.info("✅ Feedback saved for submission_id=%s", submission_id)
            return True
        except Exception as e:
            logger.error(
                "❌ Feedback save failed for submission_id=%s: %s", submission_id, e
            )
            return False

    @staticmethod
    def get_feedback_for_submission(submission_id: int):
        """
        Fetch feedback (if any) for a given submission.
        """
        try:
            conn = get_connection()
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT feedback_id, submission_id, feedback_score",
                (submission_id,),
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
            return row
        except Exception as e:
            logger.error(
                "❌ Failed to fetch feedback for submission_id=%s: %s", submission_id, e
            )
            return None


# ---------------------------
# Scheduler: orchestrates updates
# ---------------------------
def start_scheduler(interval_hours: int = 1):
    """
    Start background scheduler that runs full updates at `interval_hours`.
    Call start_scheduler() from a safe place (e.g. when app starts) if you want automatic refresh.
    """
    scheduler = BackgroundScheduler()
    assn = AssignmentAnalyticsService()
    gd = GradeDistributionAnalytics()
    sdiff = StudentDifficultyAnalytics()
    idiff = InstructorDifficultyAnalytics()
    sp = StudentPerformanceAnalytics()
    ip = InstructorPerformanceAnalytics()
    sysa = SystemAnalytics()

    def job_update_all():
        logger.info("Scheduled analytics update started")
        try:
            assn.update_all()
            gd.recalculate_all_from_evaluations()
            sdiff.update_all_users()
            idiff.update_all_users()
            sp.update_all()
            ip.update_all()
            sysa.save_snapshot(sysa.collect_data())
            logger.info("Scheduled analytics update finished")
        except Exception:
            logger.exception("Scheduled update failed")

    scheduler.add_job(
        job_update_all, "interval", hours=interval_hours, next_run_time=None
    )
    scheduler.start()

    # keep a daemon thread alive for long-running environments where scheduler needs a thread
    threading.Thread(target=lambda: None, daemon=True).start()
    logger.info("Analytics scheduler started (interval_hours=%s)", interval_hours)
    return scheduler


# ---------------------------
# Small convenience factory instances
# ---------------------------
# Use these in app.py or other modules:
assignment_analytics = AssignmentAnalyticsService()
grade_distribution_analytics = GradeDistributionAnalytics()
student_difficulty_analytics = StudentDifficultyAnalytics()
instructor_difficulty_analytics = InstructorDifficultyAnalytics()
student_performance_analytics = StudentPerformanceAnalytics()
instructor_performance_analytics = InstructorPerformanceAnalytics()
system_analytics = SystemAnalytics()
instructor_analytics = InstructorAnalytics()
feedback = FeedbackAnalytics()


# ---------------------------
# Example main for manual run
# ---------------------------
if __name__ == "__main__":
    # start scheduler (in production you may start it from app boot code)
    start_scheduler(interval_hours=1)

    # run a single manual update (safe to call)
    logger.info("Running manual all-updates")
    print("start testing")
    assignment_analytics.update_all()
    grade_distribution_analytics.recalculate_all_from_evaluations()
    student_difficulty_analytics.update_all_users()
    instructor_difficulty_analytics.update_all_users()
    student_performance_analytics.update_all()
    instructor_performance_analytics.update_all()
    system_analytics.save_snapshot(system_analytics.collect_data())
    print("end testing")
    logger.info("Manual all-updates finished")
