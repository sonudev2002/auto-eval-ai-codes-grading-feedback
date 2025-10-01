import os
import sys
import logging
import secrets
import json
import subprocess
import backend.reporting_and_issue as ri
from functools import wraps
from backend.db import get_connection
from backend.notification_system import NotificationSystem
from datetime import datetime
from dotenv import load_dotenv
from werkzeug.middleware.shared_data import SharedDataMiddleware
from typing import Any, Dict

# import at the top of app.py
from backend.admin_control_panel import AdminControlPanel
from flask import (
    Flask,
    render_template,
    session,
    flash,
    redirect,
    url_for,
    request,
    jsonify,
    Blueprint,
    send_from_directory,
    current_app as app,
    abort,
    g,
)
from flask_cors import CORS

load_dotenv()  # Load environment variables
# Add backend directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))
# --- Import Config and Backend Modules ---
from config import Config
from backend.user_management import (
    register_user,
    user_verify,
    check_email,
    send_otp,
    verify_otp,
    user_logout,
    send_otp_email,
    change_password,
    StudentProfileData,
    InstructorProfileData,
    AdminProfileData,
    UpdateProfileData,
)
from backend.assignment_management import (
    upload_assignment,
    get_all_repositories,
    get_assignments_by_repo,
    get_assignment_details,
    update_assignment_backend,
    delete_assignment,
    create_repository,
    delete_repository_by_id,
    AssignmentsStudent,
    Code_editor,
)
from backend.code_submission import (
    submit_code,
    CodeRunner,
    get_submission_details,
)
from backend.evaluation_pipeline import EvaluationPipeline
from backend.analytics import (
    student_difficulty_analytics,
    instructor_difficulty_analytics,
    student_performance_analytics,
    instructor_performance_analytics,
    system_analytics,
    instructor_analytics,
    assignment_analytics,
    SubmissionAnalytics,
    AssignmentAnalytics,
    UserAnalytics,
    FeedbackAnalytics,
)

from backend.grade_distribution import GradeDistributionManager


from flask import render_template, request, jsonify, flash, redirect, url_for


# --- Flask App Configuration ---
app = Flask(__name__, template_folder="frontend/templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))

grade_distribution_analytics = GradeDistributionManager()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Serve uploads folder as static at /uploads
app.wsgi_app = SharedDataMiddleware(
    app.wsgi_app,
    {"/uploads": os.path.join(os.path.abspath(os.path.dirname(__file__)), "uploads")},
)

CORS(app)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def health():
    return "OK", 200


@app.route("/")
def home():
    return "Server is Live ✅"


@app.route("/healthz")
def health_check():
    return {"status": "ok"}, 200


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# Create pipeline instance
pipeline = EvaluationPipeline()


@app.before_request
def inject_user_role():
    """Inject current user role into Flask's g object for role_required decorator."""
    user = session.get("user")
    g.current_user_role = user.get("role") if user else None


# ------------------ Page Routes ------------------ #
@app.route("/")
def index():
    return render_template("index.html")


# ------------------ Role-Based Dashboard ------------------ #
def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            user = session.get("user")
            user_role = user.get("role") if user else None
            if user_role not in roles:
                return jsonify({"error": "Unauthorized", "allowed_roles": roles}), 403
            return f(*args, **kwargs)

        return wrapper

    return decorator


# ------------------ Student Dashboard ------------------ #


logger = logging.getLogger(__name__)


@app.route("/student_dashboard")
@role_required("student")
def student_dashboard():
    """
    Student dashboard:
    - Shows repositories + assignments
    - Injects logged-in user_id from session for notifications
    """
    repo_id = request.args.get("repo_id", type=int)
    logger.info("📌 Student dashboard accessed. Repo ID: %s", repo_id)

    try:
        repo_detail = AssignmentsStudent.get_repository_details()

        if repo_id:
            assignments = AssignmentsStudent.get_assignments_by_repo_detailed(repo_id)
        else:
            assignments = AssignmentsStudent.get_dashboard_assignment_detail()

        user = session.get("user")
        user_id = user.get("user_id") if user else None

        return render_template(
            "student_dashboard.html",
            repo_detail=repo_detail,
            assignments=assignments,
            selected_repo=repo_id,
            now=datetime.now(),
            user_id=user_id,  # ✅ for JS
            current_user=user,  # ✅ safe dict, not Flask-Login
        )

    except Exception:
        logger.exception("❌ Error in student_dashboard")
        return (
            render_template(
                "student_dashboard.html",
                repo_detail=[],
                assignments=[],
                selected_repo=None,
                now=datetime.now(),
                user_id=session.get("user", {}).get("user_id"),
                error="Something went wrong while loading your dashboard. Please try again.",
            ),
            500,
        )


@app.route("/fetch_assignments_for_student")
@role_required("student")
def get_assignments_by_repo_id():
    repo_id = request.args.get("repo_id", type=int)
    if not repo_id:
        return jsonify([])

    assignments = AssignmentsStudent.get_assignments_by_repo_detailed(repo_id)
    for a in assignments:
        if isinstance(a.get("due_date"), datetime):
            a["due_date"] = a["due_date"].strftime("%d %b %Y")

    return jsonify(assignments)


# ---------------- Instructor Dashboard ---------------- #


@app.route("/instructor_dashboard")
def instructor_dashboard():
    """
    Render the Instructor Dashboard frontend page.
    Access is still role-checked, but instead of 403 it redirects to login.
    """
    user = session.get("user")
    if not user or user.get("role", "").strip().lower() != "instructor":
        return redirect(url_for("login"))  # safer than returning 403 for frontend pages
    return render_template("instructor_dashboard.html")


@app.route("/api/instructor/dashboard")
@role_required("instructor")
def api_instructor_dashboard():
    """
    API endpoint returning instructor analytics (tables + charts).
    Consumed by instructor_dashboard.html via fetch().
    """

    user = session.get("user")
    instructor_id = user.get("user_id")

    try:

        data = {
            "assignment_analytics": instructor_analytics.get_assignment_analytics(
                instructor_id
            ),
            "performance_summary": instructor_analytics.get_student_performance_summary(
                instructor_id
            ),
            "charts": {
                "feedback": instructor_analytics.get_feedback_chart_data(instructor_id),
                "grades": instructor_analytics.get_grade_distribution(instructor_id),
                "score_trend": instructor_analytics.get_score_trend(instructor_id),
                "submission_trend": instructor_analytics.get_submission_trend(
                    instructor_id, request.args.get("interval", "day")
                ),
                "performance_bands": instructor_analytics.get_student_performance_bands(
                    instructor_id
                ),
                "difficulty": instructor_analytics.get_difficulty_chart_data(
                    instructor_id
                ),
                "popularity": instructor_analytics.get_popularity_chart_data(
                    instructor_id
                ),
                "plagiarism_trend": instructor_analytics.get_plagiarism_trend(
                    instructor_id
                ),
                "feedback_trend": instructor_analytics.get_feedback_trend(
                    instructor_id
                ),
            },
            "status": "success",
        }
        return jsonify(data)
    except Exception as e:
        app.logger.exception("❌ Instructor dashboard API failed")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        instructor_analytics.close()


# ------------------ Login & Logout ------------------ #
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("index.html")
    return user_verify()


@app.route("/logout")
def logout():
    if "user" not in session:
        return redirect(url_for("index"))
    return user_logout(session)


# ------------------ Signup & OTP ------------------ #
@app.route("/signup", methods=["POST"])
def signup():
    return register_user()


@app.route("/check-email")
def check_email_route():
    return check_email()


@app.route("/send-otp", methods=["POST"])
def send_otp_route():
    return send_otp()


@app.route("/verify-otp", methods=["POST"])
def verify_otp_route():
    return verify_otp()


@app.route("/send-otp-email", methods=["POST"])
def send_email_otp_route():
    return send_otp_email()


@app.route("/reset-password", methods=["GET", "POST"])
def reset_password_route():
    return change_password()


# ------------------ Assignment Management ------------------ #
@app.route("/create-assignment")
def show_form():
    user = session.get("user")
    if not user or user.get("role") not in ("admin", "instructor"):
        return jsonify(success=False, message="Access denied"), 403
    return render_template("create_assignment.html")


@app.route("/create-repository", methods=["POST"])
def create_repository_route():
    try:
        user = session.get("user")
        if not user or user.get("role") != "admin":
            return jsonify({"success": False, "message": "Unauthorized"}), 403

        data = request.get_json()
        title = data.get("title", "").strip()
        if not title:
            return jsonify(success=False, message="Missing title")

        success, message = create_repository(title, user.get("user_id"))
        return jsonify(success=success, message=message)
    except Exception as e:
        logger.error("🔥 Exception in create_repository_route: %s", e)
        return jsonify(success=False, message="Internal server error"), 500


@app.route("/api/repositories")
def fetch_repositories():
    repos = get_all_repositories()
    return jsonify(repos)


@app.route("/delete-repository", methods=["POST"])
def delete_repository():
    try:
        user = session.get("user")
        if not user or user.get("role") != "admin":
            return jsonify({"success": False, "message": "Unauthorized"}), 403

        data = request.get_json()
        repo_id = data.get("repository_id")
        if not repo_id:
            return jsonify({"success": False, "error": "Repository ID is missing"}), 400

        success, message = delete_repository_by_id(repo_id)
        if success:
            return jsonify({"success": True, "message": message}), 200
        else:
            return jsonify({"success": False, "message": message}), 500
    except Exception as e:
        logger.error("❌ Exception in delete_repository route: %s", e)
        return jsonify({"success": False, "message": "Internal server error."}), 500


@app.route("/api/assignments/<int:repo_id>")
def fetch_assignments(repo_id):
    assignments = get_assignments_by_repo(repo_id)
    return jsonify(assignments)


@app.route("/api/assignment/<int:assignment_id>")
def fetch_assignment_details(assignment_id):
    assignment = get_assignment_details(assignment_id)
    if assignment:
        return jsonify(assignment)
    return jsonify({"error": "Assignment not found"}), 404


@app.route("/upload-assignment", methods=["GET", "POST"])
def handle_submission():
    if request.method == "GET":
        return redirect("/create-assignment")

    user = session.get("user")
    if not user:
        flash("Session expired. Please log in again.", "warning")
        return redirect("/login")

    if user.get("role") == "student":
        flash("Access denied. Only instructors can upload assignments.", "danger")
        return "Access denied", 403

    form_data = request.form.to_dict(flat=False)
    form_data["instructor_id"] = user["user_id"]

    repo_raw = form_data.get("repository_id", [None])[0]
    if not repo_raw or repo_raw == "undefined" or not repo_raw.isdigit():
        return jsonify(success=False, message="Invalid repository selected."), 400

    # ✅ Extract CSV file safely
    csv_file = request.files.get("testcase_csv")

    success = upload_assignment(form_data, csv_file=csv_file)

    if success is True:
        return jsonify(success=True, message="Assignment uploaded successfully!")
    elif isinstance(success, dict):  # if error dict returned
        return (
            jsonify(success=False, message=success.get("message", "Upload failed")),
            400,
        )
    else:
        return jsonify(success=False, message="Upload failed due to server error."), 500


@app.route("/update-assignment", methods=["POST"])
def handle_update():
    user = session.get("user")
    if not user or user.get("role") not in ("admin", "instructor"):
        return jsonify(success=False, message="Access denied"), 403

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(success=False, message="Invalid or missing JSON payload"), 400

    try:
        data["assignment_id"] = int(data.get("assignment_id"))
    except (TypeError, ValueError):
        return jsonify(success=False, message="Invalid assignment_id"), 400

    data["instructor_id"] = user.get("user_id")
    ok = update_assignment_backend(data)
    if ok:
        return jsonify(success=True)
    return jsonify(success=False, message="Update failed due to server error"), 500


@app.route("/delete-assignment", methods=["POST"])
def delete_assignment_route():
    user = session.get("user")
    if not user or user.get("role") not in ("admin", "instructor"):
        return jsonify(success=False, message="Access denied"), 403

    data = request.get_json()
    assignment_id = data.get("assignment_id")
    if not assignment_id:
        return jsonify({"success": False, "message": "Assignment ID required"}), 400

    success = delete_assignment(assignment_id)
    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Deletion failed"})


# ------------------ Code Editor ------------------ #
@app.route("/view-code")
def view_code():
    rel_path = request.args.get("path")
    if not rel_path:
        return jsonify({"error": "Path missing"}), 400

    # 🧹 Step 1: strip whitespace
    rel_path = rel_path.strip()

    # 🧹 Step 2: remove any leading slashes/backslashes (fix old DB rows like "\submitted_codes\...")
    rel_path = rel_path.lstrip("/\\")

    # 🧹 Step 3: normalize separators
    rel_path = rel_path.replace("\\", "/")

    # 🧹 Step 4: remove "submitted_codes/" prefix if still present
    if rel_path.lower().startswith("submitted_codes/"):
        rel_path = rel_path[len("submitted_codes/") :]

    # ✅ Base directory = project submitted_codes folder
    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "mca_final_project/submitted_codes"
        )
    )

    # ✅ Final absolute path
    safe_path = os.path.abspath(os.path.join(base_dir, rel_path))

    # 🚨 Security check
    if not safe_path.startswith(base_dir):
        return jsonify({"error": f"Invalid path {safe_path}"}), 400

    if not os.path.exists(safe_path):
        return jsonify({"error": f"File not found at {safe_path}"}), 404

    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            code = f.read()
        return jsonify({"code": code})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/code_submission_report")
@role_required("student", "admin")
def code_submission_report():
    submission_id = request.args.get("submission_id", type=int)
    if not submission_id:
        return (
            render_template(
                "code_submission_report.html",
                result={"status": "error", "message": "❌ Missing submission_id"},
            ),
            400,
        )
    # ✅ Fetch submission details from DB
    details = get_submission_details(submission_id)
    if not details:
        return (
            render_template(
                "code_submission_report.html",
                result={
                    "status": "error",
                    "message": f"❌ Submission {submission_id} not found",
                },
            ),
            404,
        )
    return render_template("code_submission_report.html", result=details)


@app.route(
    "/code_editor/<int:assignment_id>", methods=["GET"], endpoint="open_code_editor"
)
@role_required("student", "admin", "instructor")
def open_code_editor(assignment_id):
    try:
        user = session.get("user")
        if not user:
            flash("You must be logged in.", "danger")
            return redirect(url_for("login"))

        details = Code_editor.assignment_detail_by_id(
            assignment_id,
            user_id=user.get("user_id"),
            role=user.get("role"),
        )

        if not details:
            flash(f"Assignment #{assignment_id} not found.", "warning")
            dashboard_map = {
                "student": "student_dashboard",
                "instructor": "instructor_dashboard",
                "admin": "admin_dashboard",
            }
            return redirect(url_for(dashboard_map.get(user.get("role"), "index")))

        return render_template("code_editor.html", detail=details, id=assignment_id)

    except Exception as e:
        app.logger.error(f"Error in open_code_editor: {e}")
        flash("An unexpected error occurred. Please try again.", "danger")
        dashboard_map = {
            "student": "student_dashboard",
            "instructor": "instructor_dashboard",
            "admin": "admin_dashboard",
        }
        return redirect(url_for(dashboard_map.get(user.get("role"), "index")))


@app.route("/run", methods=["POST"])
def run_code_route():
    try:
        data = request.get_json()
        code = data.get("code")
        lang = data.get("lang", "python3")
        inputs = data.get("inputs", [])  # ✅ list of example inputs from frontend
        runner = CodeRunner(language=lang)
        runner.start_container(code)
        results = runner.run_multiple_inputs(inputs)  # ✅ must pass inputs list
        return jsonify(results)
    except Exception as e:
        app.logger.error("Run error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/submit-code", methods=["POST"])
@role_required("student", "admin", "instructor")
def submit_code_route():
    try:
        # Handle both JSON (fetch) and form submission
        data = None
        if request.is_json:
            data = request.get_json(silent=True)
        if not data:
            if "payload" in request.form:
                data = json.loads(request.form["payload"])
            else:
                data = request.form.to_dict()

        assignment_id = data.get("assignmentId") or data.get("assignment_id")
        student_id = session.get("user", {}).get("user_id")
        code = data.get("code")
        lang = data.get("lang")

        if not assignment_id or not student_id or not code:
            flash("❌ Missing assignment_id, student_id, or code", "danger")
            return redirect(url_for("open_code_editor", assignment_id=assignment_id))

        # ✅ Run evaluation
        result = submit_code(
            assignment_id=assignment_id,
            student_id=student_id,
            source_code=code,
            language=lang,
        )

        submission_id = result.get("submission_id")
        if not submission_id:
            flash("❌ Submission failed. Please try again.", "danger")
            return redirect(url_for("open_code_editor", assignment_id=assignment_id))

        # ✅ Directly redirect to report page
        return redirect(url_for("code_submission_report", submission_id=submission_id))

    except Exception as e:
        app.logger.error("Submit error: %s", e)
        flash(f"❌ {str(e)}", "danger")
        return redirect(url_for("student_dashboard"))


@app.route("/api/feedback/submit", methods=["POST"])
@role_required("student")
def submit_feedback():
    try:
        data = request.get_json(force=True)
        submission_id = data.get("submission_id")
        score = data.get("score")

        if not submission_id or score is None:
            return (
                jsonify(
                    {"status": "error", "message": "Missing submission_id or score"}
                ),
                400,
            )

        ok = FeedbackAnalytics.save_feedback(submission_id, score)
        if ok:
            return jsonify({"status": "success", "message": "Feedback saved ✅"})
        return jsonify({"status": "error", "message": "DB insert failed"}), 500

    except Exception as e:
        app.logger.error("Feedback API error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ------------------ Formatters ------------------ #
@app.route("/format/python", methods=["POST"])
def format_python():
    code = request.json.get("code", "")
    try:
        result = subprocess.run(
            ["black", "-q", "-"],
            input=code.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return jsonify({"formatted": result.stdout.decode() or code})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/format/cpp", methods=["POST"])
def format_cpp():
    code = request.json.get("code", "")
    try:
        result = subprocess.run(
            [r"C:\Program Files\LLVM\bin\clang-format.exe"],
            input=code,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return jsonify({"formatted": result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/format/java", methods=["POST"])
def format_java():
    code = request.json.get("code", "")
    try:
        result = subprocess.run(
            [
                "java",
                "-jar",
                r"D:\mca_final_project\google-java-format-1.28.0-all-deps.jar",
                "-",
            ],
            input=code.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return jsonify({"formatted": result.stdout.decode()})
    except Exception as e:
        return jsonify({"error": str(e)})


# ----------------------profile page--------------------------#
profile_bp = Blueprint("profile", __name__)


@profile_bp.route("/<int:user_id>", methods=["GET"])
def profile(user_id):
    # ----------------------------
    # 🔒 Authorization Check
    # ----------------------------
    user_session = session.get("user")
    if not user_session:
        flash("You must be logged in to view profile.", "danger")
        return redirect(url_for("login"))

    if user_session["role"] != "admin" and user_session["user_id"] != user_id:
        abort(403)

    # ----------------------------
    # Decide Target Role
    # ----------------------------
    session_role = user_session["role"]
    target_role = request.args.get("role", session_role)  # ✅ use query param if given

    # ----------------------------
    # Load Profile Based on Target Role
    # ----------------------------
    if target_role == "student":
        pdata = StudentProfileData(user_id)
        profile_data = pdata.get_profile_data()
        return render_template(
            "profile.html",
            user=profile_data["user"],
            profile=profile_data["profile"],
            report=profile_data.get("report"),
            completed_assignments=profile_data.get("completed_assignments", {}),
        )

    elif target_role == "instructor":
        pdata = InstructorProfileData(user_id)
        profile_data = pdata.get_profile_data()
        return render_template(
            "profile.html",
            user=profile_data["user"],
            profile=profile_data["profile"],
            report=profile_data.get("report"),
            completed_assignments=profile_data.get("managed_assignments", {}),
        )

    elif target_role == "admin":
        pdata = AdminProfileData(user_id)
        profile_data = pdata.get_profile_data()
        return render_template(
            "profile.html",
            user=profile_data["user"],
            profile=profile_data["profile"],
            completed_assignments={},  # admins don’t have assignments
        )

    else:
        abort(403)


# ----------------------------
# Profile Update Endpoints
# ----------------------------
@profile_bp.route("/<int:user_id>/update_picture", methods=["POST"])
def update_picture(user_id):
    user_session = session.get("user")
    if not user_session:
        abort(403)

    # 🚫 Prevent admin from updating others
    if user_session["role"] == "admin" and user_session["user_id"] != user_id:
        flash("Admins can only view other profiles, not update them.", "warning")
        return redirect(
            url_for("profile.profile", user_id=user_id, role=request.form.get("role"))
        )

    # ✅ Student/Instructor updating own profile OR Admin updating own profile
    file = request.files.get("profile_picture")
    delete_picture = request.form.get("delete_picture") == "1"

    updater = UpdateProfileData(user_id)
    result = updater.update_picture(file=file, delete_picture=delete_picture)
    flash(result["message"], "success" if result["status"] == "success" else "danger")
    return redirect(
        url_for("profile.profile", user_id=user_id, role=request.form.get("role"))
    )


@profile_bp.route("/<int:user_id>/update_info", methods=["POST"])
def update_info(user_id):
    user_session = session.get("user")
    if not user_session:
        abort(403)

    # 🚫 Prevent admin from updating others
    if user_session["role"] == "admin" and user_session["user_id"] != user_id:
        flash("Admins can only view other profiles, not update them.", "warning")
        return redirect(
            url_for("profile.profile", user_id=user_id, role=request.form.get("role"))
        )

    # ✅ Allowed
    updater = UpdateProfileData(user_id)
    result = updater.update_info(request.form)
    flash(result["message"], "success" if result["status"] == "success" else "danger")
    return redirect(
        url_for("profile.profile", user_id=user_id, role=request.form.get("role"))
    )


@profile_bp.route("/<int:user_id>/update_password", methods=["POST"])
def update_password(user_id):
    user_session = session.get("user")
    if not user_session:
        abort(403)

    # 🚫 Prevent admin from updating others
    if user_session["role"] == "admin" and user_session["user_id"] != user_id:
        flash("Admins can only view other profiles, not update them.", "warning")
        return redirect(
            url_for("profile.profile", user_id=user_id, role=request.form.get("role"))
        )

    # ✅ Allowed
    form_data = request.form
    updater = UpdateProfileData(user_id)

    result = updater.update_password(
        current_password=form_data.get("current_password"),
        new_password=form_data.get("password"),
        confirm_password=form_data.get("confirm_password"),
    )

    flash(result["message"], "success" if result["status"] == "success" else "danger")
    return redirect(
        url_for("profile.profile", user_id=user_id, role=request.form.get("role"))
    )


# ✅ Register Blueprint
app.register_blueprint(profile_bp, url_prefix="/profile")


# --------------------------------Reporting issue and resolve---------------------------#
# ----------------------------
# Report an Issue (Form Display)
# ----------------------------
@app.route("/report_issue/form")
def report_issue_form():
    if "user" not in session:
        flash("You must be logged in to report an issue.", "danger")
        return redirect(url_for("login"))
    user_id = session["user"]["user_id"]
    issues = ri.get_user_issues(user_id)
    return render_template("report_issue.html", user_id=user_id, issues=issues)


@app.route("/issue/<int:issue_id>/screenshots")
def issue_screenshots(issue_id):
    screenshots = ri.get_screenshots(issue_id)
    return jsonify({"screenshots": screenshots})


# ----------------------------
# Submit Issue
# ----------------------------
@app.route("/report_issue/submit", methods=["POST"])
def submit_issue():
    if "user" not in session:
        flash("You must be logged in to submit an issue.", "danger")
        return redirect(url_for("login"))

    user_id = session["user"]["user_id"]
    issue_type = request.form["issue_type"]
    description = request.form["description"]
    screenshots = request.files.getlist("screenshots")

    result = ri.submit_issue(user_id, issue_type, description, screenshots)

    if result["success"]:
        flash("Issue reported successfully!", "success")
    else:
        flash(f"Error: {result['message']}", "danger")

    # ✅ Redirect based on role
    role = session["user"]["role"]
    if role == "student":
        return redirect(url_for("student_dashboard"))
    elif role == "instructor":
        return redirect(url_for("instructor_dashboard"))
    elif role == "admin":
        return redirect(url_for("admin_dashboard"))
    else:
        return redirect(url_for("login"))


# ----------------------------
# Student/Instructor: Track My Issues
# ----------------------------
@app.route("/my_issues")
def my_issues():
    if "user" not in session:
        flash("Please log in to view your issues.", "danger")
        return redirect(url_for("login"))

    user_id = session["user"]["user_id"]
    issues = ri.get_user_issues(user_id)
    return render_template("my_issues.html", issues=issues)


# ----------------------------
# Admin: View All Issues
# ----------------------------
@app.route("/admin/issues")
def admin_issues():
    if "user" not in session or session["user"]["role"] != "admin":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    issues = ri.get_all_issues()
    return render_template("admin_issues.html", issues=issues)


# ----------------------------
# Admin: Resolve Issue
# ----------------------------
@app.route("/admin/resolve/<int:issue_id>", methods=["POST"])
def resolve_issue(issue_id):
    if "user" not in session or session["user"]["role"] != "admin":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    result = ri.resolve_issue(issue_id)
    if result["success"]:
        flash("Issue marked as resolved ✅", "success")
    else:
        flash(f"Error: {result['message']}", "danger")

    return redirect(url_for("admin_issues"))


# ---------------------------------------- notification system ----------------- #


# ✅ Instantiate Notification System once
notifier = NotificationSystem(use_background=True)


@app.route("/create_notification")
@role_required("admin")
def creator_admin():
    return render_template("create_notification.html")


# ============================================================
# Targeted Notifications (Notification table)
# ============================================================
@app.route("/notifications/send", methods=["POST"])
@role_required("admin")
def send_notification():
    """
    Send notification to specific recipients.
    Supports:
    - Single user (with user_id)
    - Group of users (comma-separated user_ids)
    - Role-based (students, instructors, all, students_and_instructors)
    Works with both JSON API and HTML form.
    """
    try:
        is_json = request.is_json
        data = request.get_json(silent=True) or request.form.to_dict(flat=True)

        # -------------------------
        # Validation
        # -------------------------
        message = (data.get("message") or "").strip()
        if not message:
            raise ValueError("Message cannot be empty.")

        rec_value = data.get("recipients")
        recipients = []

        if rec_value == "single":
            # Single user
            user_id = data.get("user_id")
            if not user_id:
                raise ValueError("User ID required for single recipient.")
            try:
                user_id = int(user_id)
            except ValueError:
                raise ValueError("User ID must be an integer.")

            # ✅ Validate via notifier
            email = notifier.users.get_user_email(user_id)
            mobile = notifier.users.get_user_mobile_number(user_id)
            if not (email or mobile):
                raise ValueError(f"User with ID {user_id} does not exist.")
            recipients = [user_id]

        elif rec_value == "group":
            # Group of users
            user_ids_raw = data.get("user_ids")
            if not user_ids_raw:
                raise ValueError("User IDs required for group recipients.")
            try:
                user_ids = [
                    int(uid.strip()) for uid in user_ids_raw.split(",") if uid.strip()
                ]
            except ValueError:
                raise ValueError("User IDs must be valid integers (comma-separated).")

            if not user_ids:
                raise ValueError("No valid user IDs provided.")

            # ✅ Validate via notifier
            valid_ids = []
            for uid in user_ids:
                email = notifier.users.get_user_email(uid)
                mobile = notifier.users.get_user_mobile_number(uid)
                if email or mobile:
                    valid_ids.append(uid)

            if not valid_ids:
                raise ValueError("None of the provided user IDs exist.")
            recipients = valid_ids

        elif rec_value in (
            "students",
            "instructors",
            "all",
            "students_and_instructors",
        ):
            # Role-based groups
            recipients = [rec_value]

        else:
            raise ValueError("Invalid recipient type.")

        # -------------------------
        # Channels
        # -------------------------
        if not is_json and request.form:
            channels = request.form.getlist("channels")
        else:
            channels = data.get("channels", ["dashboard"])

        if not channels:
            raise ValueError("At least one channel must be selected.")

        # -------------------------
        # Notification Engine
        # -------------------------
        sender_role = str(data.get("sender_role", "admin"))  # ✅ force string
        sender_id = int(data.get("sender_id", 0))

        notifier.send_message(
            sender_role=sender_role,
            sender_id=sender_id,
            message=message,
            recipients=recipients,
            channels=channels,
            subject=data.get("subject"),
            notif_type=data.get("type", "info"),
        )

        # -------------------------
        # Response
        # -------------------------
        if is_json:
            return jsonify({"status": "success"}), 200
        else:
            flash("✅ Notification sent successfully!", "success")
            return redirect(url_for("creator_admin"))

    except Exception as e:
        app.logger.exception("Failed to send notification")
        if request.is_json:
            return jsonify({"status": "error", "error": str(e)}), 400
        else:
            flash(f"❌ Failed to send notification: {str(e)}", "danger")
            return redirect(url_for("creator_admin"))


# ============================================================
# Broadcasts (BroadCast_Notification table)
# ============================================================
@app.route("/notifications/broadcast", methods=["POST"])
@role_required("admin")
def broadcast_notification():
    """
    Broadcast to all users.
    JSON Payload:
    {
        "message": "System maintenance at 12 AM",
        "channels": ["dashboard","email"],
        "subject": "Maintenance Notice",
        "type": "system",
        "broadcast_type": "system",
        "broadcast_mode": "dashboard+email"
    }
    """
    try:
        data = request.get_json(force=True)
        message = (data.get("message") or "").strip()
        if not message:
            raise ValueError("Message cannot be empty.")

        notifier.broadcast(
            message=message,
            channels=data.get("channels", ["dashboard"]),
            subject=data.get("subject"),
            notif_type=data.get("type", "broadcast"),
            btype=data.get("broadcast_type", "general"),
            mode=data.get("broadcast_mode", "system"),
        )
        return jsonify({"status": "success"}), 200
    except Exception as e:
        app.logger.exception("Broadcast failed")
        return jsonify({"status": "error", "error": str(e)}), 500


# ============================================================
# User APIs
# ============================================================
@app.route("/notifications/<int:user_id>", methods=["GET"])
def fetch_notifications(user_id):
    """
    Fetch targeted/system notifications (not broadcasts).
    Query params: status (optional), limit, offset
    """
    try:
        status = request.args.get("status")
        limit = int(request.args.get("limit", 20))
        offset = int(request.args.get("offset", 0))
        data = notifier.fetch_user_notifications(user_id, status, limit, offset)
        return jsonify(data), 200
    except Exception as e:
        app.logger.exception("Fetch failed")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/notifications/mark_read/<int:notification_id>", methods=["PUT"])
def mark_read(notification_id):
    try:
        notifier.mark_notification_read(notification_id)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        app.logger.exception("Mark read failed")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/notifications/<int:user_id>/count", methods=["GET"])
def count_unread(user_id):
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT COUNT(*) AS cnt FROM Notification WHERE user_id=%s AND status='unread'",
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()
        return jsonify({"unread_count": row["cnt"]}), 200
    except Exception as e:
        app.logger.exception("Count unread failed")
        return jsonify({"status": "error", "error": str(e)}), 500


# ============================================================
# System-triggered Examples
# ============================================================
@app.route("/notifications/feed/<int:user_id>", methods=["GET"])
def unified_feed(user_id):
    """
      Unified feed → merges notifications + broadcasts.
      Query params:- role: student|instructor|admin|user (default = session role, fallback=student)
    - limit, offset
    """
    try:
        user = session.get("user", {})
        role = (request.args.get("role") or user.get("role", "student")).strip().lower()
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))

        # Personal notifications
        notifications = (
            notifier.fetch_user_notifications(user_id, None, limit, offset) or []
        )
        for n in notifications:
            n["source"] = "notification"
        # Broadcasts for this role
        broadcasts = notifier.fetch_broadcasts_for_role(role, limit, offset) or []
        for b in broadcasts:
            b["source"] = "broadcast"
        combined = notifications + broadcasts

        def _created_at_key(item):
            ca = item.get("created_at")
            if isinstance(ca, str):
                try:
                    return datetime.fromisoformat(ca)
                except Exception:
                    return datetime.now()
            if isinstance(ca, datetime):
                return ca
            return datetime.now()

        combined.sort(key=_created_at_key, reverse=True)

        # Convert datetime to ISO
        for entry in combined:
            ca = entry.get("created_at")
            if isinstance(ca, datetime):
                entry["created_at"] = ca.isoformat()

        return jsonify({"user_id": user_id, "role": role, "feed": combined}), 200

    except Exception as e:
        app.logger.exception("Unified feed fetch failed")
        return jsonify({"status": "error", "error": str(e)}), 500


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# ========================================================# ============================================================
# ---------------- Assignment Analytics ----------------
@app.route("/api/analytics/assignment/update/group", methods=["POST"])
@role_required("admin", "instructor")
def update_assignments_group():
    ids = request.json.get("assignment_ids", [])
    if not ids:
        return (
            jsonify({"status": "error", "message": "assignment_ids list required"}),
            400,
        )
    results = {aid: assignment_analytics.update_assignment(aid) for aid in ids}
    return jsonify({"status": "success", "updated": results})


@app.route("/api/analytics/assignment/update/all", methods=["POST"])
@role_required("admin", "instructor")
def update_assignments_all():
    return jsonify({"status": "success", "updated": assignment_analytics.update_all()})


@app.route("/api/analytics/assignment/fetch/<int:assignment_id>", methods=["GET"])
def fetch_assignment_analytics(assignment_id):
    data = assignment_analytics.fetch_assignment(assignment_id)
    if not data:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"No analytics found for assignment {assignment_id}",
                }
            ),
            404,
        )
    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/assignment/fetch/group", methods=["POST"])
def fetch_assignments_group():
    ids = request.json.get("assignment_ids", [])
    if not ids:
        return (
            jsonify({"status": "error", "message": "assignment_ids list required"}),
            400,
        )
    results = {aid: assignment_analytics.fetch_assignment(aid) for aid in ids}
    return jsonify({"status": "success", "data": results})


@app.route("/api/analytics/assignment/fetch/all", methods=["GET"])
def fetch_assignments_all():
    return jsonify({"status": "success", "data": assignment_analytics.update_all()})


# ---------------- Difficulty Analytics ----------------
@app.route("/api/analytics/difficulty/student/<int:user_id>", methods=["GET"])
def fetch_student_difficulty(user_id):
    return jsonify(
        {
            "status": "success",
            "data": student_difficulty_analytics.get_user_all_levels(user_id),
        }
    )


@app.route("/api/analytics/difficulty/instructor/<int:user_id>", methods=["GET"])
def fetch_instructor_difficulty(user_id):
    return jsonify(
        {
            "status": "success",
            "data": instructor_difficulty_analytics.get_user_all_levels(user_id),
        }
    )


@app.route("/api/analytics/difficulty/student/update/<int:user_id>", methods=["POST"])
@role_required("admin", "student")
def update_student_difficulty(user_id):
    ok = student_difficulty_analytics.update_user_stats(user_id)
    return jsonify({"status": "success", "updated": ok})


@app.route(
    "/api/analytics/difficulty/instructor/update/<int:user_id>", methods=["POST"]
)
@role_required("admin", "instructor")
def update_instructor_difficulty(user_id):
    ok = instructor_difficulty_analytics.update_user_stats(user_id)
    return jsonify({"status": "success", "updated": ok})


@app.route("/api/analytics/difficulty/student/update/all", methods=["POST"])
@role_required("admin")
def update_all_students_difficulty():
    return jsonify(
        {
            "status": "success",
            "updated": student_difficulty_analytics.update_all_users(),
        }
    )


@app.route("/api/analytics/difficulty/instructor/update/all", methods=["POST"])
@role_required("admin")
def update_all_instructors_difficulty():
    return jsonify(
        {
            "status": "success",
            "updated": instructor_difficulty_analytics.update_all_users(),
        }
    )


# ---------------- Performance Analytics ----------------
@app.route("/api/analytics/performance/student/<int:user_id>", methods=["GET"])
def fetch_student_performance(user_id):
    return jsonify(
        {
            "status": "success",
            "data": student_performance_analytics.get_user_performance(user_id),
        }
    )


@app.route("/api/analytics/performance/instructor/<int:user_id>", methods=["GET"])
def fetch_instructor_performance(user_id):
    return jsonify(
        {
            "status": "success",
            "data": instructor_performance_analytics.get_user_performance(user_id),
        }
    )


@app.route("/api/analytics/performance/student/update/<int:user_id>", methods=["POST"])
@role_required("admin", "student")
def update_student_performance(user_id):
    ok = student_performance_analytics.update_user(user_id)
    return jsonify({"status": "success", "updated": ok})


@app.route(
    "/api/analytics/performance/instructor/update/<int:user_id>", methods=["POST"]
)
@role_required("admin", "instructor")
def update_instructor_performance(user_id):
    ok = instructor_performance_analytics.update_user(user_id)
    return jsonify({"status": "success", "updated": ok})


@app.route("/api/analytics/performance/student/update/all", methods=["POST"])
@role_required("admin")
def update_all_students_performance():
    return jsonify(
        {"status": "success", "updated": student_performance_analytics.update_all()}
    )


@app.route("/api/analytics/performance/instructor/update/all", methods=["POST"])
@role_required("admin")
def update_all_instructors_performance():
    return jsonify(
        {"status": "success", "updated": instructor_performance_analytics.update_all()}
    )


# ---------------- Grade Distribution ----------------
@app.route("/api/analytics/grades/<int:user_id>", methods=["GET"])
def fetch_user_grades(user_id):
    data = grade_distribution_analytics.get_distribution(user_id)
    if not data:
        return (
            jsonify(
                {"status": "error", "message": f"No grades found for user {user_id}"}
            ),
            404,
        )
    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/student", methods=["GET"])
@role_required("student")
def api_student_grades():
    user = session.get("user")
    student_id = user["user_id"]
    data = grade_distribution_analytics.get_distribution(student_id)
    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/<int:user_id>/<grade>", methods=["GET"])
def fetch_user_single_grade(user_id, grade):
    grade_map = {
        "A": "grade_a",
        "B": "grade_b",
        "C": "grade_c",
        "D": "grade_d",
        "E": "grade_e",
        "F": "grade_f",
    }
    col = grade_map.get(grade.upper())
    if not col:
        return jsonify({"status": "error", "message": "Invalid grade"}), 400
    count = grade_distribution_analytics.get_user_grade_column(user_id, col)
    return jsonify({"status": "success", "data": {grade: count or 0}})


@app.route("/api/analytics/grades/update/<int:user_id>/<grade>", methods=["POST"])
def update_user_grade(user_id, grade):
    grade_map = {
        "A": "grade_a",
        "B": "grade_b",
        "C": "grade_c",
        "D": "grade_d",
        "E": "grade_e",
        "F": "grade_f",
    }
    col = grade_map.get(grade.upper())
    if not col:
        return jsonify({"status": "error", "message": "Invalid grade"}), 400
    ok = grade_distribution_analytics.increment_user_grade(user_id, col)
    return jsonify({"status": "success", "updated": ok})


# ---------------------------
# 📊 Admin Grade Distribution Routes
# ---------------------------
@app.route("/api/analytics/grades/admin/overall", methods=["GET"])
@role_required("admin")
def api_admin_overall_grades():
    """System-wide grade distribution"""
    data = grade_distribution_analytics.get_overall_distribution()
    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/admin/trends", methods=["GET"])
@role_required("admin")
def api_admin_trends():
    """Grade trends over time"""
    interval = request.args.get("interval", "day")  # 'day' | 'week' | 'month'
    data = grade_distribution_analytics.get_chart_data_trend(interval)
    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/admin/group/<role>", methods=["GET"])
@role_required("admin")
def api_admin_group_grades(role):
    """Distributions grouped by role: student / instructor"""
    if role not in ["student", "instructor"]:
        return jsonify({"status": "error", "message": "Invalid role"}), 400
    data = grade_distribution_analytics.get_group_distribution_charts(role)
    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/admin/search", methods=["GET"])
@role_required("admin")
def api_admin_search_grades():
    """Search distribution for a specific user (student or instructor)"""
    role = request.args.get("role")
    identifier = request.args.get("id_or_email")

    if not role or not identifier:
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    if role not in ["student", "instructor"]:
        return jsonify({"status": "error", "message": "Invalid role"}), 400

    data = grade_distribution_analytics.search_distribution(role, identifier)
    if not data:
        return jsonify({"status": "error", "message": "No data found"}), 404

    return jsonify({"status": "success", "data": data})


# ---------------------------
# 📊 Instructor Grade Distribution Routes
# ---------------------------
@app.route("/api/grade/distribution/assignment/<int:assignment_id>")
@role_required("admin", "instructor")
def api_assignment_distribution(assignment_id):
    try:
        dist = grade_distribution_analytics.get_chart_data_assignment(assignment_id)
        return jsonify({"status": "success", "data": dist})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analytics/grades/instructor/overall", methods=["GET"])
@role_required("instructor")
def api_instructor_all_students():
    """Overall distribution for all students managed by the instructor"""
    user = session.get("user")
    # instructor_id = user["user_id"]

    # If you have a direct "instructor distribution", you can call that.
    # Otherwise, fetch group distribution for students.
    data = grade_distribution_analytics.get_group_distribution_charts("student")

    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/instructor/search", methods=["GET"])
@role_required("instructor")
def api_instructor_search_student():
    """Search a student distribution by ID/email (within instructor’s scope)"""
    identifier = request.args.get("id_or_email")
    if not identifier:
        return jsonify({"status": "error", "message": "Missing identifier"}), 400

    data = grade_distribution_analytics.search_distribution("student", identifier)
    if not data:
        return jsonify({"status": "error", "message": "No data found"}), 404

    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/instructor/self", methods=["GET"])
@role_required("instructor")
def api_instructor_self_distribution():
    """Distribution of the instructor’s own grades (profile view)"""
    user = session.get("user")
    instructor_id = user["user_id"]
    data = grade_distribution_analytics.get_distribution(instructor_id)
    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/admin/aggregate/<role>", methods=["GET"])
@role_required("admin")
def api_admin_aggregate_role(role):
    if role not in ["student", "instructor"]:
        return jsonify({"status": "error", "message": "Invalid role"}), 400
    data = grade_distribution_analytics.get_aggregated_distribution(role)
    return jsonify({"status": "success", "data": data})


# ---------------- System Analytics ----------------


@app.route("/api/analytics/system/fetch", methods=["GET"])
def fetch_system_analytics():
    try:
        snapshot = system_analytics.fetch_latest_snapshot()
        if snapshot:
            return jsonify({"status": "success", "data": snapshot})

        # fallback if no snapshot yet
        data = system_analytics.collect_data()
        system_analytics.save_snapshot(data)
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        app.logger.exception("❌ Failed to fetch system analytics snapshot")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analytics/system/history", methods=["GET"])
@role_required("admin")
def fetch_system_analytics_history():
    try:
        rows = system_analytics.fetch_all_snapshots()
        return jsonify({"status": "success", "data": rows})
    except Exception as e:
        app.logger.exception("❌ Failed to fetch system analytics history")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analytics/system/update", methods=["POST"])
@role_required("admin")
def update_system_analytics():
    try:
        data = system_analytics.collect_data()
        ok = system_analytics.save_snapshot(data)
        return jsonify({"status": "success", "updated": ok, "snapshot": data})
    except Exception as e:
        app.logger.exception("❌ Failed to update system analytics")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================#
# =============================================================================#
# =============================================================================#
# ============Admin control Panel==============================================#
# =============================================================================#
# =============================================================================#
@app.route("/admin_dashboard")
@role_required("admin")
def admin_dashboard():
    try:
        login_logs = AdminControlPanel.get_login_logs()
        broadcast_notification = AdminControlPanel.get_broadcasts()

        # fetch issue groups
        open_issues = AdminControlPanel.get_reported_issues(status_group="open")
        other_issues = AdminControlPanel.get_reported_issues(status_group="other")

        return render_template(
            "admin_dashboard.html",
            login_logs=login_logs,
            broadcast_notification=broadcast_notification,
            open_issues=open_issues,
            other_issues=other_issues,
            issues=open_issues + other_issues,  # ✅ safe for JS
        )
    except Exception as e:
        app.logger.error("❌ Failed to load admin dashboard: %s", e)
        return render_template(
            "admin_dashboard.html",
            login_logs=[],
            notifications=[],
            open_issues=[],
            other_issues=[],
            error="Something went wrong loading dashboard data",
        )


@app.route("/admin/update_issue_status/<int:issue_id>", methods=["POST"])
def update_issue_status(issue_id):
    if "user" not in session or session["user"]["role"] != "admin":
        return jsonify(success=False, message="Unauthorized"), 403
    try:
        data = request.get_json(silent=True) or {}
        new_status = data.get("status")
        if not new_status:
            return jsonify(success=False, message="Missing status"), 400

        # ✅ Allowed statuses safeguard
        allowed = {
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
        if new_status not in allowed:
            return jsonify(success=False, message="Invalid status"), 400

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE Reported_Issue
                    SET status=%s,
                        resolved_at = CASE
                            WHEN %s IN ('RESOLVED','CLOSED','REJECTED')
                            THEN NOW()
                            ELSE NULL
                        END
                    WHERE issue_id=%s
                    """,
                    (new_status, new_status, issue_id),
                )
            conn.commit()
            return jsonify(success=True)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    except Exception as e:
        app.logger.error("❌ Failed to update issue %s: %s", issue_id, e)
        return jsonify(success=False, message=str(e)), 500


@app.route("/admin/fetch_notifications")
@role_required("admin")
def fetch_notifications_admin():
    try:
        limit = int(request.args.get("limit", 50))
        notes = AdminControlPanel.get_notifications(limit=limit)
        return jsonify({"success": True, "data": notes})
    except Exception as e:
        app.logger.error("❌ Failed to fetch notifications: %s", e)
        return jsonify({"success": False, "message": str(e)}), 500


# ✅ Optional: Filter endpoint for pagination / future scaling
@app.route("/admin/issues/filter/<string:group>")
@role_required("admin")
def filter_issues(group):
    try:
        issues = AdminControlPanel.get_reported_issues(status_group=group, limit=100)
        return jsonify({"success": True, "data": issues})
    except Exception as e:
        app.logger.error("❌ Failed to fetch issues for %s: %s", group, e)
        return jsonify(success=False, message=str(e)), 500


@app.route("/admin/search/notifications")
@role_required("admin")
def search_notifications():
    email = request.args.get("email")
    user_id = request.args.get("user_id", type=int)

    if email and not user_id:
        user_id = AdminControlPanel.get_user_id_by_email(email)
        if not user_id:
            return jsonify({"success": False, "message": "No user found"}), 404

    if not user_id:
        return jsonify({"success": False, "message": "Provide email or user_id"}), 400

    notes = AdminControlPanel.get_notifications_by_user(user_id)
    return jsonify({"success": True, "data": notes})


@app.route("/admin/search/issues")
@role_required("admin")
def search_issues():
    email = request.args.get("email")
    user_id = request.args.get("user_id", type=int)

    if email and not user_id:
        user_id = AdminControlPanel.get_user_id_by_email(email)
        if not user_id:
            return jsonify({"success": False, "message": "No user found"}), 404

    if not user_id:
        return jsonify({"success": False, "message": "Provide email or user_id"}), 400

    issues = AdminControlPanel.get_reported_issues_by_user(user_id)  # ✅ used here
    return jsonify({"success": True, "data": issues})


# ---------------- Submissions ----------------
@app.route("/admin/submissions")
def admin_submissions():
    query = request.args.get("query")
    try:
        data = SubmissionAnalytics.list(query=query)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# ---------------- Assignments ----------------
@app.route("/admin/assignments")
def admin_assignments():
    sort = request.args.get("sort")
    repo_id = request.args.get("repo_id", type=int)
    assignment_id = request.args.get("assignment_id", type=int)
    try:
        data = AssignmentAnalytics.list(
            sort=sort,
            repo_id=repo_id,
            assignment_id=assignment_id,
        )
        return jsonify({"success": True, "data": data})
    except Exception as e:
        app.logger.error("❌ Failed to fetch assignments: %s", e)
        return jsonify({"success": False, "message": str(e)})


# ---------------- Instructors ----------------
@app.route("/admin/instructors")
def admin_instructors():
    search = request.args.get("search")
    score = request.args.get("score", type=float)  # exact score match

    try:
        data = UserAnalytics.list(
            role="instructor", search=search, score=score  # ✅ new arg
        )
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# ---------------- Students ----------------
@app.route("/admin/students")
def admin_students():
    search = request.args.get("search")
    sort = request.args.get("sort")
    try:
        data = UserAnalytics.list(role="student", search=search, sort=sort)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# ------------------ Global Template Data ------------------ #
@app.context_processor
def inject_config():
    return dict(config=Config)


# ------------------ App Runner ------------------ #
if __name__ == "__main__":
    app.run(debug=True)
