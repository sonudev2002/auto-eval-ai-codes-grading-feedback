# ============================================================
# app.py (Flask Application Entrypoint)
# Handles web routes, authentication, analytics, and backend integration
# ============================================================

import os
import sys
import logging
import secrets
import json
import subprocess
import time
import backend.reporting_and_issue as ri
from functools import wraps
from backend.db import get_connection
from backend.notification_system import NotificationSystem
from datetime import datetime
from dotenv import load_dotenv
from werkzeug.middleware.shared_data import SharedDataMiddleware
from typing import Any, Dict

# ============================================================
# 🧩 Admin & Backend Modules
# ============================================================
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

# Load environment variables from .env file
load_dotenv()

# Ensure backend modules are discoverable by Flask app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

# ============================================================
# ⚙️ Core Configurations & Imports
# ============================================================
from config import Config

# -------------------- User Management --------------------
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

# -------------------- Assignment Management --------------------
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

# -------------------- Code Submission & Evaluation --------------------
from backend.code_submission import (
    submit_code,
    CodeRunner,
    get_submission_details,
)
from backend.evaluation_pipeline import EvaluationPipeline
from notification_system import EmailDelivery

# -------------------- Analytics & Reporting --------------------
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
    AssignmentAnalyticsService,
)
from backend.grade_distribution import GradeDistributionManager

# ============================================================
# ⚙️ Flask App Initialization
# ============================================================
app = Flask(__name__, template_folder="frontend/templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))

# Initialize Grade Distribution Analytics Manager
grade_distribution_analytics = GradeDistributionManager()

# ============================================================
# 🧾 Logging Configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# 📂 Static File Serving & Uploads
# ============================================================
# Serve uploaded files under /uploads URL
app.wsgi_app = SharedDataMiddleware(
    app.wsgi_app,
    {"/uploads": os.path.join(os.path.abspath(os.path.dirname(__file__)), "uploads")},
)

# Enable CORS for frontend-backend communication
CORS(app)

# Ensure uploads directory exists
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# ============================================================
# 🌐 Public Routes (Basic Pages)
# ============================================================


@app.route("/")
def index():
    """Render the home page."""
    return render_template("index.html")


@app.route("/healthz")
def health_check():
    """Simple health check endpoint for uptime monitoring."""
    return {"status": "ok", "message": "Healthy"}, 200


@app.route("/about")
def about():
    """Render the About page."""
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    """Render the contact page and handle contact form submissions."""
    if request.method == "GET":
        return render_template("contact.html")

    # Capture form data
    name = request.form.get("name")
    email = request.form.get("email")
    message = request.form.get("message")

    if not all([name, email, message]):
        return jsonify({"status": "error", "message": "All fields are required."}), 400

    try:
        mailer = EmailDelivery()
        subject = f"New Contact Message from {name}"
        body = f"""
        <h3>📩 New Contact Message</h3>
        <p><b>Name:</b> {name}</p>
        <p><b>Email:</b> {email}</p>
        <p><b>Message:</b><br>{message}</p>
        <hr>
        <p>Sent from AutoEval Contact Page</p>
        """

        # Send email via Brevo to your admin inbox
        admin_email = os.getenv("EMAIL_SENDER", "aiyugabharat@aiyugabharat.com")
        mailer.send(admin_email, subject, body)

        return (
            jsonify({"status": "success", "message": "✅ Message sent successfully!"}),
            200,
        )

    except Exception as e:
        app.logger.exception("❌ Contact form email send failed")
        return (
            jsonify({"status": "error", "message": f"Failed to send message: {e}"}),
            500,
        )


@app.route("/privacy")
def privacy():
    """Render the Privacy Policy page."""
    return render_template("privacy.html")


# ============================================================
# 📂 File Handling Routes
# ============================================================


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded files from the /uploads directory."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ============================================================
# 🧠 Evaluation Pipeline Setup
# ============================================================
# Create a shared instance of EvaluationPipeline for code analysis
pipeline = EvaluationPipeline()


# ============================================================
# 🔐 Request Context Hook
# ============================================================


@app.before_request
def inject_user_role():
    """Attach current user's role to Flask's global `g` object."""
    user = session.get("user")
    g.current_user_role = user.get("role") if user else None


# ============================================================
# 🔐 Authentication & Session Management
# Handles login, logout, registration, and OTP verification
# ============================================================

# ------------------ Login & Logout ------------------ #


@app.route("/login", methods=["GET", "POST"])
def login():
    """Render login page (GET) or verify user credentials (POST)."""
    if request.method == "GET":
        return render_template("index.html")
    return user_verify()


@app.route("/logout")
def logout():
    """Logs out the current user and clears session data."""
    if "user" not in session:
        return redirect(url_for("index"))
    return user_logout(session)


# ------------------ Signup & OTP ------------------ #


@app.route("/signup", methods=["POST"])
def signup():
    """Registers a new user after OTP verification."""
    return register_user()


@app.route("/check-email")
def check_email_route():
    """Checks if the provided email is already registered."""
    return check_email()


@app.route("/send-otp", methods=["POST"])
def send_otp_route():
    """Sends OTP to user's email and mobile number for signup."""
    return send_otp()


@app.route("/verify-otp", methods=["POST"])
def verify_otp_route():
    """Verifies the OTP entered by the user."""
    return verify_otp()


@app.route("/send-otp-email", methods=["POST"])
def send_email_otp_route():
    """Sends OTP via email for password reset or verification."""
    return send_otp_email()


@app.route("/reset-password", methods=["GET", "POST"])
def reset_password_route():
    """Handles password reset requests (GET/POST)."""
    return change_password()


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


# ============================================================
# 🎓 Student Dashboard Routes
# ============================================================

# Initialize logger for this module
logger = logging.getLogger(__name__)


# -------------------- Main Student Dashboard --------------------
@app.route("/student_dashboard")
@role_required("student")
def student_dashboard():
    """
    Render the student dashboard page.
    Displays repositories, assignments, and notifications.
    """
    repo_id = request.args.get("repo_id", type=int)
    logger.info("📌 Student dashboard accessed. Repo ID: %s", repo_id)

    try:
        # Fetch repository details for sidebar
        repo_detail = AssignmentsStudent.get_repository_details()

        # Load assignments — either for a specific repo or all
        if repo_id:
            assignments = AssignmentsStudent.get_assignments_by_repo_detailed(repo_id)
        else:
            assignments = AssignmentsStudent.get_dashboard_assignment_detail()

        # Extract user info from session for notifications
        user = session.get("user")
        user_id = user.get("user_id") if user else None

        # Render dashboard template with relevant data
        return render_template(
            "student_dashboard.html",
            repo_detail=repo_detail,
            assignments=assignments,
            selected_repo=repo_id,
            now=datetime.now(),
            user_id=user_id,  # For frontend JS use
            current_user=user,  # Safe dictionary, not Flask-Login
        )

    except Exception:
        # Log error and display fallback page
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


# -------------------- Fetch Assignments for Repo --------------------
@app.route("/fetch_assignments_for_student")
@role_required("student")
def get_assignments_by_repo_id():
    """
    Returns all assignments for the selected repository (AJAX endpoint).
    """
    repo_id = request.args.get("repo_id", type=int)
    if not repo_id:
        return jsonify([])

    # Fetch assignments for given repository
    assignments = AssignmentsStudent.get_assignments_by_repo_detailed(repo_id)

    # Format due date for better frontend readability
    for a in assignments:
        if isinstance(a.get("due_date"), datetime):
            a["due_date"] = a["due_date"].strftime("%d %b %Y")

    return jsonify(assignments)


# ============================================================
# 👨‍🏫 Instructor Dashboard & Analytics
# ============================================================


@app.route("/instructor_dashboard")
def instructor_dashboard():
    """Render instructor dashboard page (redirect to login if unauthorized)."""
    user = session.get("user")
    if not user or user.get("role", "").strip().lower() != "instructor":
        return redirect(url_for("login"))
    return render_template("instructor_dashboard.html")


# ------------------------------------------------------------
# 📋 Instructor: Student List
# ------------------------------------------------------------
@app.route("/instructor/students")
def instructor_students():
    """Allow instructors to view list of students (filtered access)."""
    user_session = session.get("user")
    if not user_session or user_session.get("role") != "instructor":
        abort(403)

    search = request.args.get("search")
    sort = request.args.get("sort")

    try:
        data = UserAnalytics.list(role="student", search=search, sort=sort)
        # Optional: filter students by instructor if needed
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# ------------------------------------------------------------
# 📊 Instructor Dashboard API (Analytics + Caching)
# ------------------------------------------------------------

_dashboard_cache = {}
CACHE_TTL = 30  # seconds


@app.route("/api/instructor/dashboard")
@role_required("instructor")
def api_instructor_dashboard():
    """Return instructor analytics data with simple in-memory caching."""
    user = session.get("user")
    instructor_id = user.get("user_id")
    lite = request.args.get("lite", "0") == "1"

    cache_key = f"instructor_{instructor_id}_lite_{lite}"
    now = time.time()
    cached = _dashboard_cache.get(cache_key)
    if cached and (now - cached["time"]) < CACHE_TTL:
        return jsonify(cached["data"])

    try:
        data = {
            "assignment_analytics": instructor_analytics.get_assignment_analytics(
                instructor_id
            ),
            "performance_summary": instructor_analytics.get_student_performance_summary(
                instructor_id
            ),
            "status": "success",
        }

        # Include heavy charts only when lite mode is off
        if not lite:
            data["charts"] = {
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
            }

        _dashboard_cache[cache_key] = {"data": data, "time": now}
        return jsonify(data)

    except Exception as e:
        app.logger.exception("❌ Instructor dashboard API failed")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        instructor_analytics.close()


# ============================================================
# 📦 Assignment Management
# ============================================================


@app.route("/create-assignment")
def show_form():
    """Render assignment creation form (admin/instructor only)."""
    user = session.get("user")
    if not user or user.get("role") not in ("admin", "instructor"):
        return jsonify(success=False, message="Access denied"), 403
    return render_template("create_assignment.html")


@app.route("/create-repository", methods=["POST"])
def create_repository_route():
    """Create a new repository (admin only)."""
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
    """Fetch all repositories."""
    repos = get_all_repositories()
    return jsonify(repos)


@app.route("/delete-repository", methods=["POST"])
def delete_repository():
    """Delete a repository (admin only)."""
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
        return jsonify({"success": False, "message": message}), 500
    except Exception as e:
        logger.error("❌ Exception in delete_repository route: %s", e)
        return jsonify({"success": False, "message": "Internal server error."}), 500


@app.route("/api/assignments/<int:repo_id>")
def fetch_assignments(repo_id):
    """Fetch assignments by repository ID."""
    assignments = get_assignments_by_repo(repo_id)
    return jsonify(assignments)


@app.route("/api/assignment/<int:assignment_id>")
def fetch_assignment_details(assignment_id):
    """Fetch assignment details by assignment ID."""
    assignment = get_assignment_details(assignment_id)
    if assignment:
        return jsonify(assignment)
    return jsonify({"error": "Assignment not found"}), 404


@app.route("/upload-assignment", methods=["GET", "POST"])
def handle_submission():
    """Handle assignment upload form (instructors/admin only)."""
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

    csv_file = request.files.get("testcase_csv")
    success = upload_assignment(form_data, csv_file=csv_file)

    if success is True:
        return jsonify(success=True, message="Assignment uploaded successfully!")
    elif isinstance(success, dict):
        return (
            jsonify(success=False, message=success.get("message", "Upload failed")),
            400,
        )
    return jsonify(success=False, message="Upload failed due to server error."), 500


@app.route("/update-assignment", methods=["POST"])
def handle_update():
    """Update assignment details (admin/instructor only)."""
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
    """Delete assignment (admin/instructor only)."""
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
    return jsonify({"success": False, "message": "Deletion failed"})


# ============================================================
# 💻 Code Editor & Viewer
# ============================================================


@app.route("/view-code")
def view_code():
    """Serve source code from local storage or remote URL (e.g., Cloudinary)."""
    import requests
    from urllib.parse import urlparse

    rel_path = request.args.get("path")
    if not rel_path:
        return jsonify({"error": "No path provided"}), 400

    try:
        # --- Remote file fetch (e.g., Cloudinary raw link) ---
        if rel_path.startswith(("http://", "https://")):
            r = requests.get(rel_path, timeout=10)
            if r.status_code == 200:
                return jsonify({"code": r.text})
            return (
                jsonify({"error": f"Failed to fetch remote file ({r.status_code})"}),
                404,
            )

        # --- Local file read (submitted_codes folder) ---
        import os

        base_dir = os.path.join(app.root_path, "mca_final_project", "submitted_codes")
        safe_path = os.path.abspath(
            os.path.join(base_dir, rel_path.replace("\\", "/").lstrip("/"))
        )

        if not safe_path.startswith(base_dir) or not os.path.exists(safe_path):
            return jsonify({"error": "File not found"}), 404

        with open(safe_path, "r", encoding="utf-8") as f:
            return jsonify({"code": f.read()})

    except Exception as e:
        app.logger.exception("Error loading code")
        return jsonify({"error": f"Exception: {e}"}), 500


# ============================================================
# 🧾 Code Submission Report
# ============================================================


@app.route("/code_submission_report")
@role_required("student", "admin")
def code_submission_report():
    """Render detailed code submission report."""
    submission_id = request.args.get("submission_id", type=int)
    if not submission_id:
        return (
            render_template(
                "code_submission_report.html",
                result={"status": "error", "message": "❌ Missing submission_id"},
            ),
            400,
        )

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


# ============================================================
# 🧑‍💻 Code Editor View
# ============================================================


@app.route(
    "/code_editor/<int:assignment_id>", methods=["GET"], endpoint="open_code_editor"
)
@role_required("student", "admin", "instructor")
def open_code_editor(assignment_id):
    """Open in-browser code editor for the given assignment."""
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


# ============================================================
# ⚙️ Code Execution API
# ============================================================


@app.route("/run", methods=["POST"])
def run_code_route():
    """Execute submitted code in isolated container with sample inputs."""
    try:
        data = request.get_json()
        code = data.get("code")
        lang = data.get("lang", "python3")
        inputs = data.get("inputs", [])

        runner = CodeRunner(language=lang)
        runner.start_container(code)
        results = runner.run_multiple_inputs(inputs)
        return jsonify(results)
    except Exception as e:
        app.logger.error("Run error: %s", e)
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🚀 Code Submission API
# ============================================================


@app.route("/submit-code", methods=["POST"])
def submit_code():
    """Process code submission and trigger evaluation pipeline."""
    from backend.code_submission import submit_code as process_submission

    try:
        data = request.get_json(force=True)
        assignment_id = int(data.get("assignment_id"))
        student_id = int(session.get("user", {}).get("user_id", 0))
        source_code = data.get("source_code", "")
        language = data.get("language", "python3")

        if not student_id:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401

        result = process_submission(assignment_id, student_id, source_code, language)

        # --- Syntax Error ---
        if result.get("stage") == "syntax_check":
            return jsonify(result), 400

        # --- Successful Evaluation ---
        if result.get("status") == "success":
            report_url = url_for(
                "code_submission_report", submission_id=result["submission_id"]
            )
            return jsonify({"status": "success", "report_url": report_url}), 200

        # --- General Evaluation Error ---
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Evaluation failed",
                    "details": result,
                }
            ),
            500,
        )

    except Exception as e:
        app.logger.exception("Error in /submit-code: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# ⭐ Feedback Submission
# ============================================================


@app.route("/api/feedback/submit", methods=["POST"])
@role_required("student")
def submit_feedback():
    """Submit feedback score for evaluated submission."""
    try:
        data = request.get_json(force=True)
        submission_id = data.get("submission_id")
        score = data.get("score")

        if not submission_id or score is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Missing submission_id or score",
                    }
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


# ============================================================
# 🧹 Code Formatters
# ============================================================


@app.route("/format/python", methods=["POST"])
def format_python():
    """Format Python code using Black."""
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
    """Format C++ code using clang-format."""
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
    """Format Java code using Google Java Formatter."""
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


# ============================================================
# 👤 Profile Management
# ============================================================

profile_bp = Blueprint("profile", __name__)


@profile_bp.route("/<int:user_id>", methods=["GET"])
def profile(user_id):
    """Display user profile (role-based: student, instructor, admin)."""
    user_session = session.get("user")
    if not user_session:
        flash("You must be logged in to view profile.", "danger")
        return redirect(url_for("login"))

    if user_session["role"] != "admin" and user_session["user_id"] != user_id:
        abort(403)

    session_role = user_session["role"]
    target_role = request.args.get("role", session_role)

    # --- Student Profile ---
    if target_role == "student":
        pdata = StudentProfileData(user_id)
        data = pdata.get_basic_profile()
        pdata.close()
        return render_template(
            "profile.html", user=data["user"], profile=data["profile"]
        )

    # --- Instructor Profile ---
    elif target_role == "instructor":
        pdata = InstructorProfileData(user_id)
        data = pdata.get_basic_profile()
        pdata.close()
        return render_template(
            "profile.html", user=data["user"], profile=data["profile"]
        )

    # --- Admin Profile ---
    elif target_role == "admin":
        pdata = AdminProfileData(user_id)
        data = pdata.get_profile_data()
        pdata.close()
        return render_template(
            "profile.html",
            user=data["user"],
            profile=data["profile"],
            completed_assignments={},  # Admins have no assignment data
        )

    abort(403)


# ============================================================
# 📈 Performance History
# ============================================================


@app.route("/performance_history/<int:user_id>")
def performance_history(user_id):
    """Display performance analytics for student/instructor/admin."""
    user_session = session.get("user")
    if not user_session:
        return redirect(url_for("login"))

    viewer_role = user_session.get("role")
    viewer_id = user_session.get("user_id")

    # --- Admin: Can view any user ---
    if viewer_role == "admin":
        target_role = request.args.get("role", "student").lower()
        if target_role == "instructor":
            pdata = InstructorProfileData(user_id)
        else:
            pdata = StudentProfileData(user_id)

    # --- Instructor: Can view self or students ---
    elif viewer_role == "instructor":
        target_role = request.args.get("role", "student").lower()
        if int(user_id) == int(viewer_id):
            pdata = InstructorProfileData(user_id)
        else:
            pdata = StudentProfileData(user_id)

    # --- Student: Can view self only ---
    elif viewer_role == "student":
        if int(user_id) != int(viewer_id):
            abort(403)
        pdata = StudentProfileData(user_id)

    else:
        abort(403)

    # Fetch performance data
    perf_data = pdata.get_performance_data()
    user_info = pdata.get_user_info()
    pdata.close()
    report = perf_data.get("report", {})
    completed_assignments = perf_data.get("completed_assignments", {}) or perf_data.get(
        "managed_assignments", {}
    )

    return render_template(
        "performance_history.html",
        user=user_info,
        report=report,
        completed_assignments=completed_assignments,
    )


# ============================================================
# ✏️ Profile Update Endpoints
# ============================================================


@profile_bp.route("/<int:user_id>/update_picture", methods=["POST"])
def update_picture(user_id):
    """Update or delete user profile picture."""
    user_session = session.get("user")
    if not user_session:
        abort(403)

    if user_session["role"] == "admin" and user_session["user_id"] != user_id:
        flash("Admins can only view other profiles, not update them.", "warning")
        return redirect(
            url_for("profile.profile", user_id=user_id, role=request.form.get("role"))
        )

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
    """Update user profile information (name, address, etc.)."""
    user_session = session.get("user")
    if not user_session:
        abort(403)

    if user_session["role"] == "admin" and user_session["user_id"] != user_id:
        flash("Admins can only view other profiles, not update them.", "warning")
        return redirect(
            url_for("profile.profile", user_id=user_id, role=request.form.get("role"))
        )

    updater = UpdateProfileData(user_id)
    result = updater.update_info(request.form)
    flash(result["message"], "success" if result["status"] == "success" else "danger")

    return redirect(
        url_for("profile.profile", user_id=user_id, role=request.form.get("role"))
    )


@profile_bp.route("/<int:user_id>/update_password", methods=["POST"])
def update_password(user_id):
    """Update user password (with validation and role checks)."""
    user_session = session.get("user")
    if not user_session:
        abort(403)

    if user_session["role"] == "admin" and user_session["user_id"] != user_id:
        flash("Admins can only view other profiles, not update them.", "warning")
        return redirect(
            url_for("profile.profile", user_id=user_id, role=request.form.get("role"))
        )

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

# ============================================================
# 🐞 Issue Reporting & Resolution
# ============================================================


@app.route("/report_issue/form")
def report_issue_form():
    """Render issue reporting form for logged-in users."""
    if "user" not in session:
        flash("You must be logged in to report an issue.", "danger")
        return redirect(url_for("login"))
    user_id = session["user"]["user_id"]
    issues = ri.get_user_issues(user_id)
    return render_template("report_issue.html", user_id=user_id, issues=issues)


@app.route("/issue/<int:issue_id>/screenshots")
def issue_screenshots(issue_id):
    """Fetch screenshots linked to a specific issue."""
    screenshots = ri.get_screenshots(issue_id)
    return jsonify({"screenshots": screenshots})


# ------------------------------------------------------------
# 📨 Submit Issue
# ------------------------------------------------------------
@app.route("/report_issue/submit", methods=["POST"])
def submit_issue():
    """Handle issue submission from students/instructors/admins."""
    if "user" not in session:
        flash("You must be logged in to submit an issue.", "danger")
        return redirect(url_for("login"))

    user_id = session["user"]["user_id"]
    issue_type = request.form["issue_type"]
    description = request.form["description"]
    screenshots = request.files.getlist("screenshots")

    result = ri.submit_issue(user_id, issue_type, description, screenshots)

    flash(
        (
            "Issue reported successfully!"
            if result["success"]
            else f"Error: {result['message']}"
        ),
        "success" if result["success"] else "danger",
    )

    # Redirect based on user role
    role = session["user"]["role"]
    if role == "student":
        return redirect(url_for("student_dashboard"))
    elif role == "instructor":
        return redirect(url_for("instructor_dashboard"))
    elif role == "admin":
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("login"))


# ------------------------------------------------------------
# 📋 User: View My Reported Issues
# ------------------------------------------------------------
@app.route("/my_issues")
def my_issues():
    """Allow users to track their submitted issues."""
    if "user" not in session:
        flash("Please log in to view your issues.", "danger")
        return redirect(url_for("login"))
    user_id = session["user"]["user_id"]
    issues = ri.get_user_issues(user_id)
    return render_template("my_issues.html", issues=issues)


# ------------------------------------------------------------
# 🛠️ Admin: Manage Reported Issues
# ------------------------------------------------------------
@app.route("/admin/issues")
def admin_issues():
    """Display all reported issues to admin users."""
    if "user" not in session or session["user"]["role"] != "admin":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))
    issues = ri.get_all_issues()
    return render_template("admin_issues.html", issues=issues)


@app.route("/admin/resolve/<int:issue_id>", methods=["POST"])
def resolve_issue(issue_id):
    """Mark an issue as resolved (admin only)."""
    if "user" not in session or session["user"]["role"] != "admin":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    result = ri.resolve_issue(issue_id)
    flash(
        (
            "Issue marked as resolved ✅"
            if result["success"]
            else f"Error: {result['message']}"
        ),
        "success" if result["success"] else "danger",
    )
    return redirect(url_for("admin_issues"))


# ============================================================
# 🔔 Notification System
# ============================================================

# Initialize notification handler
notifier = NotificationSystem(use_background=True)


@app.route("/create_notification")
@role_required("admin")
def creator_admin():
    """Render form to create admin notifications."""
    return render_template("create_notification.html")


# ------------------------------------------------------------
# 🎯 Targeted Notifications
# ------------------------------------------------------------
@app.route("/notifications/send", methods=["POST"])
@role_required("admin")
def send_notification():
    """Send targeted notifications (single, group, or role-based)."""
    try:
        is_json = request.is_json
        data = request.get_json(silent=True) or request.form.to_dict(flat=True)

        # --- Validate message ---
        message = (data.get("message") or "").strip()
        if not message:
            raise ValueError("Message cannot be empty.")

        rec_value = data.get("recipients")
        recipients = []

        # Single user
        if rec_value == "single":
            user_id = data.get("user_id")
            if not user_id:
                raise ValueError("User ID required for single recipient.")
            user_id = int(user_id)
            email = notifier.users.get_user_email(user_id)
            mobile = notifier.users.get_user_mobile_number(user_id)
            if not (email or mobile):
                raise ValueError(f"User with ID {user_id} does not exist.")
            recipients = [user_id]

        # Group of users
        elif rec_value == "group":
            user_ids_raw = data.get("user_ids")
            if not user_ids_raw:
                raise ValueError("User IDs required for group recipients.")
            user_ids = [
                int(uid.strip()) for uid in user_ids_raw.split(",") if uid.strip()
            ]
            valid_ids = [
                uid
                for uid in user_ids
                if notifier.users.get_user_email(uid)
                or notifier.users.get_user_mobile_number(uid)
            ]
            if not valid_ids:
                raise ValueError("None of the provided user IDs exist.")
            recipients = valid_ids

        # Role-based broadcast
        elif rec_value in (
            "students",
            "instructors",
            "all",
            "students_and_instructors",
        ):
            recipients = [rec_value]
        else:
            raise ValueError("Invalid recipient type.")

        # --- Channels ---
        channels = (
            request.form.getlist("channels")
            if not is_json
            else data.get("channels", ["dashboard"])
        )
        if not channels:
            raise ValueError("At least one channel must be selected.")

        # --- Send notification ---
        notifier.send_message(
            sender_role=str(data.get("sender_role", "admin")),
            sender_id=int(data.get("sender_id", 0)),
            message=message,
            recipients=recipients,
            channels=channels,
            subject=data.get("subject"),
            notif_type=data.get("type", "info"),
        )

        # --- Response ---
        if is_json:
            return jsonify({"status": "success"}), 200
        flash("✅ Notification sent successfully!", "success")
        return redirect(url_for("creator_admin"))

    except Exception as e:
        app.logger.exception("Failed to send notification")
        if request.is_json:
            return jsonify({"status": "error", "error": str(e)}), 400
        flash(f"❌ Failed to send notification: {str(e)}", "danger")
        return redirect(url_for("creator_admin"))


# ------------------------------------------------------------
# 📢 Broadcast Notifications
# ------------------------------------------------------------
@app.route("/notifications/broadcast", methods=["POST"])
@role_required("admin")
def broadcast_notification():
    """Broadcast system-wide messages to all users."""
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
# 🔔 User Notification APIs
# ============================================================


@app.route("/notifications/<int:user_id>", methods=["GET"])
def fetch_notifications(user_id):
    """Fetch user-specific notifications (optionally filtered by status)."""
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
    """Mark a single notification as read."""
    try:
        notifier.mark_notification_read(notification_id)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        app.logger.exception("Mark read failed")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/notifications/<int:user_id>/count", methods=["GET"])
def count_unread(user_id):
    """Return the count of unread notifications for a user."""
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


@app.route("/notifications/mark_all_read/<int:user_id>", methods=["PUT"])
def mark_all_read(user_id):
    """Mark all notifications as read for a given user."""
    try:
        notifier.mark_all_as_read(user_id)
        return (
            jsonify(
                {
                    "status": "success",
                    "message": "All notifications marked as read",
                }
            ),
            200,
        )
    except Exception as e:
        app.logger.exception("Mark all as read failed")
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# 📰 Unified Notification Feed
# ============================================================


@app.route("/notifications/feed/<int:user_id>", methods=["GET"])
def unified_feed(user_id):
    """Return a combined feed of personal and broadcast notifications."""
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

        # Broadcast notifications
        broadcasts = notifier.fetch_broadcasts_for_role(role, limit, offset) or []
        for b in broadcasts:
            b["source"] = "broadcast"

        combined = notifications + broadcasts

        # Sort by creation time (descending)
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

        # Convert datetimes to ISO format
        for entry in combined:
            ca = entry.get("created_at")
            if isinstance(ca, datetime):
                entry["created_at"] = ca.isoformat()

        return jsonify({"user_id": user_id, "role": role, "feed": combined}), 200
    except Exception as e:
        app.logger.exception("Unified feed fetch failed")
        return jsonify({"status": "error", "error": str(e)}), 500


# ============================================================
# 📘 Assignment Analytics APIs
# ============================================================


@app.route("/api/analytics/assignment/update/group", methods=["POST"])
@role_required("admin", "instructor")
def update_assignments_group():
    """Update analytics for a group of specific assignments."""
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
    """Recalculate analytics for all assignments."""
    return jsonify({"status": "success", "updated": assignment_analytics.update_all()})


@app.route("/api/analytics/assignment/fetch/<int:assignment_id>", methods=["GET"])
def fetch_assignment_analytics(assignment_id):
    """Fetch detailed analytics for a specific assignment."""
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
    """Fetch analytics for multiple assignments by IDs."""
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
    """Fetch analytics for all assignments."""
    return jsonify({"status": "success", "data": assignment_analytics.update_all()})


# ============================================================
# 🧩 Difficulty Analytics Routes
# ============================================================


@app.route("/api/analytics/difficulty/student/<int:user_id>", methods=["GET"])
def fetch_student_difficulty(user_id):
    """Fetch difficulty-level analytics for a specific student."""
    return jsonify(
        {
            "status": "success",
            "data": student_difficulty_analytics.get_user_all_levels(user_id),
        }
    )


@app.route("/api/analytics/difficulty/instructor/<int:user_id>", methods=["GET"])
def fetch_instructor_difficulty(user_id):
    """Fetch difficulty-level analytics for a specific instructor."""
    return jsonify(
        {
            "status": "success",
            "data": instructor_difficulty_analytics.get_user_all_levels(user_id),
        }
    )


@app.route("/api/analytics/difficulty/student/update/<int:user_id>", methods=["POST"])
@role_required("admin", "student")
def update_student_difficulty(user_id):
    """Update difficulty statistics for an individual student."""
    ok = student_difficulty_analytics.update_user_stats(user_id)
    return jsonify({"status": "success", "updated": ok})


@app.route(
    "/api/analytics/difficulty/instructor/update/<int:user_id>", methods=["POST"]
)
@role_required("admin", "instructor")
def update_instructor_difficulty(user_id):
    """Update difficulty statistics for an individual instructor."""
    ok = instructor_difficulty_analytics.update_user_stats(user_id)
    return jsonify({"status": "success", "updated": ok})


@app.route("/api/analytics/difficulty/student/update/all", methods=["POST"])
@role_required("admin")
def update_all_students_difficulty():
    """Recalculate difficulty analytics for all students."""
    return jsonify(
        {
            "status": "success",
            "updated": student_difficulty_analytics.update_all_users(),
        }
    )


@app.route("/api/analytics/difficulty/instructor/update/all", methods=["POST"])
@role_required("admin")
def update_all_instructors_difficulty():
    """Recalculate difficulty analytics for all instructors."""
    return jsonify(
        {
            "status": "success",
            "updated": instructor_difficulty_analytics.update_all_users(),
        }
    )


# ============================================================
# ⚡ Performance Analytics Routes
# ============================================================


@app.route("/api/analytics/performance/student/<int:user_id>", methods=["GET"])
def fetch_student_performance(user_id):
    """Fetch individual student performance metrics."""
    return jsonify(
        {
            "status": "success",
            "data": student_performance_analytics.get_user_performance(user_id),
        }
    )


@app.route("/api/analytics/performance/instructor/<int:user_id>", methods=["GET"])
def fetch_instructor_performance(user_id):
    """Fetch instructor performance metrics."""
    return jsonify(
        {
            "status": "success",
            "data": instructor_performance_analytics.get_user_performance(user_id),
        }
    )


@app.route("/api/analytics/performance/student/update/<int:user_id>", methods=["POST"])
@role_required("admin", "student")
def update_student_performance(user_id):
    """Update analytics for a specific student."""
    ok = student_performance_analytics.update_user(user_id)
    return jsonify({"status": "success", "updated": ok})


@app.route(
    "/api/analytics/performance/instructor/update/<int:user_id>", methods=["POST"]
)
@role_required("admin", "instructor")
def update_instructor_performance(user_id):
    """Update analytics for a specific instructor."""
    ok = instructor_performance_analytics.update_user(user_id)
    return jsonify({"status": "success", "updated": ok})


@app.route("/api/analytics/performance/student/update/all", methods=["POST"])
@role_required("admin")
def update_all_students_performance():
    """Recalculate performance for all students (admin only)."""
    return jsonify(
        {
            "status": "success",
            "updated": student_performance_analytics.update_all(),
        }
    )


@app.route("/api/analytics/performance/instructor/update/all", methods=["POST"])
@role_required("admin")
def update_all_instructors_performance():
    """Recalculate performance for all instructors (admin only)."""
    return jsonify(
        {
            "status": "success",
            "updated": instructor_performance_analytics.update_all(),
        }
    )


# ============================================================
# 🧮 Grade Distribution Routes
# ============================================================


@app.route("/api/analytics/grades/profile/<int:user_id>", methods=["GET"])
def fetch_user_grades(user_id):
    """Fetch full grade distribution for a specific user."""
    data = grade_distribution_analytics.get_distribution(user_id)
    if not data:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"No grades found for user {user_id}",
                }
            ),
            404,
        )
    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/student", methods=["GET"])
@role_required("student")
def api_student_grades():
    """Fetch logged-in student’s grade distribution."""
    user = session.get("user")
    student_id = user["user_id"]
    data = grade_distribution_analytics.get_distribution(student_id)
    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/<int:user_id>/<grade>", methods=["GET"])
def fetch_user_single_grade(user_id, grade):
    """Fetch count of a specific grade (A-F) for a user."""
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
    """Increment the count for a specific grade (A-F) for a user."""
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


# ============================================================
# 🧾 Admin Grade Distribution Routes
# ============================================================


@app.route("/api/analytics/grades/admin/overall", methods=["GET"])
@role_required("admin")
def api_admin_overall_grades():
    """Return overall system-wide grade distribution (admin view)."""
    data = grade_distribution_analytics.get_overall_distribution()
    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/admin/trends", methods=["GET"])
@role_required("admin")
def api_admin_trends():
    """Return grade trend data over time (daily, weekly, or monthly)."""
    interval = request.args.get("interval", "day")
    data = grade_distribution_analytics.get_chart_data_trend(interval)
    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/admin/group/<role>", methods=["GET"])
@role_required("admin")
def api_admin_group_grades(role):
    """Return grade distribution grouped by role (student or instructor)."""
    if role not in ["student", "instructor"]:
        return jsonify({"status": "error", "message": "Invalid role"}), 400

    data = grade_distribution_analytics.get_group_distribution_charts(role)
    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/admin/search", methods=["GET"])
@role_required("admin")
def api_admin_search_grades():
    """Search and fetch grade distribution for a specific user."""
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


# ============================================================
# 📊 Instructor Grade Distribution Routes
# ============================================================


@app.route("/api/grade/distribution/assignment/<int:assignment_id>")
@role_required("admin", "instructor")
def api_assignment_distribution(assignment_id):
    """Return grade distribution data for a specific assignment."""
    try:
        dist = grade_distribution_analytics.get_chart_data_assignment(assignment_id)
        return jsonify({"status": "success", "data": dist})
    except Exception as e:
        app.logger.exception("Grade distribution failed")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analytics/grades/instructor/overall", methods=["GET"])
@role_required("instructor")
def api_instructor_all_students():
    """Return overall grade distribution for all students under an instructor."""
    user = session.get("user")
    data = grade_distribution_analytics.get_group_distribution_charts("student")
    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/instructor/search", methods=["GET"])
@role_required("instructor")
def api_instructor_search_student():
    """Search and fetch a student’s grade distribution by ID or email."""
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
    """Return the instructor’s own grade distribution (self-analysis)."""
    user = session.get("user")
    instructor_id = user["user_id"]
    data = grade_distribution_analytics.get_distribution(instructor_id)
    return jsonify({"status": "success", "data": data})


@app.route("/api/analytics/grades/admin/aggregate/<role>", methods=["GET"])
@role_required("admin")
def api_admin_aggregate_role(role):
    """Return aggregated grade distributions for all students/instructors."""
    if role not in ["student", "instructor"]:
        return jsonify({"status": "error", "message": "Invalid role"}), 400
    data = grade_distribution_analytics.get_aggregated_distribution(role)
    return jsonify({"status": "success", "data": data})


# ============================================================
# 🧠 System Analytics Routes
# ============================================================


@app.route("/api/analytics/system/fetch", methods=["GET"])
def fetch_system_analytics():
    """Fetch latest system analytics snapshot (collect if missing)."""
    try:
        snapshot = system_analytics.fetch_latest_snapshot()
        if snapshot:
            return jsonify({"status": "success", "data": snapshot})

        # Fallback: collect new data if no snapshot found
        data = system_analytics.collect_data()
        system_analytics.save_snapshot(data)
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        app.logger.exception("❌ Failed to fetch system analytics snapshot")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analytics/system/history", methods=["GET"])
@role_required("admin")
def fetch_system_analytics_history():
    """Retrieve system analytics history snapshots (admin only)."""
    try:
        rows = system_analytics.fetch_all_snapshots()
        return jsonify({"status": "success", "data": rows})
    except Exception as e:
        app.logger.exception("❌ Failed to fetch system analytics history")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/analytics/system/update", methods=["POST"])
@role_required("admin")
def update_system_analytics():
    """Update assignment analytics and refresh overall system analytics."""
    try:
        # Step 1️⃣: Update all assignment-level analytics
        assignment_analytics = AssignmentAnalyticsService()
        updated_count = assignment_analytics.update_all()
        app.logger.info(f"✅ Updated analytics for {updated_count} assignments")

        # Step 2️⃣: Collect and save new system-wide analytics snapshot
        data = system_analytics.collect_data()
        ok = system_analytics.save_snapshot(data)

        return jsonify(
            {
                "status": "success",
                "updated_assignments": updated_count,
                "system_snapshot_saved": ok,
                "snapshot": data,
                "message": f"System + {updated_count} assignment analytics updated successfully",
            }
        )
    except Exception as e:
        app.logger.exception("❌ Failed to update system analytics (with assignments)")
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# 🛠️ Admin Control Panel Routes
# ============================================================


# -------------------- Admin Dashboard --------------------
@app.route("/admin_dashboard")
@role_required("admin")
def admin_dashboard():
    """
    Admin dashboard:
    - Displays login logs, broadcasts, notifications, issues, and users.
    - Provides an overview of system activity and user management.
    """
    try:
        from admin_control_panel import AdminControlPanel

        # Fetch recent logs, broadcasts, notifications, and issues
        login_logs = AdminControlPanel.get_login_logs()
        broadcast = AdminControlPanel.get_broadcasts()
        notifications = AdminControlPanel.get_notifications()
        open_issues = AdminControlPanel.get_reported_issues(status_group="open")
        other_issues = AdminControlPanel.get_reported_issues(status_group="other")

        # Fetch all registered users
        conn = get_connection()
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(
                "SELECT user_id, first_name, last_name, email, role "
                "FROM user_profile ORDER BY user_id ASC"
            )
            users = cursor.fetchall()
        conn.close()

        # Render admin dashboard page
        return render_template(
            "admin_dashboard.html",
            login_logs=login_logs,
            broadcasts=broadcast,
            notifications=notifications,
            open_issues=open_issues,
            other_issues=other_issues,
            issues=open_issues + other_issues,
            users=users,
        )

    except Exception as e:
        app.logger.error("❌ Failed to load admin dashboard: %s", e)
        return render_template(
            "admin_dashboard.html",
            login_logs=[],
            notifications=[],
            open_issues=[],
            other_issues=[],
            users=[],
            error="Something went wrong loading dashboard data",
        )


# -------------------- Update Issue Status --------------------
@app.route("/admin/update_issue_status/<int:issue_id>", methods=["POST"])
def update_issue_status(issue_id):
    """Update issue status (admin only)."""
    if "user" not in session or session["user"]["role"] != "admin":
        return jsonify(success=False, message="Unauthorized"), 403

    try:
        data = request.get_json(silent=True) or {}
        new_status = data.get("status")
        if not new_status:
            return jsonify(success=False, message="Missing status"), 400

        result = ri.update_issue_status(issue_id, new_status)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"❌ Failed to update issue {issue_id}: {e}")
        return jsonify(success=False, message=str(e)), 500


# -------------------- Fetch Notifications --------------------
@app.route("/admin/fetch_notifications")
@role_required("admin")
def fetch_notifications_admin():
    """Fetch latest admin notifications."""
    try:
        limit = int(request.args.get("limit", 50))
        notes = AdminControlPanel.get_notifications(limit=limit)
        return jsonify({"success": True, "data": notes})
    except Exception as e:
        app.logger.error("❌ Failed to fetch notifications: %s", e)
        return jsonify({"success": False, "message": str(e)}), 500


# -------------------- Filter Issues --------------------
@app.route("/admin/issues/filter/<string:group>")
@role_required("admin")
def filter_issues(group):
    """Filter reported issues by status group (open, closed, etc.)."""
    try:
        issues = AdminControlPanel.get_reported_issues(status_group=group, limit=100)
        return jsonify({"success": True, "data": issues})
    except Exception as e:
        app.logger.error("❌ Failed to fetch issues for %s: %s", group, e)
        return jsonify(success=False, message=str(e)), 500


# -------------------- Search Notifications --------------------
@app.route("/admin/search/notifications")
@role_required("admin")
def search_notifications():
    """Search user notifications by email or user ID."""
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


# -------------------- Search Issues --------------------
@app.route("/admin/search/issues")
@role_required("admin")
def search_issues():
    """Search reported issues by email or user ID."""
    email = request.args.get("email")
    user_id = request.args.get("user_id", type=int)

    if email and not user_id:
        user_id = AdminControlPanel.get_user_id_by_email(email)
        if not user_id:
            return jsonify({"success": False, "message": "No user found"}), 404

    if not user_id:
        return jsonify({"success": False, "message": "Provide email or user_id"}), 400

    issues = AdminControlPanel.get_reported_issues_by_user(user_id)
    return jsonify({"success": True, "data": issues})


# -------------------- Submissions --------------------
@app.route("/admin/submissions")
def admin_submissions():
    """Fetch submission analytics for admin view."""
    query = request.args.get("query")
    try:
        data = SubmissionAnalytics.list(query=query)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# -------------------- Assignments --------------------
@app.route("/admin/assignments")
def admin_assignments():
    """Fetch all assignments for admin management."""
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


# -------------------- Instructors --------------------
@app.route("/admin/instructors")
def admin_instructors():
    """Fetch instructor analytics or filter by search criteria."""
    search = request.args.get("search")
    score = request.args.get("score", type=float)
    try:
        data = UserAnalytics.list(role="instructor", search=search, score=score)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# -------------------- Students --------------------
@app.route("/admin/students")
def admin_students():
    """Fetch student analytics or filter by search/sort options."""
    search = request.args.get("search")
    sort = request.args.get("sort")
    try:
        data = UserAnalytics.list(role="student", search=search, sort=sort)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# -------------------- Check Email --------------------
@app.route("/check_email")
def check_email():
    """AJAX endpoint to check if an email is already registered."""
    from backend.user_management import is_email_registered

    email = request.args.get("email", "").strip().lower()
    if not email:
        return jsonify({"exists": False, "message": "No email provided"}), 400

    try:
        if is_email_registered(email):
            return jsonify({"exists": True, "message": "Email already registered"})
        else:
            return jsonify({"exists": False, "message": "Email available"})
    except Exception as e:
        return jsonify({"exists": False, "message": f"Error checking email: {e}"})


# -------------------- Create User --------------------
@app.route("/admin/create_user", methods=["POST"])
def admin_create_user():
    """Allow admin to create a new user via backend route."""
    user_session = session.get("user")
    if not user_session or user_session.get("role") != "admin":
        abort(403)

    from backend.user_management import register_user

    return register_user()


# -------------------- Delete User --------------------
@app.route("/admin/delete_user", methods=["POST"])
@role_required("admin")
def admin_delete_user():
    """Delete a user and related data (admin only)."""
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get("user_id")
        if not user_id:
            return jsonify({"success": False, "message": "Missing user_id"}), 400

        # Get acting admin ID from session
        acting_admin = session.get("user", {}).get("user_id", 1)

        success, message = AdminControlPanel.delete_user_and_data(
            int(user_id), acting_admin_id=int(acting_admin)
        )

        if success:
            return jsonify({"success": True, "message": message})
        else:
            return jsonify({"success": False, "message": message}), 400
    except Exception as e:
        app.logger.exception("Admin delete user failed")
        return jsonify({"success": False, "message": str(e)}), 500


# ============================================================
# 🌍 Global Template Context
# ============================================================
@app.context_processor
def inject_config():
    """Inject global configuration object into all Jinja templates."""
    return dict(config=Config)


# ============================================================
# 🚀 Application Entry Point
# ============================================================
if __name__ == "__main__":
    """Run Flask application in debug mode (for development only)."""
    app.run(debug=True)
