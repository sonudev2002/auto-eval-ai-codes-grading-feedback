USE railway;
SET FOREIGN_KEY_CHECKS = 0;

-- ==========================================================
-- 1. MASTER TABLES (NO DEPENDENCIES)
-- ==========================================================

-- Address details of all users
CREATE TABLE address (
    address_id INT PRIMARY KEY AUTO_INCREMENT,
    country_name VARCHAR(50),
    state_name VARCHAR(50),
    district_name VARCHAR(50),
    local_address TEXT,
    pincode VARCHAR(10),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- User profile for students, instructors, and admins
CREATE TABLE user_profile (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    middle_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    role ENUM('student', 'instructor', 'admin') NOT NULL,
    profile_picture_path VARCHAR(255),
    mobile_number VARCHAR(15) UNIQUE,
    address_id INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (address_id) REFERENCES address(address_id)
);

-- Difficulty levels for assignments
CREATE TABLE difficulty_level (
    level_id INT PRIMARY KEY AUTO_INCREMENT,
    difficulty_types VARCHAR(50) UNIQUE NOT NULL,
    marks INT NOT NULL
);

-- Repository for instructors to store their assignment sets
CREATE TABLE assignment_repository (
    repository_id INT PRIMARY KEY AUTO_INCREMENT,
    repo_title VARCHAR(255) NOT NULL,
    created_by INT,
    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES user_profile(user_id)
);

-- Grade distribution pattern for analytics
CREATE TABLE grade_distribution (
    distribution_id INT PRIMARY KEY AUTO_INCREMENT,
    related_id INT,
    grade_a INT DEFAULT 0,
    grade_b INT DEFAULT 0,
    grade_c INT DEFAULT 0,
    grade_d INT DEFAULT 0,
    grade_e INT DEFAULT 0,
    grade_f INT DEFAULT 0,
    FOREIGN KEY (related_id) REFERENCES user_profile(user_id)
);

-- ==========================================================
-- 2. ASSIGNMENT AND TEST-RELATED TABLES
-- ==========================================================

-- Assignment master table
CREATE TABLE assignment (
    assignment_id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    hint TEXT,
    instructor_id INT,
    difficulty_level INT,
    due_date DATETIME NOT NULL,
    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    repository_id INT,
    FOREIGN KEY (instructor_id) REFERENCES user_profile(user_id),
    FOREIGN KEY (difficulty_level) REFERENCES difficulty_level(level_id),
    FOREIGN KEY (repository_id) REFERENCES assignment_repository(repository_id)
);

-- Examples shown with each assignment
CREATE TABLE example (
    example_id INT PRIMARY KEY AUTO_INCREMENT,
    assignment_id INT,
    example_input TEXT,
    example_output TEXT,
    description TEXT,
    FOREIGN KEY (assignment_id) REFERENCES assignment(assignment_id)
);

-- Test cases linked with each assignment
CREATE TABLE test_cases (
    testcase_id INT PRIMARY KEY AUTO_INCREMENT,
    assignment_id INT,
    input_data TEXT NOT NULL,
    expected_data TEXT NOT NULL,
    FOREIGN KEY (assignment_id) REFERENCES assignment(assignment_id)
);

-- ==========================================================
-- 3. SUBMISSION AND EVALUATION TABLES
-- ==========================================================

-- Student submissions of code
CREATE TABLE code_submission (
    submission_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    assignment_id INT,
    language VARCHAR(50) NOT NULL,
    code_path VARCHAR(255) NOT NULL,
    submitted_on DATETIME DEFAULT CURRENT_TIMESTAMP,
    version INT DEFAULT 1,
    FOREIGN KEY (user_id) REFERENCES user_profile(user_id),
    FOREIGN KEY (assignment_id) REFERENCES assignment(assignment_id)
);

-- Evaluation results of submitted code
CREATE TABLE code_evaluation (
    code_evaluation_id INT PRIMARY KEY AUTO_INCREMENT,
    submission_id INT,
    feedback TEXT,
    grade CHAR(1) CHECK (grade IN ('A','B','C','D','E','F')),
    score FLOAT CHECK (score BETWEEN 0 AND 100),
    plagiarism_score FLOAT CHECK (plagiarism_score BETWEEN 0 AND 100),
    has_syntax_error BOOLEAN DEFAULT FALSE,
    code_quality_score FLOAT CHECK (code_quality_score BETWEEN 0 AND 100),
    code_length INT NOT NULL,
    cyclomatic_complexity INT NOT NULL,
    total_testcases INT DEFAULT 0,
    passed_testcases INT DEFAULT 0,
    failed_testcases INT DEFAULT 0,
    average_execution_time FLOAT DEFAULT 0.0,
    memory_usage FLOAT,
    FOREIGN KEY (submission_id) REFERENCES code_submission(submission_id)
);

-- Mapping of plagiarism between submissions
CREATE TABLE plagiarism_match (
    id INT PRIMARY KEY AUTO_INCREMENT,
    evaluation_id INT,
    matched_submission_id INT,
    FOREIGN KEY (evaluation_id) REFERENCES code_evaluation(code_evaluation_id),
    FOREIGN KEY (matched_submission_id) REFERENCES code_submission(submission_id)
);

-- Result of each test case for a submission
CREATE TABLE test_case_result (
    testcase_result_id INT PRIMARY KEY AUTO_INCREMENT,
    submission_id INT,
    testcase_id INT,
    output TEXT NOT NULL,
    passed BOOLEAN NOT NULL,
    execution_time FLOAT NOT NULL,
    FOREIGN KEY (submission_id) REFERENCES code_submission(submission_id),
    FOREIGN KEY (testcase_id) REFERENCES test_cases(testcase_id)
);

-- Feedback scoring (student or system-based)
CREATE TABLE feedback_score (
    feedback_id INT PRIMARY KEY AUTO_INCREMENT,
    submission_id INT,
    feedback_score INT CHECK (feedback_score BETWEEN 0 AND 5),
    FOREIGN KEY (submission_id) REFERENCES code_submission(submission_id)
);

-- ==========================================================
-- 4. ANALYTICS AND PERFORMANCE TABLES
-- ==========================================================

-- Instructor analytics by difficulty level
CREATE TABLE instructor_difficulty_stats (
    instructor_stats_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    difficulty_level INT,
    assignment_count INT DEFAULT 0,
    average_score FLOAT CHECK (average_score BETWEEN 0 AND 100),
    average_pass_rate FLOAT CHECK (average_pass_rate BETWEEN 0 AND 100),
    average_feedback_score INT CHECK (average_feedback_score BETWEEN 0 AND 5),
    FOREIGN KEY (user_id) REFERENCES user_profile(user_id),
    FOREIGN KEY (difficulty_level) REFERENCES difficulty_level(level_id)
);

-- Overall instructor performance analytics
CREATE TABLE instructor_performance_analytics (
    analytics_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    total_assignments_created INT DEFAULT 0,
    total_submissions_received INT DEFAULT 0,
    overall_avg_score FLOAT CHECK (overall_avg_score BETWEEN 0 AND 100),
    avg_pass_rate FLOAT CHECK (avg_pass_rate BETWEEN 0 AND 100),
    plagiarism_rate FLOAT CHECK (plagiarism_rate BETWEEN 0 AND 100),
    feedback_score_avg FLOAT,
    responsiveness_score FLOAT,
    consistency_score FLOAT,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    instructor_ds_id INT,
    FOREIGN KEY (user_id) REFERENCES user_profile(user_id),
    FOREIGN KEY (instructor_ds_id) REFERENCES instructor_difficulty_stats(instructor_stats_id)
);

-- Student analytics by difficulty level
CREATE TABLE student_difficulty_stats (
    student_stats_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    difficulty_level INT,
    assignment_count INT DEFAULT 0,
    average_score FLOAT CHECK (average_score BETWEEN 0 AND 100),
    average_pass_rate FLOAT CHECK (average_pass_rate BETWEEN 0 AND 100),
    FOREIGN KEY (user_id) REFERENCES user_profile(user_id),
    FOREIGN KEY (difficulty_level) REFERENCES difficulty_level(level_id)
);

-- Overall student performance
CREATE TABLE student_performance_analytics (
    analytics_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    average_score FLOAT CHECK (average_score BETWEEN 0 AND 100),
    completion_rate FLOAT CHECK (completion_rate BETWEEN 0 AND 100),
    pass_rate FLOAT CHECK (pass_rate BETWEEN 0 AND 100),
    plagiarism_incidents INT DEFAULT 0,
    performance_band VARCHAR(50),
    total_assignments INT DEFAULT 0,
    performance_level VARCHAR(5),
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    distribution_id INT,
    FOREIGN KEY (user_id) REFERENCES user_profile(user_id),
    FOREIGN KEY (distribution_id) REFERENCES grade_distribution(distribution_id)
);

-- Analytics per assignment
CREATE TABLE assignment_analytics (
    analytics_id INT PRIMARY KEY AUTO_INCREMENT,
    assignment_id INT,
    total_submission INT NOT NULL,
    average_score FLOAT CHECK (average_score BETWEEN 0 AND 100),
    plagiarism_cases INT DEFAULT 0,
    pass_percentage FLOAT CHECK (pass_percentage BETWEEN 0 AND 100),
    average_time FLOAT,
    most_common_error TEXT,
    FOREIGN KEY (assignment_id) REFERENCES assignment(assignment_id)
);

-- ==========================================================
-- 5. NOTIFICATION SYSTEM
-- ==========================================================

CREATE TABLE broadcast_notification (
    broadcast_id INT PRIMARY KEY AUTO_INCREMENT,
    broadcast_type VARCHAR(50) NOT NULL,
    broadcast_mode VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE notification (
    notification_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    message TEXT NOT NULL,
    notification_mode VARCHAR(50) NOT NULL,
    type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'unread',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user_profile(user_id)
);

-- ==========================================================
-- 6. SYSTEM LOGS AND ISSUES
-- ==========================================================

CREATE TABLE login_log (
    log_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    ip_address VARCHAR(50),
    login_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    logout_time DATETIME,
    device_info TEXT,
    os VARCHAR(50),
    browser VARCHAR(50),
    type VARCHAR(50),
    FOREIGN KEY (user_id) REFERENCES user_profile(user_id)
);

CREATE TABLE reported_issue (
    issue_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    issue_type VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'open',
    reported_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolved_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES user_profile(user_id)
);

CREATE TABLE screenshots (
    screenshot_id INT PRIMARY KEY AUTO_INCREMENT,
    issue_id INT,
    screenshot_path VARCHAR(255) NOT NULL,
    FOREIGN KEY (issue_id) REFERENCES reported_issue(issue_id)
);

CREATE TABLE system_statistics (
    snapshot_id INT PRIMARY KEY AUTO_INCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_students INT DEFAULT 0,
    total_instructors INT DEFAULT 0,
    total_assignments INT DEFAULT 0,
    active_users_today INT DEFAULT 0,
    average_score FLOAT DEFAULT 0.0,
    total_submissions INT DEFAULT 0,
    new_users_last_week INT DEFAULT 0,
    grade_distribution JSON
);

-- ==========================================================
-- 7. STUDENT ASSIGNMENT STATUS
-- ==========================================================

CREATE TABLE student_assignment_status (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    assignment_id INT NOT NULL,
    status ENUM('pending submission', 'submitted', 'graded', 'late') DEFAULT 'pending submission',
    submitted_at DATETIME,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_user_assignment (user_id, assignment_id),
    FOREIGN KEY (user_id) REFERENCES user_profile(user_id),
    FOREIGN KEY (assignment_id) REFERENCES assignment(assignment_id)
);

SET FOREIGN_KEY_CHECKS = 1;
