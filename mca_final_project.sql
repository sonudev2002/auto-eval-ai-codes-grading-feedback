USE railway;
SET FOREIGN_KEY_CHECKS=0;


-- 1. Base tables (no dependencies)
CREATE TABLE Address (
    address_id INT PRIMARY KEY AUTO_INCREMENT,
    country_name VARCHAR(50) NULL,
    state_name VARCHAR(50) NULL,
    district_name VARCHAR(50) NULL,
    local_address TEXT NULL,
    pincode VARCHAR(10) NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE User_Profile (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    middle_name VARCHAR(50) NULL,
    last_name VARCHAR(50) NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    role ENUM('student', 'instructor', 'admin') NOT NULL,
    profile_picture_path VARCHAR(255) NULL,
    mobile_number VARCHAR(15) UNIQUE NULL,
    address_id INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (address_id) REFERENCES Address(address_id)
);

CREATE TABLE Difficulty_Level (
    level_id INT PRIMARY KEY AUTO_INCREMENT,
    difficulty_types VARCHAR(50) UNIQUE NOT NULL,
    marks INT NOT NULL
);

CREATE TABLE Assignment_Repository (
    repository_id INT PRIMARY KEY AUTO_INCREMENT,
    repo_title VARCHAR(255) NOT NULL,
    created_by INT,
    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES User_Profile(user_id)
);

CREATE TABLE Grade_distribution (
    distribution_id INT PRIMARY KEY AUTO_INCREMENT,
    related_id INT,
    grade_a INT DEFAULT 0,
    grade_b INT DEFAULT 0,
    grade_c INT DEFAULT 0,
    grade_d INT DEFAULT 0,
    grade_e INT DEFAULT 0,
    grade_f INT DEFAULT 0,
    FOREIGN KEY (related_id) REFERENCES User_Profile(user_id)
);

-- 2. Assignment-related
CREATE TABLE Assignment (
    assignment_id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    hint TEXT NULL,
    instructor_id INT,
    difficulty_level INT,
    due_date DATETIME NOT NULL,
    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    repository_id INT,
    FOREIGN KEY (instructor_id) REFERENCES User_Profile(user_id),
    FOREIGN KEY (difficulty_level) REFERENCES Difficulty_Level(level_id),
    FOREIGN KEY (repository_id) REFERENCES Assignment_Repository(repository_id)
);

CREATE TABLE example (
  example_id INT PRIMARY KEY AUTO_INCREMENT,
  assignment_id INT,
  example_input TEXT,
  example_output TEXT,
  description TEXT
);

CREATE TABLE Test_Cases (
    testcase_id INT PRIMARY KEY AUTO_INCREMENT,
    assignment_id INT,
    input_data TEXT NOT NULL,
    expected_data TEXT NOT NULL,
    FOREIGN KEY (assignment_id) REFERENCES Assignment(assignment_id)
);

-- 3. Submissions & evaluations
CREATE TABLE Code_Submission (
    submission_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    assignment_id INT,
    language VARCHAR(50) NOT NULL,
    code_path VARCHAR(255) NOT NULL,
    submitted_on DATETIME DEFAULT CURRENT_TIMESTAMP,
    version INT DEFAULT 1,
    FOREIGN KEY (user_id) REFERENCES User_Profile(user_id),
    FOREIGN KEY (assignment_id) REFERENCES Assignment(assignment_id)
);

CREATE TABLE Code_Evaluation (
    code_evaluation_id INT PRIMARY KEY AUTO_INCREMENT,
    submission_id INT,
    feedback TEXT NULL,
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
    memory_usage FLOAT NULL,
    FOREIGN KEY (submission_id) REFERENCES Code_Submission(submission_id)
);

CREATE TABLE Plagiarism_match (
    id INT PRIMARY KEY AUTO_INCREMENT,
    evaluation_id INT,
    matched_submission_id INT,
    FOREIGN KEY (evaluation_id) REFERENCES Code_Evaluation(code_evaluation_id),
    FOREIGN KEY (matched_submission_id) REFERENCES Code_Submission(submission_id)
);

CREATE TABLE Test_Case_Result (
    testcase_result_id INT PRIMARY KEY AUTO_INCREMENT,
    submission_id INT,
    testcase_id INT,
    output TEXT NOT NULL,
    passed BOOLEAN NOT NULL,
    execution_time FLOAT NOT NULL,
    FOREIGN KEY (submission_id) REFERENCES Code_Submission(submission_id),
    FOREIGN KEY (testcase_id) REFERENCES Test_Cases(testcase_id)
);

CREATE TABLE Feedback_Score (
    feedback_id INT PRIMARY KEY AUTO_INCREMENT,
    submission_id INT,
    feedback_score INT CHECK (feedback_score BETWEEN 0 AND 5),
    FOREIGN KEY (submission_id) REFERENCES Code_Submission(submission_id)
);

-- 4. Analytics
CREATE TABLE Instructor_Difficulty_Stats (
    instructor_stats_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    difficulty_level INT,
    assignment_count INT DEFAULT 0,
    average_score FLOAT CHECK (average_score BETWEEN 0 AND 100),
    average_pass_rate FLOAT CHECK (average_pass_rate BETWEEN 0 AND 100),
    average_feedback_score INT CHECK (average_feedback_score BETWEEN 0 AND 5),
    FOREIGN KEY (user_id) REFERENCES User_Profile(user_id),
    FOREIGN KEY (difficulty_level) REFERENCES Difficulty_Level(level_id)
);

CREATE TABLE Instructor_Performance_Analytics (
    analytics_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    total_assignments_created INT DEFAULT 0,
    total_submissions_received INT DEFAULT 0,
    overall_avg_score FLOAT CHECK (overall_avg_score BETWEEN 0 AND 100),
    avg_pass_rate FLOAT CHECK (avg_pass_rate BETWEEN 0 AND 100),
    plagiarism_rate FLOAT CHECK (plagiarism_rate BETWEEN 0 AND 100),
    feedback_score_avg FLOAT NULL,
    responsiveness_score FLOAT NULL,
    consistency_score FLOAT NULL,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    instructor_ds_id INT,
    FOREIGN KEY (user_id) REFERENCES User_Profile(user_id),
    FOREIGN KEY (instructor_ds_id) REFERENCES Instructor_Difficulty_Stats(instructor_stats_id)
);

CREATE TABLE Student_Difficulty_Stats (
    student_stats_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    difficulty_level INT,
    assignment_count INT DEFAULT 0,
    average_score FLOAT CHECK (average_score BETWEEN 0 AND 100),
    average_pass_rate FLOAT CHECK (average_pass_rate BETWEEN 0 AND 100),
    FOREIGN KEY (user_id) REFERENCES User_Profile(user_id),
    FOREIGN KEY (difficulty_level) REFERENCES Difficulty_Level(level_id)
);

CREATE TABLE Student_Performance_Analytics (
    analytics_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    average_score FLOAT CHECK (average_score BETWEEN 0 AND 100),
    completion_rate FLOAT CHECK (completion_rate BETWEEN 0 AND 100),
    pass_rate FLOAT CHECK (pass_rate BETWEEN 0 AND 100),
    plagiarism_incidents INT DEFAULT 0,
    performance_band VARCHAR(50) NULL,
    total_assignments INT DEFAULT 0,
    performance_level VARCHAR(5) NULL,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    distribution_id INT,
    FOREIGN KEY (user_id) REFERENCES User_Profile(user_id),
    FOREIGN KEY (distribution_id) REFERENCES Grade_distribution(distribution_id)
);

CREATE TABLE Assignment_Analytics (
    analytics_id INT PRIMARY KEY AUTO_INCREMENT,
    assignment_id INT,
    total_submission INT NOT NULL,
    average_score FLOAT CHECK (average_score BETWEEN 0 AND 100),
    plagiarism_cases INT DEFAULT 0,
    pass_percentage FLOAT CHECK (pass_percentage BETWEEN 0 AND 100),
    average_time FLOAT NULL,
    most_common_error TEXT NULL,
    FOREIGN KEY (assignment_id) REFERENCES Assignment(assignment_id)
);

-- 5. Notifications
CREATE TABLE BroadCast_Notification (
    broadcast_id INT PRIMARY KEY AUTO_INCREMENT,
    broadcast_type varchar(50) NOT NULL,
    broadcast_mode varchar(50) NOT NULL,
    message TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Notification (
    notification_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    message TEXT NOT NULL,
    notification_mode varchar(50) NOT NULL,
    type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'unread',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES User_Profile(user_id)
);

-- 6. Logs, issues, system stats
CREATE TABLE Login_Log (
    log_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    ip_address VARCHAR(50) NULL,
    login_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    logout_time DATETIME NULL,
    device_info TEXT NULL,
    os VARCHAR(50) NULL,
    browser VARCHAR(50) NULL,
    type VARCHAR(50) NULL,
    FOREIGN KEY (user_id) REFERENCES User_Profile(user_id)
);

CREATE TABLE Reported_Issue (
    issue_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    issue_type VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'open',
    reported_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolved_at DATETIME NULL,
    FOREIGN KEY (user_id) REFERENCES User_Profile(user_id)
);

CREATE TABLE Screenshots (
    screenshot_id INT PRIMARY KEY AUTO_INCREMENT,
    issue_id INT,
    screenshot_path VARCHAR(255) NOT NULL,
    FOREIGN KEY (issue_id) REFERENCES Reported_Issue(issue_id)
);

CREATE TABLE System_Statistics (
    snapshot_id INT PRIMARY KEY AUTO_INCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_students INT DEFAULT 0,
    total_instructors INT DEFAULT 0,
    total_assignments INT DEFAULT 0,
    active_users_today INT DEFAULT 0,
    average_score FLOAT DEFAULT 0.0,
    total_submissions INT DEFAULT 0,
    new_users_last_week INT DEFAULT 0,
    grade_distribution JSON NULL
);
