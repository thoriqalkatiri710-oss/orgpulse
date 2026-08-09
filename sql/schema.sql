-- OrgPulse Database Schema
-- ========================

CREATE TABLE employees (
    employee_id     VARCHAR(10) PRIMARY KEY,
    full_name       VARCHAR(100),
    department      VARCHAR(50),
    job_level       INT,
    job_level_name  VARCHAR(30),
    hire_date       DATE,
    tenure_years    NUMERIC(5,2),
    monthly_salary  NUMERIC(12,0),
    gender          CHAR(1),
    age             INT,
    education       VARCHAR(5),
    manager_id      VARCHAR(10) REFERENCES employees(employee_id),
    engagement_score NUMERIC(3,1)
);

CREATE TABLE interactions (
    interaction_id      VARCHAR(15) PRIMARY KEY,
    sender_id           VARCHAR(10) REFERENCES employees(employee_id),
    receiver_id         VARCHAR(10) REFERENCES employees(employee_id),
    channel             VARCHAR(20),
    interaction_date    DATE,
    response_time_hours NUMERIC(6,2),
    sentiment_raw       NUMERIC(4,3)
);

CREATE TABLE performance_reviews (
    review_id   VARCHAR(10) PRIMARY KEY,
    employee_id VARCHAR(10) REFERENCES employees(employee_id),
    review_cycle VARCHAR(10),
    review_date  DATE,
    reviewer_id  VARCHAR(10) REFERENCES employees(employee_id),
    rating       INT,
    review_text  TEXT,
    word_count   INT
);

CREATE TABLE training_completion (
    id              SERIAL PRIMARY KEY,
    employee_id     VARCHAR(10) REFERENCES employees(employee_id),
    training_name   VARCHAR(100),
    category        VARCHAR(30),
    training_date   DATE,
    duration_hours  INT,
    completion_rate NUMERIC(4,3),
    assessment_score NUMERIC(5,1)
);

CREATE TABLE attendance (
    id           SERIAL PRIMARY KEY,
    employee_id  VARCHAR(10) REFERENCES employees(employee_id),
    date         DATE,
    status       VARCHAR(20),
    hours_worked NUMERIC(4,1)
);