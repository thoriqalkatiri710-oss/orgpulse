-- ── 3.2.1 Headcount & Attrition Trend ────────────────────────────────────────
WITH monthly_headcount AS (
    SELECT
        DATE_TRUNC('month', generate_series) AS month,
        COUNT(e.employee_id) AS headcount
    FROM generate_series('2024-01-01'::date, '2024-12-31'::date, '1 month') gs
    JOIN employees e ON e.hire_date <= gs
    GROUP BY month
)
SELECT
    month,
    headcount,
    headcount - LAG(headcount) OVER (ORDER BY month) AS net_change,
    ROUND(100.0 * (headcount - LAG(headcount) OVER (ORDER BY month))
        / NULLIF(LAG(headcount) OVER (ORDER BY month), 0), 2) AS growth_pct
FROM monthly_headcount
ORDER BY month;

-- ── 3.2.2 Cross-Departmental Interaction Matrix ───────────────────────────────
SELECT
    e_sender.department   AS sender_dept,
    e_receiver.department AS receiver_dept,
    COUNT(*)              AS total_interactions,
    ROUND(AVG(i.sentiment_raw), 3) AS avg_sentiment,
    COUNT(DISTINCT i.sender_id)    AS unique_senders,
    COUNT(DISTINCT i.receiver_id)  AS unique_receivers
FROM interactions i
JOIN employees e_sender   ON i.sender_id   = e_sender.employee_id
JOIN employees e_receiver ON i.receiver_id = e_receiver.employee_id
WHERE e_sender.department != e_receiver.department
GROUP BY e_sender.department, e_receiver.department
ORDER BY total_interactions DESC;

-- ── 3.2.3 Absensi Trend 30 Hari Sliding Window ───────────────────────────────
SELECT
    employee_id,
    date,
    status,
    SUM(CASE WHEN status != 'present' THEN 1 ELSE 0 END) OVER (
        PARTITION BY employee_id
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS absent_days_30d,
    AVG(hours_worked) OVER (
        PARTITION BY employee_id
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS avg_hours_30d
FROM attendance
ORDER BY employee_id, date;

-- ── 3.2.4 Hierarchical Org Tree (Recursive CTE) ──────────────────────────────
WITH RECURSIVE org_tree AS (
    SELECT employee_id, full_name, department, job_level, manager_id, 0 AS depth
    FROM employees
    WHERE manager_id IS NULL
    UNION ALL
    SELECT e.employee_id, e.full_name, e.department, e.job_level, e.manager_id, ot.depth + 1
    FROM employees e
    JOIN org_tree ot ON e.manager_id = ot.employee_id
    WHERE ot.depth < 10
)
SELECT
    employee_id,
    REPEAT(' ', depth) || full_name AS indented_name,
    department,
    job_level_name,
    depth AS org_depth
FROM org_tree
JOIN employees USING (employee_id)
ORDER BY department, depth, full_name;

-- ── 3.2.5 Training Completion vs Performance Correlation ─────────────────────
SELECT
    e.department,
    ROUND(AVG(tc.completion_rate), 3)   AS avg_completion_rate,
    ROUND(AVG(tc.assessment_score), 1)  AS avg_assessment_score,
    ROUND(AVG(pr.rating), 2)            AS avg_performance_rating,
    CORR(tc.completion_rate, pr.rating::numeric) AS completion_rating_corr
FROM employees e
JOIN training_completion tc ON e.employee_id = tc.employee_id
JOIN performance_reviews pr ON e.employee_id = pr.employee_id
    AND pr.review_cycle = '2024-H2'
GROUP BY e.department
ORDER BY completion_rating_corr DESC NULLS LAST;