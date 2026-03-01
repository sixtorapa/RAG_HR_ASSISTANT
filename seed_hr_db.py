#!/usr/bin/env python3
"""
seed_hr_db.py  —  Creates and populates the SQLite HR toy database.

Run once before starting the app:
    python seed_hr_db.py

Creates: hr_data.db (same directory as this script)
"""

import sqlite3
import random
from datetime import date, timedelta
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "hr_data.db")

DEPARTMENTS = [
    (1, "Engineering",      850_000, None),
    (2, "HR",               210_000, None),
    (3, "Sales",            620_000, None),
    (4, "Finance",          300_000, None),
    (5, "Marketing",        380_000, None),
    (6, "Product",          430_000, None),
]

ROLES_BY_DEPT = {
    1: ["Software Engineer", "Data Engineer", "DevOps Engineer", "QA Engineer"],
    2: ["HR Generalist", "Talent Acquisition Specialist", "L&D Manager"],
    3: ["Account Executive", "Sales Development Rep", "Sales Manager"],
    4: ["Financial Analyst", "Accountant", "FP&A Manager"],
    5: ["Marketing Specialist", "Content Manager", "Growth Analyst"],
    6: ["Product Manager", "UX Designer", "Product Analyst"],
}

LEVELS = ["Junior", "Mid", "Senior", "Lead", "Manager"]
LEVEL_SALARY = {
    "Junior":  (28_000, 38_000),
    "Mid":     (38_000, 55_000),
    "Senior":  (55_000, 80_000),
    "Lead":    (75_000, 100_000),
    "Manager": (85_000, 120_000),
}
LOCATIONS = ["Madrid", "Barcelona", "Remote", "London"]
NAMES = [
    "Alice Johnson", "Bob Smith", "Carlos García", "Diana Müller", "Elena Rossi",
    "Frank Chen", "Grace Kim", "Hector López", "Irene Dubois", "James Taylor",
    "Karen Nguyen", "Liam O'Brien", "Marta Kowalski", "Nils Andersen", "Olivia Brown",
    "Pablo Fernández", "Quinn Davis", "Rosa Martínez", "Sven Hansen", "Tara Patel",
    "Umar Khalid", "Vera Novak", "William Scott", "Xiaoming Li", "Yuki Tanaka",
    "Zara Ahmed", "Aaron White", "Beatriz Alves", "Cian Murphy", "Daria Petrov",
    "Emre Yilmaz", "Fatima Benali", "Gabriele Ricci", "Hannah Wolf", "Ivan Sokolov",
    "Julia Becker", "Kevin O'Sullivan", "Laura Blanc", "Marco Ferrari", "Nina Larsson",
    "Oscar Moreno", "Patricia Santos", "Raj Patel", "Sandra Holst", "Thomas Berger",
    "Uma Johansson", "Victor Hugo", "Wendy Clarke", "Xu Ming", "Yasmin Abbas",
]

RATING_LABELS = {
    (1.0, 2.0): "Needs Improvement",
    (2.0, 3.5): "Meets Expectations",
    (3.5, 4.5): "Exceeds Expectations",
    (4.5, 5.1): "Outstanding",
}

def rating_label(score: float) -> str:
    for (lo, hi), label in RATING_LABELS.items():
        if lo <= score < hi:
            return label
    return "Meets Expectations"

def random_date(start_year=2018, end_year=2024) -> str:
    start = date(start_year, 1, 1)
    end   = date(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()


def seed():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"🗑  Removed existing {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # ── Schema ────────────────────────────────────────────────────────────────
    cur.executescript("""
    CREATE TABLE departments (
        id      INTEGER PRIMARY KEY,
        name    TEXT NOT NULL,
        budget  REAL,
        head_id INTEGER
    );

    CREATE TABLE employees (
        id            INTEGER PRIMARY KEY,
        name          TEXT NOT NULL,
        department_id INTEGER REFERENCES departments(id),
        role          TEXT,
        level         TEXT,
        salary        REAL,
        hire_date     TEXT,
        manager_id    INTEGER REFERENCES employees(id),
        location      TEXT,
        status        TEXT DEFAULT 'active'
    );

    CREATE TABLE performance_reviews (
        id           INTEGER PRIMARY KEY,
        employee_id  INTEGER REFERENCES employees(id),
        review_year  INTEGER,
        score        REAL,
        rating_label TEXT,
        reviewer_id  INTEGER REFERENCES employees(id)
    );

    CREATE TABLE job_postings (
        id            INTEGER PRIMARY KEY,
        title         TEXT,
        department_id INTEGER REFERENCES departments(id),
        posted_date   TEXT,
        status        TEXT,
        applicants    INTEGER
    );
    """)

    # ── Departments ───────────────────────────────────────────────────────────
    cur.executemany(
        "INSERT INTO departments (id, name, budget, head_id) VALUES (?,?,?,?)",
        DEPARTMENTS,
    )

    # ── Employees ─────────────────────────────────────────────────────────────
    random.seed(42)
    employees = []
    emp_id = 1
    dept_managers: dict[int, int] = {}   # dept_id → manager emp_id

    # First pass: create one Manager per department
    for dept_id, dept_name, _, _ in DEPARTMENTS:
        level   = "Manager"
        lo, hi  = LEVEL_SALARY[level]
        salary  = round(random.uniform(lo, hi), 2)
        role    = random.choice(ROLES_BY_DEPT[dept_id])
        name    = NAMES[(emp_id - 1) % len(NAMES)]
        location = random.choice(LOCATIONS)
        employees.append((emp_id, name, dept_id, role, level, salary,
                          random_date(2016, 2020), None, location, "active"))
        dept_managers[dept_id] = emp_id
        emp_id += 1

    # Second pass: fill remaining employees
    for i in range(len(NAMES) - len(DEPARTMENTS)):
        dept_id = random.choice([d[0] for d in DEPARTMENTS])
        level   = random.choice(["Junior", "Mid", "Senior", "Lead"])
        lo, hi  = LEVEL_SALARY[level]
        salary  = round(random.uniform(lo, hi), 2)
        role    = random.choice(ROLES_BY_DEPT[dept_id])
        name    = NAMES[emp_id - 1] if emp_id - 1 < len(NAMES) else f"Employee {emp_id}"
        location = random.choice(LOCATIONS)
        status  = "terminated" if random.random() < 0.08 else "active"
        employees.append((emp_id, name, dept_id, role, level, salary,
                          random_date(2018, 2024), dept_managers[dept_id], location, status))
        emp_id += 1

    cur.executemany(
        "INSERT INTO employees VALUES (?,?,?,?,?,?,?,?,?,?)",
        employees,
    )

    # Update department heads
    for dept_id, manager_id in dept_managers.items():
        cur.execute("UPDATE departments SET head_id=? WHERE id=?", (manager_id, dept_id))

    # ── Performance reviews (2022–2024 for active employees) ─────────────────
    active_ids = [e[0] for e in employees if e[9] == "active"]
    review_id  = 1
    for year in [2022, 2023, 2024]:
        for emp_id_rev in active_ids:
            score    = round(random.uniform(1.5, 5.0), 1)
            label    = rating_label(score)
            reviewer = dept_managers.get(
                next(e[2] for e in employees if e[0] == emp_id_rev), active_ids[0]
            )
            cur.execute(
                "INSERT INTO performance_reviews VALUES (?,?,?,?,?,?)",
                (review_id, emp_id_rev, year, score, label, reviewer),
            )
            review_id += 1

    # ── Job postings ──────────────────────────────────────────────────────────
    open_roles = [
        ("Senior Software Engineer", 1, "open"),
        ("HR Business Partner",      2, "open"),
        ("Account Executive EMEA",   3, "open"),
        ("Data Analyst",             4, "filled"),
        ("Growth Marketing Manager", 5, "open"),
        ("Product Manager — Mobile", 6, "cancelled"),
        ("DevOps Engineer",          1, "open"),
        ("Talent Acquisition Lead",  2, "filled"),
    ]
    for i, (title, dept, status) in enumerate(open_roles, start=1):
        cur.execute(
            "INSERT INTO job_postings VALUES (?,?,?,?,?,?)",
            (i, title, dept, random_date(2024, 2025), status, random.randint(5, 120)),
        )

    conn.commit()
    conn.close()

    print(f"✅ HR toy database created: {DB_PATH}")
    print(f"   Employees     : {len(employees)}")
    print(f"   Departments   : {len(DEPARTMENTS)}")
    print(f"   Reviews       : {(2024-2022+1) * len(active_ids)}")
    print(f"   Job postings  : {len(open_roles)}")


if __name__ == "__main__":
    seed()
