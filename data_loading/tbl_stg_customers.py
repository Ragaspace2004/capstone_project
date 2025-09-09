# ============================================
# ETL Script: CSV → MySQL with Audit Logging
# ============================================
# This script:
# 1. Reads data from a CSV into a Pandas DataFrame
# 2. Loads it into a MySQL table (replace or append mode)
# 3. Logs the load details into an audit table (etl_audit)
#    - Who triggered the process
#    - Which table was populated
#    - How many rows were inserted
#    - When it started/finished
#    - Status (SUCCESS/FAIL) and error details if failed
#
# Requirements:
#   pip install pandas sqlalchemy pymysql python-dotenv
# ============================================

import os, time, socket, getpass, traceback
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.mysql import TEXT

# ====== CONFIGURATION ======
HOST = "localhost"                         # MySQL host (your local machine)
PORT = 3306                                # MySQL port (default 3306)
USER = "root"                              # MySQL user
PASSWORD = "root"                          # MySQL password
DATABASE = "adventureworks"                # Target database
CSV_PATH = r"AdventureWorks_Customers.csv" # Path to CSV file
TARGET_TABLE = "customers"                 # Table where data will be loaded
LOAD_MODE = "replace"                      # "replace" (drop & recreate) OR "append" (insert new rows)
PROCESS_NAME = "python_csv_loader"         # Identifier for this ETL process
# ===========================

AUDIT_TABLE = "etl_audit"  # Name of the audit table to store load metadata


# ---------------------------------------------------
# Function: ensure_audit_table
# ---------------------------------------------------
# Creates the audit table if it doesn’t already exist.
# This ensures every load attempt (success/failure) is logged.
def ensure_audit_table(conn):
    ddl = f"""
    CREATE TABLE IF NOT EXISTS `{AUDIT_TABLE}` (
        audit_id           BIGINT AUTO_INCREMENT PRIMARY KEY,   -- unique ID
        process_name       VARCHAR(128) NOT NULL,               -- ETL process name
        database_name      VARCHAR(128) NOT NULL,               -- target DB
        table_name         VARCHAR(128) NOT NULL,               -- target table
        load_mode          ENUM('replace','append') NOT NULL,   -- replace/append
        csv_path           TEXT,                                -- source file path
        records_attempted  BIGINT,                              -- rows read from CSV
        records_inserted   BIGINT,                              -- rows inserted into DB
        started_at         DATETIME NOT NULL,                   -- start timestamp
        finished_at        DATETIME NOT NULL,                   -- end timestamp
        duration_seconds   DECIMAL(12,3) NOT NULL,              -- runtime duration
        status             ENUM('SUCCESS','FAIL') NOT NULL,     -- job status
        error_message      TEXT,                                -- error details (if failed)
        triggered_by       VARCHAR(128),                        -- OS user who ran it
        host_name          VARCHAR(128),                        -- machine hostname
        mysql_user         VARCHAR(128)                         -- MySQL account used
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    conn.execute(text(ddl))


# ---------------------------------------------------
# Function: main
# ---------------------------------------------------
# Core ETL workflow
# ---------------------------------------------------
def main():
    # Step 1: Read CSV file into DataFrame
    # Using dtype=str ensures all data is treated as text (safe for mixed types)
    df = pd.read_csv(CSV_PATH, encoding="latin-1", dtype=str, na_filter=False)
    attempted = len(df)  # number of rows read from CSV

    # Step 2: Build SQLAlchemy engine for MySQL
    url = f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}?charset=utf8mb4"
    engine = create_engine(url, pool_pre_ping=True)

    # Step 3: Collect audit context (user/machine info)
    triggered_by = getpass.getuser()    # OS user (who ran the script)
    host_name = socket.gethostname()    # machine hostname
    mysql_user = None                   # will fetch from DB later
    started_ts = time.time()            # start time (epoch)

    try:
        # ------------------------------
        # START TRANSACTION
        # ------------------------------
        with engine.begin() as conn:
            # Get MySQL user running this connection
            mysql_user = conn.execute(text("SELECT CURRENT_USER()")).scalar()

            # Ensure audit table exists
            ensure_audit_table(conn)

            # Step 4: Load DataFrame into target MySQL table
            # Map every column as TEXT for simplicity
            dtype_map = {col: TEXT() for col in df.columns}

            df.to_sql(
                TARGET_TABLE,
                conn,                 # use the same transaction
                if_exists=LOAD_MODE,  # replace or append
                index=False,
                dtype=dtype_map,
                method="multi",       # batch inserts
                chunksize=1000
            )

            inserted = attempted  # rows inserted = rows attempted (safe assumption)

            # Step 5: Insert SUCCESS audit log
            finished_ts = time.time()
            conn.execute(
                text(f"""
                    INSERT INTO `{AUDIT_TABLE}` 
                    (process_name, database_name, table_name, load_mode, csv_path, 
                     records_attempted, records_inserted, started_at, finished_at, 
                     duration_seconds, status, error_message, triggered_by, host_name, mysql_user)
                    VALUES 
                    (:process_name, :database_name, :table_name, :load_mode, :csv_path,
                     :records_attempted, :records_inserted, FROM_UNIXTIME(:started), FROM_UNIXTIME(:finished),
                     :duration, 'SUCCESS', NULL, :triggered_by, :host_name, :mysql_user)
                """),
                dict(
                    process_name=PROCESS_NAME,
                    database_name=DATABASE,
                    table_name=TARGET_TABLE,
                    load_mode=LOAD_MODE,
                    csv_path=os.path.abspath(CSV_PATH),
                    records_attempted=attempted,
                    records_inserted=inserted,
                    started=started_ts,
                    finished=finished_ts,
                    duration=round(finished_ts - started_ts, 3),
                    triggered_by=triggered_by,
                    host_name=host_name,
                    mysql_user=mysql_user,
                ),
            )

        print(f"✅ Loaded {attempted} rows into `{DATABASE}.{TARGET_TABLE}`. Audit logged in `{AUDIT_TABLE}`.")

    except Exception as e:
        # ------------------------------
        # ERROR HANDLING
        # ------------------------------
        finished_ts = time.time()
        err = traceback.format_exc(limit=5)  # capture error stack trace

        # Best-effort attempt to log FAIL entry in audit table
        with engine.begin() as conn:
            ensure_audit_table(conn)
            if mysql_user is None:
                try:
                    mysql_user = conn.execute(text("SELECT CURRENT_USER()")).scalar()
                except Exception:
                    mysql_user = None

            conn.execute(
                text(f"""
                    INSERT INTO `{AUDIT_TABLE}` 
                    (process_name, database_name, table_name, load_mode, csv_path, 
                     records_attempted, records_inserted, started_at, finished_at, 
                     duration_seconds, status, error_message, triggered_by, host_name, mysql_user)
                    VALUES 
                    (:process_name, :database_name, :table_name, :load_mode, :csv_path,
                     :records_attempted, 0, FROM_UNIXTIME(:started), FROM_UNIXTIME(:finished),
                     :duration, 'FAIL', :error_message, :triggered_by, :host_name, :mysql_user)
                """),
                dict(
                    process_name=PROCESS_NAME,
                    database_name=DATABASE,
                    table_name=TARGET_TABLE,
                    load_mode=LOAD_MODE,
                    csv_path=os.path.abspath(CSV_PATH),
                    records_attempted=attempted,
                    started=started_ts,
                    finished=finished_ts,
                    duration=round(finished_ts - started_ts, 3),
                    error_message=err[:5000],  # store first 5000 chars of error
                    triggered_by=triggered_by,
                    host_name=host_name,
                    mysql_user=mysql_user,
                ),
            )

        print("❌ Load failed. Error has been logged to audit table.")
        raise  # re-raise error so caller sees failure


# ---------------------------------------------------
# Entry point
# ---------------------------------------------------
if __name__ == "__main__":
    main()
