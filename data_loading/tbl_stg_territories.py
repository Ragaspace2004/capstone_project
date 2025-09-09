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
CSV_PATH = r"AdventureWorks_Territories.csv" # Path to CSV file
TARGET_TABLE = "territories"               # Table where data will be loaded
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
            df.to_sql(TARGET_TABLE, conn, if_exists=LOAD_MODE, index=False, dtype={col: TEXT for col in df.columns})

            inserted = len(df)  # assuming all rows inserted (in replace mode)

            # Step 5: Log the audit record
            finished_ts = time.time()
            duration = finished_ts - started_ts
            audit_sql = f"""
            INSERT INTO `{AUDIT_TABLE}` (
                process_name, database_name, table_name, load_mode, csv_path,
                records_attempted, records_inserted, started_at, finished_at, duration_seconds,
                status, triggered_by, host_name, mysql_user
            ) VALUES (
                '{PROCESS_NAME}', '{DATABASE}', '{TARGET_TABLE}', '{LOAD_MODE}', '{CSV_PATH}',
                {attempted}, {inserted}, FROM_UNIXTIME({started_ts}), FROM_UNIXTIME({finished_ts}), {duration:.3f},
                'SUCCESS', '{triggered_by}', '{host_name}', '{mysql_user}'
            )
            """
            conn.execute(text(audit_sql))

        print(f"✅ Loaded {inserted} rows into `{DATABASE}.{TARGET_TABLE}`. Audit logged in `{AUDIT_TABLE}`.")

    except Exception as e:
        # Log failure
        finished_ts = time.time()
        duration = finished_ts - started_ts
        error_msg = str(e).replace("'", "''")  # escape single quotes
        try:
            with engine.begin() as conn:
                mysql_user = conn.execute(text("SELECT CURRENT_USER()")).scalar() or "unknown"
                ensure_audit_table(conn)
                audit_sql = f"""
                INSERT INTO `{AUDIT_TABLE}` (
                    process_name, database_name, table_name, load_mode, csv_path,
                    records_attempted, records_inserted, started_at, finished_at, duration_seconds,
                    status, error_message, triggered_by, host_name, mysql_user
                ) VALUES (
                    '{PROCESS_NAME}', '{DATABASE}', '{TARGET_TABLE}', '{LOAD_MODE}', '{CSV_PATH}',
                    {attempted}, 0, FROM_UNIXTIME({started_ts}), FROM_UNIXTIME({finished_ts}), {duration:.3f},
                    'FAIL', '{error_msg}', '{triggered_by}', '{host_name}', '{mysql_user}'
                )
                """
                conn.execute(text(audit_sql))
        except:
            pass  # if even audit fails, just print
        print(f"❌ Failed to load `{DATABASE}.{TARGET_TABLE}`: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
