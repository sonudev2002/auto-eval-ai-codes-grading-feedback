import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load local .env
load_dotenv()


def get_connection():
    """
    Returns a secure MySQL connection (Railway + Render compatible)
    """
    try:
        connection = mysql.connector.connect(
            host=os.getenv("MYSQLHOST", "localhost"),
            user=os.getenv("MYSQLUSER", "root"),
            password=os.getenv("MYSQLPASSWORD", ""),
            database=os.getenv("MYSQLDATABASE", "mca_project"),
            port=int(os.getenv("MYSQLPORT", 3306)),
            ssl_disabled=False,  # ✅ Enable SSL
            ssl_verify_identity=False,  # ✅ Avoid CA cert check
            connection_timeout=10,
        )

        if connection.is_connected():
            print("[DB SUCCESS] Connected to MySQL")
            return connection
    except Error as e:
        print(f"[DB ERROR] MySQL connection failed: {e}")
        raise
