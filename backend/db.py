import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load .env if running locally
load_dotenv()


def get_connection():
    """
    Returns a MySQL connection using environment variables.
    Compatible with Railway and Render.
    """
    try:
        connection = mysql.connector.connect(
            host=os.getenv("MYSQLHOST", "localhost"),
            user=os.getenv("MYSQLUSER", "root"),
            password=os.getenv("MYSQLPASSWORD", ""),
            database=os.getenv("MYSQLDATABASE", "mca_project"),
            port=int(os.getenv("MYSQLPORT", 3306)),
        )

        if connection.is_connected():
            return connection
    except Error as e:
        print(f"[DB ERROR] MySQL connection failed: {e}")
        raise
