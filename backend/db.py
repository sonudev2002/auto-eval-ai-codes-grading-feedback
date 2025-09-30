cat > db.py << "EOF"
import os
import mysql.connector


def get_connection():
    return mysql.connector.connect(
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "mca_project"),
        port=int(os.getenv("DB_PORT", 3306)),
        connection_timeout=10,
    )


EOF
