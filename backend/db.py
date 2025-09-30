import mysql.connector

def get_connection():
    return mysql.connector.connect(
       user='root',
    password='Sonudev2002@',
    host='localhost',
    database='mca_project',
    port='3306'
    )
