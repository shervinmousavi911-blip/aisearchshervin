import mysql.connector
import sqlite3

class Database:
    def __init__(self, db_type="mysql", **kwargs):
        self.db_type = db_type
        self.conn = None
        self.kwargs = kwargs
        self.connect()

    def connect(self):
        if self.db_type == "mysql":
            self.conn = mysql.connector.connect(
                host=self.kwargs.get("host", "localhost"),
                user=self.kwargs.get("user", "root"),
                password=self.kwargs.get("password", ""),
                database=self.kwargs.get("database", "chatbotdb"),
                charset="utf8mb4"
            )
        elif self.db_type == "sqlite":
            self.conn = sqlite3.connect(self.kwargs.get("database", "chatbotdb"))
            self.conn.row_factory = sqlite3.Row
        else:
            raise ValueError("Unsupported database type")

    def fetch_all(self, table_name):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(dict(zip(columns, row)))
        return results, columns
