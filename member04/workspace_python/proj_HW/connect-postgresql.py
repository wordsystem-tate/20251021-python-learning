'''
Created on 2025/10/06

@author: t.tateishi
'''
import psycopg2

# データベースとのコネクションを確立します。
connection = psycopg2.connect("host=localhost dbname=postgres user=<your db username> password=<your db password>")

# カーソルをオープンします
cursor = connection.cursor()

cursor.execute("SELECT * FROM your_table_name")
query_result = cursor.fetchall()
print(query_result)