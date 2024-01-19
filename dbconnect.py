from config import dbhost, dbport, dbuser, dbdatabase, dbpassword
import psycopg2
from psycopg2 import sql
import os


def create_task(task_list):
    try:
        conn = psycopg2.connect(
            host=dbhost,
            port=dbport,
            user=dbuser,
            password=dbpassword,
            database=dbdatabase
        )
        cursor = conn.cursor()
        # Создание таблицы tasks, если её нет
        create_table_query = 'CREATE TABLE IF NOT EXISTS tasks (id SERIAL PRIMARY KEY, "user" TEXT, targets TEXT[], sources TEXT[], type TEXT, start_time TIMESTAMP, status TEXT, end_time TIMESTAMP);'
        cursor.execute(create_table_query)

        # Вставка данных в таблицу tasks
        insert_data_query = sql.SQL("""
            INSERT INTO tasks ("user", targets, sources, type, start_time, status, end_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """)
        cursor.execute(insert_data_query,
                       (task_list[0], task_list[1], task_list[2], task_list[3], task_list[4], task_list[5], task_list[6]))

        # Подтверждение изменений и закрытие соединения
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(e)


def change_status_parsing_complete(task_list):
    try:
        conn = psycopg2.connect(
            host=dbhost,
            port=dbport,
            user=dbuser,
            password=dbpassword,
            database=dbdatabase
        )
        cursor = conn.cursor()
        # Запрос для обновления статуса
        update_status_query = sql.SQL(f"UPDATE tasks SET status = 'parsing complete' WHERE targets @> ARRAY{task_list[1]}::TEXT[] AND sources @> ARRAY{task_list[2]}::TEXT[] AND status = 'in progress..';")
        # Выполнение запроса
        cursor.execute(update_status_query)

        # Подтверждение изменений и закрытие соединения
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(e)


def get_task_id(task_list):
    conn = psycopg2.connect(
        host=dbhost,
        port=dbport,
        user=dbuser,
        password=dbpassword,
        database=dbdatabase
    )
    cursor = conn.cursor()
    # Запрос для получения номера строки (id) по всем параметрам
    select_id_query = sql.SQL("""
        SELECT "id"
        FROM tasks
        WHERE "user" = %s
          AND targets = %s
          AND sources = %s
          AND type = %s
          AND start_time IS %s
          AND status = %s
          AND end_time IS %s;
    """)
    # Выполнение запроса
    cursor.execute(select_id_query, (task_list[0], task_list[1], task_list[2], task_list[3], task_list[4], task_list[5], task_list[6]))
    row_id = cursor.fetchone()
    # Закрытие соединения
    cursor.close()
    conn.close()
    return row_id

def parsing_to_db():
    try:
        conn = psycopg2.connect(
            host=dbhost,
            port=dbport,
            user=dbuser,
            password=dbpassword,
            database=dbdatabase
        )
        cursor = conn.cursor()

        # Проверка наличия таблицы parse_res и создание, если её нет
        create_table_query = """
            CREATE TABLE IF NOT EXISTS parsres (
                id SERIAL PRIMARY KEY,
                task_id INTEGER,
                sources TEXT,
                links TEXT,
                post_date TIMESTAMP,
                post_text TEXT
            );
        """
        cursor.execute(create_table_query)

        csv_file_path = 'chanel_posts.csv'
        copy_query = sql.SQL("""
                COPY parsres(task_id, sources, links, post_date, post_text)
                FROM %s CSV HEADER;
            """)
        cursor.execute(copy_query, (csv_file_path,))
        # Подтверждение изменений и закрытие соединения
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(e)