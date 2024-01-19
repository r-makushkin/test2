import streamlit as st
# from data_processing import get_data
import asyncio
from parsing import parsing
from dbconnect import *



def main():
    st.title("Streamlit App")

    # Ввод переменных от пользователя
    variable1 = st.text_input("Введите переменную 1", "")
    variable2 = st.text_input("Введите переменную 2", "")

    # Кнопка для запуска внешнего скрипта
    if st.button("Запустить скрипт"):
        task_list = ('testuser', variable1.split(), variable2.split(), 'tasktype', None, 'in progress..', None,)
        create_task(task_list)
        try:
            ids = get_task_id(task_list)[:-1]
            asyncio.run(parsing(variable2, ids))
            parsing_to_db()
            change_status_parsing_complete(task_list)
        except Exception as e:
            print(e)




    if st.button('Data processing'):
        # get_data('posts.csv')
        pass


if __name__ == "__main__":
    main()
