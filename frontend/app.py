import streamlit as st
import requests
import re

BACKEND_URL = "http://backend:8000/summarize"


def main():
    st.title("📝 Умный генератор саммари текста")
    st.markdown("""
        Приложение использует модель BART-large-CNN для генерации краткого содержания текста.
        Поддерживает обработку длинных документов с сохранением контекста.
    """)

    input_method = st.radio("Выберите способ ввода:", ("Текстовое поле", "Файл"))
    text = ""

    if input_method == "Текстовое поле":
        text = st.text_area("Введите текст для суммаризации:", height=300,
                            placeholder="Вставьте текст длиной до 1000 слов...")
    else:
        uploaded_file = st.file_uploader("Загрузите текстовый файл", type=['txt'])
        if uploaded_file:
            if uploaded_file.type == "text/plain":
                text = str(uploaded_file.read(), "utf-8")
            else:
                st.warning("Поддерживаются только TXT файлы в данной версии")

    if st.button("Сгенерировать саммари") and text:
        with st.spinner("Анализируем текст..."):
            try:
                words_count = len(text.split())
                chars_count = len(text)
                st.info(f"Текст содержит: {words_count} слов, {chars_count} символов")

                response = requests.post(BACKEND_URL, json={"text": text})
                if response.status_code == 200:
                    result = response.json()
                    if "summary" in result:
                        st.subheader("Результат суммаризации:")
                        st.write(result["summary"])

                        summary_length = len(result["summary"].split())
                        ratio = summary_length / words_count * 100
                        st.success(
                            f"Сжатие: {words_count} → {summary_length} слов "
                            f"({ratio:.1f}% от оригинала)"
                        )
                    else:
                        st.error(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")
                else:
                    st.error(f"Ошибка сервера: {response.status_code}")
            except Exception as e:
                st.error(f"Произошла ошибка при обработке: {str(e)}")


if __name__ == "__main__":
    main()