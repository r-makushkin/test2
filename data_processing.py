import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, MBartForConditionalGeneration, MBartTokenizer
import torch
import re
from tqdm import tqdm
import torch
from tqdm.auto import tqdm
import tokenizers
from config import minus_words


def get_data(path):
    df = pd.read_csv(path)
    df.rename(columns={'post_text': 'text'}, inplace=True)
    df_minus = pd.DataFrame()
    for word in minus_words:
        df_temp = df[df['text'].str.contains(word, case=False, na=False)]
        df_minus = pd.concat([df_minus, df_temp])
    # Удаление дубликатов строк, если они есть
    df_minus = df_minus.drop_duplicates()
    # Удаление строк, содержащих любое из минус-слов
    for word in minus_words:
        df = df[~df['text'].str.contains(word, case=False, na=False)]
    def clean_text(text):
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # Удаление ссылок
        text = re.sub(r'<.*?>', ' ', text)  # Удаление HTML тегов
        return text
    df['text'] = df['text'].apply(clean_text)
    # так как предыдущими действиями мы скорее всего удалили только ссылки, но оставили обёртки, удаляем обёртки
    df['text'] = df['text'].str.replace('<a href="', ' ')
    df = df.dropna(subset=['text'])
    # ОПРЕДЕЛЕНИЕ ЯЗЫКА
    # Установка размера пакета (batch size)
    batch_size = 16

    def preprocess_texts(texts):
        return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "papluca/xlm-roberta-base-language-detection"
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    predicted_languages_gpu = torch.tensor([], dtype=torch.int64, device=device)

    for i in tqdm(range(0, len(df['text']), batch_size)):  # Добавляем tqdm для отслеживания прогресса
        batch_texts = df['text'][i:i + batch_size].tolist()
        encoded_input = preprocess_texts(batch_texts)

        # Убедитесь, что данные находятся на GPU
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            batch_predictions = model(**encoded_input)
            batch_predicted_languages = torch.argmax(batch_predictions.logits, dim=1)
            # Сохранение предсказаний в GPU-тензоре
            predicted_languages_gpu = torch.cat((predicted_languages_gpu, batch_predicted_languages), 0)

    df['predicted_language'] = predicted_languages_gpu.tolist()
    # ОПРЕДЕЛЕНИЕ ТОНАЛЬНОСТИ

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Загрузка модели и токенизатора
    model_name = "MonoHime/rubert-base-cased-sentiment-new"
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Установка размера пакета (batch size)
    batch_size = 16

    # Функция для подготовки пакетов данных
    def preprocess_texts(texts):
        return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Инициализация массива для хранения предсказаний
    predicted_classes = torch.tensor([], dtype=torch.int64, device=device)

    # Обработка текстов пакетами и получение предсказаний
    for i in tqdm(range(0, len(df['text']), batch_size)):
        batch_texts = df['text'][i:i + batch_size].tolist()
        encoded_input = preprocess_texts(batch_texts)
        # Убедитесь, что данные находятся на GPU
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            batch_predictions = model(**encoded_input)
            batch_predicted_classes = torch.argmax(batch_predictions.logits, dim=1)
            predicted_classes = torch.cat((predicted_classes, batch_predicted_classes), 0)

    # Добавление предсказаний в DataFrame
    df['predicted_class'] = predicted_classes.tolist()
    # СУММАРИЗАЦИЯ
    # Проверка доступности CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Загрузка модели и токенизатора
    model_name = "IlyaGusev/mbart_ru_sum_gazeta"
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

    # Функция для деления текста пополам
    def split_text(text, word_limit=600):
        words = text.split()
        if len(words) <= word_limit:
            return [text]
        mid = len(words) // 2
        return [' '.join(words[:mid]), ' '.join(words[mid:])]

    # Функция для суммаризации текста
    def summarize_text(text):
        # Разделение текста на части, если он слишком длинный
        text_parts = split_text(text)
        summary = ''
        for part in text_parts:
            inputs = tokenizer([part], max_length=1024, return_tensors="pt", truncation=True)
            inputs = inputs.to(device)
            summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
            summary += tokenizer.decode(summary_ids[0], skip_special_tokens=True) + ' '
        return summary.strip()

    # Применение суммаризации с отслеживанием прогресса через tqdm
    tqdm.pandas()
    df['summary'] = df['text'].progress_apply(summarize_text)
    df.to_csv('res.csv', index = False)