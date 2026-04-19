import os
import csv
import argparse
from run_evaluation import SERVER_URL
import requests

def convert_folder_to_csv(folder_path):
    """
    Читает папку с текстовыми файлами в память.
    Возвращает (column_names, rows) где каждая строка - это список текстов.
    
    Args:
        folder_path: путь к папке с .txt файлами
        
    Returns:
        (column_names, rows) - список имен колонок и список строк с текстами
    """
    # Получаем все .txt файлы в папке
    txt_files = sorted([f for f in os.listdir(folder_path) 
                       if f.endswith('.txt')])
    
    if not txt_files:
        print(f"Не найдено .txt файлов в {folder_path}")
        return [], []
    
    print(f"Найдено файлов: {len(txt_files)}")
    print(f"Файлы: {', '.join(txt_files)}")
    
    # Column names = названия файлов без расширения .txt
    column_names = [os.path.splitext(f)[0] for f in txt_files]
    
    # Читаем содержимое всех файлов
    file_contents = {}
    max_lines = 0
    
    for txt_file in txt_files:
        filepath = os.path.join(folder_path, txt_file)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            file_contents[txt_file] = lines
            max_lines = max(max_lines, len(lines))
    
    print(f"Максимальное количество строк в файлах: {max_lines}")
    
    # Строки данных
    rows = []
    for i in range(max_lines):
        row = []
        for col_name in column_names:
            # Используем соответствующий txt_file для доступа к file_contents
            txt_file = col_name + '.txt'
            lines = file_contents.get(txt_file, [])
            if i < len(lines):
                row.append(lines[i])
            else:
                # Если в этом файле меньше строк, оставляем пустую ячейку
                row.append("")
        rows.append(row)
    
    return column_names, rows

def main():
    parser = argparse.ArgumentParser(
        description='Оценивает тексты из папки с .txt файлами'
    )
    parser.add_argument(
        'folder_path', 
        type=str,
        help='Путь к папке с .txt файлами'
    )
    parser.add_argument(
        '--topic',
        type=str,
        default=None,
        help='Название топика (опционально)'
    )
    parser.add_argument(
        '--style',
        type=str,
        required=True,
        help='Название стиля (обязательно)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Путь для сохранения результата (по умолчанию: results/{folder_name}/ranked.csv)'
    )
    
    args = parser.parse_args()
    
    # Проверяем, что папка существует
    if not os.path.isdir(args.folder_path):
        print(f"Ошибка: {args.folder_path} не является папкой")
        return
    
    # Читаем папку
    column_names, rows = convert_folder_to_csv(args.folder_path)
    if not column_names:
        return
    
    # Определяем выходной файл
    folder_name = os.path.basename(os.path.normpath(args.folder_path))
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.join("results", folder_name)
        output_path = os.path.join(output_dir, "ranked.csv")
    
    # Создаем папку если не существует
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Проверяем, что сервер запущен
    try:
        requests.get("http://localhost:1337/docs", timeout=1)
    except requests.exceptions.ConnectionError:
        print("\nПредупреждение: Сервер не запущен на http://localhost:1337")
        print("Запустите сервер с помощью: python rank_server.py")
        return
    
    # Запускаем оценку
    print("\n" + "="*50)
    print("Запуск оценки...")
    print("="*50 + "\n")
    
    run_ranking(column_names, rows, args.topic, args.style, output_path)

def run_ranking(column_names, rows, topic, style, output_path):
    """
    Ранжирует все тексты и сохраняет результат.
    """
    num_cols = len(column_names)
    num_rows = len(rows)
    
    print(f"Topic: '{topic}', Style: '{style}'")
    print(f"Ранжирование {num_rows} строк x {num_cols} колонок")
    
    output_rows = [column_names]
    
    for row_idx, row in enumerate(rows):
        texts = [cell for cell in row if cell.strip()]
        
        if len(texts) == 0:
            print(f"Пропускаем пустую строку {row_idx+1}")
            continue
        
        if len(texts) != num_cols:
            print(f"Предупреждение: Строка {row_idx+1} содержит {len(texts)} текстов, но {num_cols} колонок")
        
        payload = {
            "style_name": style,
            "texts": texts
        }
        if topic:
            payload["topic"] = topic
        
        try:
            response = requests.post(SERVER_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            ranked_indices = result['ranked_indices']
            
            # Конвертируем индексы в ранги
            ranks = [0] * len(texts)
            for rank, original_idx in enumerate(ranked_indices):
                ranks[original_idx] = rank + 1
            
            output_rows.append(ranks)
            print(f"Строка {row_idx+1} обработана.")
            
        except Exception as e:
            print(f"Ошибка обработки строки {row_idx+1}: {e}")
            output_rows.append(["ERROR"] * len(texts))
    
    # Сохраняем результат
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)
    
    print(f"\nРезультаты сохранены в: {output_path}")

if __name__ == "__main__":
    main()
