import argparse
import json
import pandas as pd
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def format_dct(dct):
    h1 = dct.get('h1')
    articles = dct.get('text')
    full_text = f'## {h1}\n\n'
    
    for article in articles:
        header2_name = article.get('header2_name', '')
        text = article.get('text', '')

        if not header2_name and h1:
            header2_name = h1
        if header2_name in ['Contents', 'See also'] or not text: 
            continue

        full_text += f"### {header2_name}\n{text.strip()}\n\n"
    
    return dct | {'full_text': full_text.strip()}


def compile_buffers_smart(page, max_len=300):
    header = page['h1']
    url = page['url']
    
    sections = []
    for dct in page['text']:
        header2_name = dct['header2_name']
        if header2_name in ['Contents', 'See also'] or not dct['text']: 
            continue
        sections.append(f"###{header2_name}\n{dct['text']}")
    
    full_text = "\n\n===SECTION_BREAK===\n\n".join(sections)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_len,
        chunk_overlap=0,
        separators=[
            "\n\n===SECTION_BREAK===\n\n",
            "\n\n",
            "\n",
            "."
        ],
        length_function=len
    )
    
    chunks = splitter.split_text(full_text)
    
    return [
        {'url': url, 'text': f'##{header}\n\n{chunk.strip()}'} 
        for chunk in chunks
    ]


def main():
    parser = argparse.ArgumentParser(description="Предобработка данных Moodle")
    parser.add_argument("--input", default="data/scraped_pages.json")
    parser.add_argument("--output", default="data/buffer_table.csv")
    parser.add_argument("--chunk-size", type=int, default=300)
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Ошибка: файл {input_path} не найден!")
        return
    
    print(f"Загружаем данные из {input_path}")
    json_data = json.loads(input_path.read_text(encoding="utf-8"))
    pages = list(json_data.values())
    print(f"Загружено {len(pages)} страниц")
    
    print("Форматируем тексты...")
    table_of_texts = pd.DataFrame(map(format_dct, pages))
    table_of_texts = (
        table_of_texts
          .sort_values('url')
          .drop_duplicates(subset='full_text', keep='first')
          .reset_index(drop=True)
    )
    
    print("Разбиваем на чанки...")
    buffer_table = table_of_texts.apply(
        lambda x: compile_buffers_smart(x, args.chunk_size), axis=1
    )
    buffer_table = pd.DataFrame(buffer_table.explode().dropna().to_list())
    
    print(f"Сохраняем {len(buffer_table)} чанков в {args.output}")
    Path(args.output).parent.mkdir(exist_ok=True)
    buffer_table.to_csv(args.output, index=False, encoding="utf-8")
    
    print(f"Готово! Средняя длина чанка: {buffer_table.text.str.len().mean():.0f}")


if __name__ == "__main__":
    main()