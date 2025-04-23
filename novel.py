import json
import os

def preprocess_and_chunk_novel(input_txt_path, output_jsonl_path, max_chunk_length=1024, paragraph_separator='\n\n'):
    """
    预处理小说文本并按段落切割成固定长度的样本。

    Args:
        input_txt_path (str): 输入的小说纯文本文件路径。
        output_jsonl_path (str): 输出的JSON Lines文件路径。
        max_chunk_length (int): 每个文本块的最大字符数（或大致Token数）。
        paragraph_separator (str): 用于分割段落的字符或字符串。
    """
    chunks = []
    current_chunk = ""

    try:
        with open(input_txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 简单的预处理：统一换行符，去除过多空白行
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # 将多个连续换行符替换为两个，作为统一的段落分隔
        text = os.linesep.join([s for s in text.splitlines() if s.strip()])
        text = text.replace('\n\n\n', '\n\n') # Reduce triple newlines to double

        paragraphs = text.split(paragraph_separator)

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # 如果当前段落加上当前chunk不会超过最大长度，就追加
            if len(current_chunk) + len(paragraph) + len(paragraph_separator) <= max_chunk_length:
                if current_chunk:
                    current_chunk += paragraph_separator
                current_chunk += paragraph
            else:
                # 如果当前chunk已非空，且加入新段落会超长，则将当前chunk保存
                if current_chunk:
                    chunks.append({"text": current_chunk})
                # 开始新的chunk，如果新段落本身就超长，则需要进一步处理
                if len(paragraph) > max_chunk_length:
                    # 这里简化处理：直接将长段落也作为单个chunk，或者按长度硬切
                    # 更复杂的处理是：将长段落再按句子或其他方式细分
                    # 简化版：直接作为chunk（可能超长，训练时会被截断或需要特殊处理）
                    # 或者，更安全的：按长度硬切长段落
                    sub_chunks = [paragraph[i:i + max_chunk_length] for i in range(0, len(paragraph), max_chunk_length)]
                    for sc in sub_chunks:
                         if sc.strip(): # 确保不是空字符串
                              chunks.append({"text": sc.strip()})
                    current_chunk = "" # 新段落处理完毕，重置current_chunk
                else:
                     # 新段落放入新的chunk
                     current_chunk = paragraph


        # 添加最后一个chunk（如果非空）
        if current_chunk:
            chunks.append({"text": current_chunk})

    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return

    # 保存为 JSON Lines
    try:
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        print(f"数据集已成功保存到 {output_jsonl_path}，共生成 {len(chunks)} 条样本。")
    except Exception as e:
        print(f"保存数据集文件时发生错误: {e}")


# --- 如何使用 ---

if  os.path.exists("novel_dataset.jsonl"):
    dataset = load_dataset('json', data_files='novel_dataset.jsonl', split='train')
    # 查看数据集结构
    print(dataset)
    print(dataset[0]) # 查看第一个样本

    # Unsloth 或 Hugging Face Trainer 通常需要一个 'text' 列，如果你的jsonl里就是 {"text": ...} 那就完美匹配。
    # 如果你的列名不同，你需要使用 .rename_column() 或在Trainer参数中指定。
    # 例如，如果你的列名是 'content'：
    # dataset = dataset.rename_column('content', 'text')
    # 接下来就可以将 dataset 传递给 Unsloth 的模型和训练器进行微调了。
    # Unsloth/transformers Trainer 会负责后续的 Tokenization, batching 等。

if __name__ == "__main__":

# --- 如何使用 ---
    input_novel_file = r'/mnt/f/ai/剑来.txt' # 替换成你的小说文件路径
    output_dataset_file = 'novel_dataset.jsonl' # 输出的数据集文件路径
    # 根据你的模型和需求调整最大长度
    # 注意：这里的长度是字符数，实际Token数会不同，通常Token数 < 字符数
    max_chars_per_chunk = 1500 # 例如，设置为1500个字符，留出Tokenizer的余地
    preprocess_and_chunk_novel(input_novel_file, output_dataset_file, max_chunk_length=max_chars_per_chunk)
    from datasets import load_dataset
    # 加载本地的jsonl文件
    # 'json' 指示加载器类型， data_files 指定文件路径
    # split='train' 是常规做法


