from datasets import load_dataset
from model import tokenizer, max_seq_length

EOS_TOKEN = tokenizer.eos_token # 必须添加 EOS_TOKEN
def formatting_prompts_func(examples):
    texts = examples["text"]+EOS_TOKEN
    return { "text" : texts, }

# 加载数据集
dataset = load_dataset("json", data_files="novel_dataset.jsonl", split="train")

# 应用格式化
# dataset = dataset.map(formatting_prompts_func, batched = True,)

# 可选：打印一些样本以进行验证
# print("Formatted dataset sample:")
# print(dataset[0]['text'])

# 数据集现在已准备好，可以在 main.py 中导入和使用
# 例如，在 main.py 中:
# from novel_dataset import dataset
# trainer = SFTTrainer(
#     model = model,
#     tokenizer = tokenizer,
#     train_dataset = dataset, # 使用导入的数据集
#     ...
# )