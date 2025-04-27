from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from model import model, tokenizer
from novel_dataset import dataset, max_seq_length

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,# 每个设备的批次大小
        gradient_accumulation_steps = 4, # 梯度累积步,可以通过增加这个值来模拟更大的批次
        warmup_steps = 5, # 预热步数
        #num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 300,
        learning_rate = 2e-4, # 学习率
        fp16 = not is_bfloat16_supported(), # 如果不支持 bfloat16 则使用 fp16
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",#使用普通AdamW优化器
        weight_decay = 0.01,# 权重衰减，防止过拟合
        lr_scheduler_type = "linear", # 学习率调度器类型
        seed = 3407, # 随机种子，确保实验可重复
        output_dir = "outputs",  # 输出目录
        report_to = "none",  # 不使用外部监控工具
        save_strategy = "steps",    # 按步数保存
        save_steps = 500,           # 每 500 步保存一次
        save_total_limit = 2,       # 最多保留 3 个检查
    ),
)

if __name__ == "__main__":
    print("Training...")
    trainer.train()
    print("Saving LoRA model...")
    model.save_pretrained("lora_model")
    #model.save_pretrained_gguf("novel_model", tokenizer)
