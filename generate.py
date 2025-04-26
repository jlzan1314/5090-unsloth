from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    # Can select any from the below:
    # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
    # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
    # And also all Instruct versions and Math. Coding verisons!
    #model_name = "unsloth/Qwen2.5-14B",
    model_name = "lora_model",
    #model_name = "outputs/checkpoint-2000",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# Define the alpaca prompt template
alpaca_prompt = "以下是一个描述任务的指令，以及提供更多上下文的输入内容。请写出恰当的回应来完成这个请求。\n\n### 指令：\n{0}\n\n### 输入：\n{1}\n\n### 回应：\n{2}"

inputs = tokenizer(
[
    alpaca_prompt.format(
        "剑来小说:十四境大妖有哪些?", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

if __name__ == "__main__":
    from transformers import TextStreamer 
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        **inputs,
        streamer = text_streamer,
        max_new_tokens = 1024,  # 生成的最大token数
        temperature = 0.5,      # 控制生成文本的随机性,值越大随机性越强
        top_p = 0.9,           # 控制采样范围,只从概率最高的tokens中采样
        top_k = 50,            # 只从概率最高的前k个token中采样
        repetition_penalty = 1.1,  # 重复惩罚因子,避免重复生成相同内容
        do_sample = True,      # 使用采样而不是贪婪解码
        num_beams = 1,         # beam search的束宽,1表示不使用beam search
        pad_token_id = tokenizer.pad_token_id,  # padding token的ID
        eos_token_id = tokenizer.eos_token_id   # 结束符token的ID
    )
