from model import model, tokenizer

inputs = tokenizer(
[
    alpaca_prompt.format(
        "What is a famous tall tower in Paris?", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

if __main__ == "__main__":
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
