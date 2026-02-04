from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def generate_response(prompt: str) -> str:
    """
    Generates a response using a Hugging Face FLAN-T5 model.
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7
)


    return tokenizer.decode(outputs[0], skip_special_tokens=True)
