from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# --- Load model/tokenizer ---
model_path = "checkpoints_sst2/best"
device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)

# --- Single-text classifier with probs + neutral threshold ---
def classify(text: str, threshold: float = 0.60):
    # tokenize -> tensors on the right device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits                      # shape [1, 2]
        probs = torch.softmax(logits, dim=-1)[0]     # tensor([neg, pos])

    neg, pos = probs.tolist()

    # neutral band: only commit if confidence >= threshold
    if pos >= threshold:
        label = "positive"
    elif neg >= threshold:
        label = "negative"
    else:
        label = "neutral"

    # print probabilities and final label
    print({"neg": float(neg), "pos": float(pos), "label": label})
    return label

if __name__ == "__main__":
    texts = [
        "I really hate this movie!",
        "I really love this movie!",
        "This movie was just alright. Not good, not bad.",
    ]
    for t in texts:
        print(t)
        classify(t)
        print()
