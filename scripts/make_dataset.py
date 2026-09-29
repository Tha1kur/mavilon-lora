import os, json, random

os.makedirs("data", exist_ok=True)

seed = [
 {"input":"Summarize: The cat sat on the mat.","output":"A cat sat on a mat."},
 {"input":"Translate to Hindi: How are you?","output":"कैसे हो?"},
 {"input":"Q: What is 7 * 8?","output":"56"},
 {"input":"Write a polite email asking for a meeting tomorrow.","output":"Hello, could we meet tomorrow at 3 PM?"},
 {"input":"Fix grammar: he go to school everyday.","output":"He goes to school every day."},
 {"input":"Explain simply: What is machine learning?","output":"Machine learning means computers learn patterns from data."},
 {"input":"Given list [1,2,3], return its sum.","output":"6"},
 {"input":"Create a 3-step plan to study.","output":"1) Make schedule\n2) Practice\n3) Revise"},
 {"input":"Who wrote Hamlet?","output":"William Shakespeare"},
 {"input":"Benefits of exercise?","output":"- Better health\n- More energy\n- Strong muscles"}
]

examples = []
for i in range(10):
    for ex in seed:
        e = ex.copy()
        if random.random() < 0.3:
            e["input"] += " (brief)"
        examples.append(e)

random.shuffle(examples)
split = int(len(examples) * 0.85)
train = examples[:split]
valid = examples[split:]

def write_jsonl(path, arr):
    with open(path, "w", encoding="utf-8") as f:
        for x in arr:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

write_jsonl("data/train.jsonl", train)
write_jsonl("data/valid.jsonl", valid)

print("Dataset created.")
