import sys, json

def check(path):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"Invalid JSON at {path}:{i}: {e}")
                return False
            if "input" not in obj or "output" not in obj:
                print(f"Missing key 'input' or 'output' at {path}:{i}")
                return False
    print(f"{path}: OK ({sum(1 for _ in open(path, 'r', encoding='utf-8'))} lines)")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/validate_jsonl.py file1.jsonl [file2.jsonl ...]")
        sys.exit(1)
    ok = True
    for p in sys.argv[1:]:
        if not check(p):
            ok = False
    sys.exit(0 if ok else 2)
