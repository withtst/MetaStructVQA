"""Type 2 split script
Splits the merged type_2.json into separate type_2a.json (multiple choice) and type_2b.json (true/false),
with re-numbering.
"""
import json
import os


def _load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        return json.loads(content)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _renumber(items):
    for i, item in enumerate(items):
        item["q_id"] = f"{i:06d}"
    return items


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    qa_pairs_dir = os.path.join(base_dir, "../MetaStructVQA_Dataset", "QA_pairs")

    src_path = os.path.join(qa_pairs_dir, "type_2.json")
    out_2a = os.path.join(qa_pairs_dir, "type_2a.json")
    out_2b = os.path.join(qa_pairs_dir, "type_2b.json")

    data = _load_json(src_path)

    type_2a = [item for item in data if item.get("q_type") == "2a"]
    type_2b = [item for item in data if item.get("q_type") == "2b"]

    _renumber(type_2a)
    _renumber(type_2b)

    _write_json(out_2a, type_2a)
    _write_json(out_2b, type_2b)

    print(f"Written: {out_2a} (count: {len(type_2a)})")
    print(f"Written: {out_2b} (count: {len(type_2b)})")


if __name__ == "__main__":
    main()
