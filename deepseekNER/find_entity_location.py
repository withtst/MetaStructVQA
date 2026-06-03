"""
Search for specified anatomical entity locations in NER result files (descriptions/*.txt).
"""
import os
import sys


def read_lines_safe(file_path):
    """Try reading file with UTF-8, fall back to GBK if that fails"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.readlines()
        except Exception:
            return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def search_entity(target_value, descriptions_dir=None):
    if descriptions_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        descriptions_dir = os.path.join(script_dir, "descriptions")

    if not os.path.exists(descriptions_dir):
        descriptions_dir = os.path.join(os.getcwd(), "deepseekNER", "descriptions")
        if not os.path.exists(descriptions_dir):
            descriptions_dir = os.path.join(os.getcwd(), "descriptions")
            if not os.path.exists(descriptions_dir):
                print(f"Error: descriptions directory not found.")
                return

    print(f"Searching for anatomical_entity = '{target_value}' in '{descriptions_dir}' ...")
    print("-" * 50)

    count = 0
    files = os.listdir(descriptions_dir)
    files.sort()

    for filename in files:
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(descriptions_dir, filename)
        lines = read_lines_safe(file_path)

        if not lines:
            continue

        for index, line in enumerate(lines):
            if '"anatomical_entity"' in line:
                if ":" in line:
                    parts = line.split(":", 1)
                    val_part = parts[1].strip()

                    if val_part.endswith(","):
                        val_part = val_part[:-1]

                    val_part = val_part.strip('"').strip("'")

                    if val_part == target_value:
                        print(f"{filename} : Line {index + 1}")
                        count += 1

    print("-" * 50)
    print(f"Search complete. Found {count} match(es).")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = input("Enter the anatomical_entity value to search for: ")
        if not target.strip():
            print("No valid input provided, exiting.")
            sys.exit(0)

    search_entity(target.strip())
