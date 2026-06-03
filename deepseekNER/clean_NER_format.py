'''
Script to check and fix NER result file format.
Ensures all descriptions/*.txt files conform to the [{...}] JSON list format.
'''

import os

def check_files(directory):
    """
    Check all txt files in the given directory for [{...}] format compliance.
    """
    print(f"Checking directory: {directory} ...")
    error_count = 0
    total_count = 0

    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            total_count += 1
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                # Check if content starts with [ and ends with ]
                if not (content.startswith('[') and content.endswith(']')):
                    print(f"[FORMAT ERROR] {filename}")
                    print(f"    Start: {content[:20]!r}")
                    print(f"    End: {content[-20:]!r}")
                    error_count += 1
            except Exception as e:
                print(f"[READ ERROR] {filename}: {e}")
                error_count += 1

    print(f"\nCheck complete. Total files: {total_count}, Errors: {error_count}")

def fix_files(directory):
    """
    Fix all txt files in the given directory by removing Markdown code block markers,
    ensuring content starts with [ and ends with ].
    """
    print(f"Fixing directory: {directory} ...")
    fixed_count = 0

    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read().strip()

                content = original_content

                # Attempt cleanup if not in standard format
                if not (content.startswith('[') and content.endswith(']')):

                    # 1. Remove leading '''json or ```json
                    if content.startswith("'''"):
                        if content.lower().startswith("'''json"):
                            content = content[7:]
                        elif content.startswith("'''"):
                            content = content[3:]
                    elif content.startswith("```"):
                        if content.lower().startswith("```json"):
                            content = content[7:]
                        elif content.startswith("```"):
                            content = content[3:]

                    content = content.strip()

                    # 2. Remove trailing ''' or ```
                    if content.endswith("'''"):
                        content = content[:-3]
                    elif content.endswith("```"):
                        content = content[:-3]

                    content = content.strip()

                    # 3. Re-check and save if fixed
                    if content.startswith('[') and content.endswith(']'):
                        if content != original_content:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                            print(f"[FIXED] {filename}")
                            fixed_count += 1
                    else:
                        print(f"[UNFIXABLE] {filename} - still not valid JSON list format after cleanup")
                        print(f"    Current start: {content[:20]!r}")
                        print(f"    Current end: {content[-20:]!r}")

            except Exception as e:
                print(f"[PROCESSING ERROR] {filename}: {e}")

    print(f"\nFix complete. Files fixed: {fixed_count}")

if __name__ == "__main__":
    target_bd = os.path.join(os.path.dirname(__file__), "descriptions")

    print("=== First pass check ===")
    check_files(target_bd)

    print("\n=== Applying fixes ===")
    fix_files(target_bd)

    print("\n=== Second pass check (verification) ===")
    check_files(target_bd)
