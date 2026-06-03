"""
Split the compound entity "肾盂肾盏" (renal pelvis and calyx) into separate
"肾盂" (renal pelvis) and "肾盏" (renal calyx) entities.
Used to fix compound entity naming issues in NER results.
"""
import json
import os
import copy


def load_file(file_path):
    """Try reading file as UTF-8, fall back to GBK"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f), 'utf-8'
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return json.load(f), 'gbk'
        except Exception as e:
            print(f"Failed to read as GBK {file_path}: {e}")
            return None, None
    except json.JSONDecodeError as e:
        print(f"JSON parse error {file_path}: {e}")
        return None, None
    except Exception as e:
        print(f"File read error {file_path}: {e}")
        return None, None


def process_item_content(item, new_entity_name):
    """Update entity descriptions with the new entity name"""
    new_item = copy.deepcopy(item)
    new_item["anatomical_entity"] = new_entity_name

    if "morphological_description" in new_item:
        new_morphs = []
        for desc in new_item["morphological_description"]:
            if desc == "None":
                new_morphs.append(desc)
            else:
                new_morphs.append(desc.replace("肾盂肾盏", new_entity_name))
        new_item["morphological_description"] = new_morphs

    if "fdg_uptake_description" in new_item:
        new_fdgs = []
        for desc in new_item["fdg_uptake_description"]:
            if desc == "None":
                new_fdgs.append(desc)
            else:
                new_fdgs.append(desc.replace("肾盂肾盏", new_entity_name))
        new_item["fdg_uptake_description"] = new_fdgs

    return new_item


def split_kidney_pelvis_calyx(data):
    """Split '肾盂肾盏' compound entities in the data list"""
    new_data = []
    modified = False

    for item in data:
        if item.get("anatomical_entity") == "肾盂肾盏":
            modified = True
            pelvis_item = process_item_content(item, "肾盂")
            new_data.append(pelvis_item)
            calyx_item = process_item_content(item, "肾盏")
            new_data.append(calyx_item)
        else:
            new_data.append(item)

    return new_data, modified


def process_all_files(file_list_str):
    """Batch process files containing the '肾盂肾盏' compound entity"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    descriptions_dir = os.path.join(script_dir, "descriptions")

    lines = file_list_str.strip().split('\n')
    files_to_process = []

    for line in lines:
        if ":" in line:
            filename = line.split(":")[0].strip()
            if filename.endswith(".txt"):
                files_to_process.append(os.path.join(descriptions_dir, filename))

    files_to_process = list(set(files_to_process))
    files_to_process.sort()

    print(f"Preparing to process {len(files_to_process)} file(s).")

    success_count = 0
    error_count = 0

    for file_path in files_to_process:
        try:
            data, encoding = load_file(file_path)
            if data is None:
                error_count += 1
                continue

            new_data, modified = split_kidney_pelvis_calyx(data)

            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=2)
                print(f"[OK] Updated {os.path.basename(file_path)}")
                success_count += 1
            else:
                print(f"[SKIP] {os.path.basename(file_path)} (target entity not found)")

        except Exception as e:
            print(f"[ERROR] Failed to process {os.path.basename(file_path)}: {e}")
            error_count += 1

    print(f"\nProcessing complete. Success: {success_count}, Failed: {error_count}")


# List of files to process (obtained by searching for "肾盂肾盏" using find_entity_location.py)
# Usage: python find_entity_location.py 肾盂肾盏 > target_files.txt
# Then paste the output into TARGET_FILES below
TARGET_FILES = """
# Paste search results from find_entity_location.py here
# Example format:
# sub_004.txt : Line 346
# sub_008.txt : Line 340
"""

if __name__ == "__main__":
    import sys

    script_dir = os.path.dirname(os.path.abspath(__file__))
    descriptions_dir = os.path.join(script_dir, "descriptions")

    if len(sys.argv) > 1 and sys.argv[1] == "run_all":
        print("====== Batch processing mode ======")
        process_all_files(TARGET_FILES)
    else:
        print("====== Usage ======")
        print("1. Run find_entity_location.py to find target entity locations")
        print("2. Paste the search results into the TARGET_FILES variable in this file")
        print("3. Run: python split_entities.py run_all")
