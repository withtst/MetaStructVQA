"""
Interactively delete questions for a specified anno_id and entity name (loop).

After running, enter anno_id and entity name in the terminal. Type 'q' to quit.
"""

import json
import os
import re
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_PATH = os.path.join(BASE_DIR, '../anno_mapping_table.csv')
TYPE1A_PATH = os.path.join(BASE_DIR, '../MetaStructVQA_Dataset/QA_pairs/type_1a.json')


def load_anno_mapping():
    """Load anno_id -> PTXH mapping"""
    mapping = {}
    with open(MAPPING_PATH, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM automatically
        reader = csv.DictReader(f)
        for row in reader:
            anno_id = row.get('anno_id', '')
            img_id = row.get('Image ID', '')
            if anno_id and img_id:
                mapping[anno_id] = img_id
    return mapping


def extract_ptxh_from_path(ct_path: str) -> str:
    """Extract PTXH ID from CT path"""
    match = re.search(r'(PTXH\d+)', ct_path)
    return match.group(1) if match else ''


def get_correct_answer_entity(qa: dict) -> str:
    """Get the entity name corresponding to the correct answer of the question"""
    answer_key = qa.get('answer', '')
    options = qa.get('options', {})
    return options.get(answer_key, '')


def normalize_anno_id(input_str: str) -> str:
    """
    Normalize input to sub_xxx format.
    Supports: 001, 12, 450, sub_012, etc.
    """
    input_str = input_str.strip().lower()
    if input_str.startswith('sub_'):
        return input_str
    # Pure number, pad to sub_xxx
    try:
        num = int(input_str)
        return f"sub_{num:03d}"
    except ValueError:
        return input_str  # Return as-is if not parseable


def delete_questions(qa_pairs, anno_to_ptxh, anno_id, entity_name):
    """Delete questions matching the given anno_id and entity name, return updated list"""
    if anno_id not in anno_to_ptxh:
        print(f"Error: PTXH ID not found for anno_id '{anno_id}'")
        return qa_pairs
    
    ptxh = anno_to_ptxh[anno_id]
    print(f"anno_id: {anno_id} -> PTXH: {ptxh}")
    print(f"Entity to delete: {entity_name}")
    print("-" * 40)
    
    original_count = len(qa_pairs)
    
    # Filter questions to delete
    to_remove = []
    for i, qa in enumerate(qa_pairs):
        ct_path = qa.get('CT_path', '')
        qa_ptxh = extract_ptxh_from_path(ct_path)
        entity = get_correct_answer_entity(qa)
        
        if qa_ptxh == ptxh and entity == entity_name:
            to_remove.append(i)
            print(f"  Found: q_id={qa.get('q_id')}, answer={qa.get('answer')}, entity={entity}")
    
    if not to_remove:
        print(f"No questions found with PTXH={ptxh} and entity={entity_name}")
        return qa_pairs
    
    print("-" * 40)
    print(f"Found {len(to_remove)} question(s) to delete")
    
    # Confirm deletion
    confirm = input("Confirm deletion? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        return qa_pairs
    
    # Delete questions (reverse order to avoid index issues)
    for i in reversed(to_remove):
        del qa_pairs[i]
    
    print(f"Deletion complete: {original_count} -> {len(qa_pairs)} (removed {len(to_remove)})")
    return qa_pairs


def main():
    print("=" * 50)
    print("Delete questions for a given anno_id and entity name")
    print("Input format: anno_id entity_name (space separated)")
    print("Type 'q' to quit and save")
    print("=" * 50)
    
    # Load mapping table
    anno_to_ptxh = load_anno_mapping()
    print(f"Loaded mapping table: {len(anno_to_ptxh)} record(s)")
    
    # Load type_1a.json
    with open(TYPE1A_PATH, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    initial_count = len(qa_pairs)
    print(f"Loaded type_1a.json: {initial_count} question(s)")
    print("=" * 50)
    
    modified = False
    
    while True:
        try:
            user_input = input("\nEnter (anno_id entity_name): ").strip()
        except EOFError:
            break
        
        if user_input.lower() == 'q':
            break
        
        if not user_input:
            continue
        
        parts = user_input.split()
        if len(parts) < 2:
            print("Format error, please enter: anno_id entity_name")
            continue
        
        anno_id = normalize_anno_id(parts[0])
        entity_name = ' '.join(parts[1:])  # Entity name may contain spaces
        
        old_count = len(qa_pairs)
        qa_pairs = delete_questions(qa_pairs, anno_to_ptxh, anno_id, entity_name)
        if len(qa_pairs) < old_count:
            modified = True
            # Save immediately after each deletion
            with open(TYPE1A_PATH, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
            print(f"Saved to: {TYPE1A_PATH}")
    
    # Save results
    if modified:
        print(f"\nSummary: {initial_count} -> {len(qa_pairs)} (deleted {initial_count - len(qa_pairs)} total)")
    else:
        print("\nNo changes made")


if __name__ == '__main__':
    main()
