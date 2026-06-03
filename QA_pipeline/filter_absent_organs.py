"""
Filter out questions from type_1a.json for organs described as "absent" in reports.

Workflow:
1. Read imaging descriptions from reports_prepare/origin_reports/report_zh.csv, extract all "absent" organs
2. Build anno_id -> img_id (PTXH) mapping from anno_mapping_table.csv
3. Delete questions from type_1a.json where the CT_path's PTXH and correct answer entity match an absent organ
"""

import json
import os
import re
import csv
import yaml
from typing import Dict, List, Set, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load language configuration
with open(os.path.join(BASE_DIR, 'configs.yaml'), 'r', encoding='utf-8') as _f:
    _lang = yaml.safe_load(_f).get('report_language', 'chinese').lower()

# Absent organ extraction only supports Chinese reports (the "XX absent" text pattern in Chinese)
# If configured for English mode, skip absent organ filtering (English reports don't use this pattern)
if _lang == 'chinese':
    REPORT_PATH = os.path.join(BASE_DIR, '../reports_prepare/origin_reports/report_zh.csv')
else:
    REPORT_PATH = os.path.join(BASE_DIR, '../reports_prepare/origin_reports/report_en.csv')

_REPORT_ENCODING = 'utf-8' if _lang == 'chinese' else 'latin-1'
MAPPING_PATH = os.path.join(BASE_DIR, '../anno_mapping_table.csv')
TYPE1A_PATH = os.path.join(BASE_DIR, '../MetaStructVQA_Dataset/QA_pairs/type_1a.json')
OUTPUT_PATH = os.path.join(BASE_DIR, '../MetaStructVQA_Dataset/QA_pairs/type_1a_filtered.json')

# Mapping of common absent organ Chinese names to standardized entity names
# Accounts for various naming conventions in reports
ORGAN_ALIASES = {
    # Gallbladder
    '胆囊': ['胆囊'],
    # Stomach
    '胃': ['胃', '胃底', '胃体', '胃窦', '胃角', '胃小弯', '胃大弯'],
    # Spleen
    '脾': ['脾', '脾门'],
    '脾脏': ['脾', '脾门'],
    # Uterus
    '子宫': ['子宫'],  # Note: uterus may not exist in the entity mapping table
    # Kidney
    '右肾': ['右肾', '肾'],
    '左肾': ['左肾', '肾'],
    '肾': ['肾', '左肾', '右肾'],
    # Lung lobes
    '右肺上叶': ['右肺上叶'],
    '右肺中叶': ['右肺中叶', '肺中叶'],
    '右肺下叶': ['右肺下叶'],
    '左肺上叶': ['左肺上叶'],
    '左肺下叶': ['左肺下叶'],
    '右肺': ['右肺', '右肺上叶', '右肺中叶', '右肺下叶'],
    '左肺': ['左肺', '左肺上叶', '左肺下叶'],
    # Liver
    '肝右叶': ['肝右叶', '肝'],
    '肝左叶': ['肝左叶', '肝'],
}


def load_anno_mapping() -> Dict[str, str]:
    """Load anno_id -> img_id (PTXH) mapping"""
    mapping = {}
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # anno_id column name may be 'anno_id'
            anno_id = row.get('anno_id', '')
            img_id = row.get('Image ID', '')
            if anno_id and img_id:
                mapping[anno_id] = img_id
    return mapping


def extract_absent_organs(description: str) -> List[str]:
    """
    Extract absent organ names from imaging description.
    
    Matching patterns:
    - "XX缺如" (e.g., gallbladder absent, right upper lung lobe absent)
    - "XX术后缺如" (e.g., post-surgical gallbladder absent)
    - "术后，XX缺如" 
    - "XX部分缺如" (e.g., partial right liver lobe absent)
    """
    absent_organs = []
    
    # Pattern 1: Direct match "organ_name + absent"
    # Organ name can be 1-6 Chinese characters
    patterns = [
        r'([\u4e00-\u9fa5]{1,6})(?:术后)?(?:部分)?缺如',  # XX(post-surgical)(partial)absent
        r'术后[，,]?\s*([\u4e00-\u9fa5]{1,6})缺如',  # post-surgical, XX absent
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, description)
        for match in matches:
            organ = match.strip()
            # Filter out non-organ matches (e.g., "FDG uptake absent")
            if organ and not any(x in organ for x in ['FDG', '摄取', '代谢', '信号', '肋']):
                absent_organs.append(organ)
    
    return list(set(absent_organs))


def expand_organ_names(organ: str) -> Set[str]:
    """
    Expand organ name to all possible entity names.
    E.g., "stomach" -> ["stomach", "gastric fundus", "gastric body", ...]
    """
    result = set()
    
    # Direct match
    if organ in ORGAN_ALIASES:
        result.update(ORGAN_ALIASES[organ])
    
    # If no mapping found, use original name
    if not result:
        result.add(organ)
    
    return result


def parse_reports() -> Dict[str, Set[str]]:
    """
    Parse reports, return set of absent organs per anno_id.
    
    Note: Absent organ extraction is based on the "XX absent" text pattern in Chinese reports.
    If configured for English mode, returns empty dict (skips absent organ filtering).
    
    Returns:
        Dict[anno_id, Set[expanded organ names]]
    """
    absent_by_anno = {}

    # English reports don't support absent organ extraction (Chinese-specific text pattern)
    if _lang != 'chinese':
        print("Note: Skipping absent organ extraction in English mode (only supports Chinese report 'XX absent' pattern)")
        return absent_by_anno
    
    with open(REPORT_PATH, 'r', encoding=_REPORT_ENCODING) as f:
        reader = csv.DictReader(f)
        for row in reader:
            anno_id = row.get('影像号', '').strip()
            description = row.get('影像描述', '') + row.get('报告诊断', '')
            
            if not anno_id:
                continue
            
            absent_organs = extract_absent_organs(description)
            if absent_organs:
                expanded = set()
                for organ in absent_organs:
                    expanded.update(expand_organ_names(organ))
                
                if anno_id in absent_by_anno:
                    absent_by_anno[anno_id].update(expanded)
                else:
                    absent_by_anno[anno_id] = expanded
    
    return absent_by_anno


def extract_ptxh_from_path(ct_path: str) -> str:
    """Extract PTXH ID from CT path"""
    # Path format: imgs_data\CT\PTXH220103024_0000.nii.gz
    match = re.search(r'(PTXH\d+)', ct_path)
    return match.group(1) if match else ''


def get_correct_answer_entity(qa: dict) -> str:
    """Get the entity name corresponding to the correct answer of the question"""
    answer_key = qa.get('answer', '')
    options = qa.get('options', {})
    return options.get(answer_key, '')


def main():
    print("=" * 60)
    print("Start filtering questions for absent organs")
    print("=" * 60)
    
    # 1. Load anno_id -> PTXH mapping
    anno_to_ptxh = load_anno_mapping()
    print(f"[1] Loaded mapping table: {len(anno_to_ptxh)} record(s)")
    
    # 2. Parse reports, get absent organs per anno_id
    absent_by_anno = parse_reports()
    print(f"[2] Parsed reports: found {len(absent_by_anno)} image(s) with absent organs")
    
    # Print details
    for anno_id, organs in sorted(absent_by_anno.items()):
        ptxh = anno_to_ptxh.get(anno_id, 'N/A')
        print(f"    {anno_id} ({ptxh}): {', '.join(sorted(organs))}")
    
    # 3. Build PTXH -> absent organs mapping
    absent_by_ptxh: Dict[str, Set[str]] = {}
    for anno_id, organs in absent_by_anno.items():
        ptxh = anno_to_ptxh.get(anno_id, '')
        if ptxh:
            if ptxh in absent_by_ptxh:
                absent_by_ptxh[ptxh].update(organs)
            else:
                absent_by_ptxh[ptxh] = organs.copy()
    
    print(f"[3] Built PTXH mapping: {len(absent_by_ptxh)} PTXH ID(s) with absent organs")
    
    # 4. Load type_1a.json
    with open(TYPE1A_PATH, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    original_count = len(qa_pairs)
    print(f"[4] Loaded type_1a.json: {original_count} question(s)")
    
    # 5. Filter questions
    filtered_pairs = []
    removed_count = 0
    removed_details = []
    
    for qa in qa_pairs:
        ct_path = qa.get('CT_path', '')
        ptxh = extract_ptxh_from_path(ct_path)
        entity = get_correct_answer_entity(qa)
        
        # Check if should be removed
        should_remove = False
        if ptxh in absent_by_ptxh:
            absent_organs = absent_by_ptxh[ptxh]
            if entity in absent_organs:
                should_remove = True
                removed_details.append({
                    'q_id': qa.get('q_id'),
                    'ptxh': ptxh,
                    'entity': entity
                })
        
        if should_remove:
            removed_count += 1
        else:
            filtered_pairs.append(qa)
    
    print(f"[5] Filtering complete: removed {removed_count}, kept {len(filtered_pairs)}")
    
    # Print removed question details
    if removed_details:
        print("\nRemoved question details:")
        for detail in removed_details[:20]:  # Show first 20 only
            print(f"    q_id={detail['q_id']}, PTXH={detail['ptxh']}, entity={detail['entity']}")
        if len(removed_details) > 20:
            print(f"    ... {len(removed_details) - 20} more")
    
    # 6. Save results
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(filtered_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"\n[6] Results saved to: {OUTPUT_PATH}")
    print("=" * 60)
    print(f"Summary: original {original_count} -> filtered {len(filtered_pairs)} (removed {removed_count})")
    print("=" * 60)


if __name__ == '__main__':
    main()
