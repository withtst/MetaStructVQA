'''
    Entity filtering module: based on TotalSegmentator segmentation results and DeepSeek NER results,
    filters meaningful entity-description pairs and associates them with corresponding mask files.

    Prerequisites before filtering:
    - TotalSegmentator segmentation completed, with corresponding mask files generated
    - DeepSeek NER entity recognition completed, with corresponding entity description files generated
    - Segmentation info and description files integrated and metadata files generated
    - Metadata path configured in the yaml config file under pre_version_folder
'''

import os
import json
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

config_path = os.path.join(BASE_DIR, 'configs.yaml')

with open(config_path, 'r') as file:
    _raw_configs = yaml.safe_load(file)

# Resolve relative paths in config to absolute paths relative to PROJECT_ROOT
configs = {}
_PATH_KEYS = {'pre_version_folder', 'output_folder', 'entities_mapping_table', 'NER_folder'}
for key, value in _raw_configs.items():
    if key in _PATH_KEYS and isinstance(value, str) and not os.path.isabs(value):
        configs[key] = os.path.join(PROJECT_ROOT, value)
    else:
        configs[key] = value


def get_seg_name(entity_name: str, seg_mapping_path: str = None) -> list:
    '''Get the list of English segmentation names for an entity'''
    if seg_mapping_path is None:
        seg_mapping_path = configs.get('entities_mapping_table',
                                       os.path.join(PROJECT_ROOT, 'entities_nonsub_mapping_table.json'))
    with open(seg_mapping_path, "r", encoding="utf-8") as f:
        seg_mapping = json.load(f)
    return seg_mapping.get(entity_name, [])


def NER_result_parser(anno_id: str) -> list:
    '''Parse NER result file and return list of entity objects'''
    ner_folder = configs.get('NER_folder', os.path.join(PROJECT_ROOT, 'deepseekNER', 'descriptions'))
    ner_file_path = os.path.join(ner_folder, f"{anno_id}.txt")
    try:
        with open(ner_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse failed: {ner_file_path}")
        print(f"Error: {e}")
        raise


def main_filter(test_mode=False):
    '''Main filtering function'''
    pre_version_folder = configs.get('pre_version_folder',
                                     os.path.join(PROJECT_ROOT, 'VQA_pre_version'))
    output_folder = configs.get('output_folder',
                                os.path.join(PROJECT_ROOT, 'MetaStructVQA_Dataset', 'annotations'))
    os.makedirs(output_folder, exist_ok=True)

    for anno_id in os.listdir(pre_version_folder):
        anno_path = os.path.join(pre_version_folder, anno_id)
        if not os.path.isdir(anno_path):
            continue

        metadata_path = os.path.join(anno_path, 'metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        CT_path = metadata['CT_path']
        PET_reg2CT_path = metadata['PET_reg2CT_path']
        CT_seg_folder = metadata['CT_seg_folder']

        ner_results = NER_result_parser(anno_id)
        filtered_results = []

        for entity in ner_results:
            entity_name = entity['anatomical_entity']
            morphological_descriptions = entity['morphological_description']
            fdg_uptake_descriptions = entity['fdg_uptake_description']
            original_sentence = entity['original_sentence']
            seg_names = get_seg_name(entity_name)
            if len(seg_names) == 0:
                continue

            valid_seg_paths = []
            for seg_name in seg_names:
                seg_path = os.path.join(CT_seg_folder, f"{seg_name}.nii.gz")
                full_seg_path = os.path.join(PROJECT_ROOT, seg_path)
                if os.path.exists(full_seg_path) or test_mode:
                    valid_seg_paths.append(seg_path)

            if valid_seg_paths:
                ct_seg_path = valid_seg_paths[0] if len(valid_seg_paths) == 1 else valid_seg_paths
                filtered_results.append({
                    'entity_name': entity_name,
                    'original_sentence': original_sentence,
                    'morphological_descriptions': morphological_descriptions,
                    'fdg_uptake_descriptions': fdg_uptake_descriptions,
                    'CT_seg_path': ct_seg_path,
                    'CT_path': CT_path,
                    'PET_reg2CT_path': PET_reg2CT_path
                })

        if not filtered_results:
            print(f"No valid entities found for {anno_id}, skipping.")
            continue
        output_path = os.path.join(output_folder, f"{anno_id}_filtered.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_results, f, ensure_ascii=False, indent=4)
        print(f"Filtered results saved for {anno_id}")


if __name__ == "__main__":
    main_filter(test_mode=False)
