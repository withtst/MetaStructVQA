"""Type 2c -> 2c_maskfree (mask_free) conversion script
Removes segmentation mask paths and replaces question templates with mask-free versions.
"""
import json
import os
import re
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, '../MetaStructVQA_Dataset/QA_pairs/type_2c.json')
OUTPUT_PATH = os.path.join(BASE_DIR, '../MetaStructVQA_Dataset/QA_pairs/type_2c_maskfree.json')
TEMPLATE_PATH = os.path.join(BASE_DIR, 'question_patterns/2c-select_the_correct_description.json')
MASKFREE_TEMPLATE_PATH = os.path.join(BASE_DIR, 'question_patterns/2c-select_the_correct_description_maskfree.json')


def _get_lang_key():
    with open(os.path.join(BASE_DIR, 'configs.yaml'), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f).get('report_language', 'chinese').lower()


def _load_templates(path: str, lang: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item[lang] for item in data]


def _build_regex(template: str) -> re.Pattern:
    pattern = re.escape(template)
    entity_placeholder = re.escape("{entity}")
    desc_placeholder = re.escape("{description}")
    # First occurrence uses named capture group, subsequent uses backreference
    pattern = pattern.replace(entity_placeholder, "<<ENTITY>>", 1)
    pattern = pattern.replace(entity_placeholder, r"(?P=entity)")  # Replace remaining with backreference
    pattern = pattern.replace("<<ENTITY>>", r"(?P<entity>.+?)")  # First replacement as named group
    pattern = pattern.replace(desc_placeholder, "<<DESC>>", 1)
    pattern = pattern.replace(desc_placeholder, r"(?P=description)")
    pattern = pattern.replace("<<DESC>>", r"(?P<description>.+?)")
    return re.compile(rf"^{pattern}$", re.DOTALL)


def _map_content(content: str, maskfree_templates, regexes):
    for idx, regex in enumerate(regexes):
        match = regex.match(content)
        if match:
            groups = match.groupdict()
            return maskfree_templates[idx].format(**groups)
    return None

def main():
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    lang = _get_lang_key()
    templates = _load_templates(TEMPLATE_PATH, lang)
    maskfree_templates = _load_templates(MASKFREE_TEMPLATE_PATH, lang)
    if len(templates) != len(maskfree_templates):
        raise RuntimeError("Template count mismatch: original and mask-free templates do not match")
    regexes = [_build_regex(t) for t in templates]
    maskfree = []
    unmatched = 0
    for qa in qa_pairs:
        if qa.get('q_type') != '2c':
            continue
        new_qa = qa.copy()
        # Rebuild content using mask-free template
        mapped = _map_content(new_qa.get('content', ''), maskfree_templates, regexes)
        if mapped:
            new_qa['content'] = mapped
        else:
            unmatched += 1
        # Remove segmentation mask path
        new_qa.pop('segmentation_path', None)
        # Update q_type
        new_qa['q_type'] = '2c_maskfree'
        maskfree.append(new_qa)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(maskfree, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(maskfree)} 2c_maskfree questions, output: {OUTPUT_PATH}")
    if unmatched:
        print(f"Warning: {unmatched} question(s) did not match any template, original content preserved")

if __name__ == '__main__':
    main()
