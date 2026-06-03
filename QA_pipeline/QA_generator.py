"""
MetaStructVQA Dataset - Core QA Generator
Supports generating 9 question types: 1a, 1b, 2a, 2b, 2c, 2a-enhanced, 2b-enhanced, 2c-enhanced, 3

Randomness in option ordering can be controlled via the RANDOM_SEED constant or --seed argument.
Note: LLM-generated distractors are inherently non-deterministic and cannot be fully reproduced.
"""
import argparse
import csv
import random
import json
import os
import tempfile
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Random seed: set to a fixed value for reproducible option ordering (does not affect LLM-generated distractors)
RANDOM_SEED = 42


def _load_config():
    """Load configs.yaml and return configuration dict"""
    import yaml
    config_path = os.path.join(BASE_DIR, 'configs.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _lang_key():
    """Return question template language key based on config: 'chinese' or 'english'"""
    cfg = _load_config()
    lang = cfg.get('report_language', 'chinese').lower()
    if lang not in ('chinese', 'english'):
        raise ValueError(f"report_language in configs.yaml must be 'chinese' or 'english', got: {lang}")
    return lang


def _report_file():
    """Return report file name based on language configuration"""
    return 'report_zh.csv' if _lang_key() == 'chinese' else 'report_en.csv'


def _project_path(*parts):
    """Build path relative to project root"""
    return os.path.join(PROJECT_ROOT, *parts)


class QAGenerator:
    '''Main QA generator class'''
    def __init__(self, entity_name_mapping_table_path):
        with open(entity_name_mapping_table_path, 'r', encoding='utf-8') as f:
            self.entity_name_mapping_table = json.load(f)
        all_entity_names = set()
        for key in self.entity_name_mapping_table:
            all_entity_names.add(key)
        self.all_entity_names = list(all_entity_names)

    def deepseek_api(self, prompt: str):
        client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "你是一个医学影像分析以及报告撰写专家！"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

    def generate_opposing_description_2ab(self, desc_list: list):
        """Generate distractor descriptions for question types 2a/2b (morphological dimension)"""
        prompt = """角色设定：你是一位资深的PET/CT诊断专家，拥有丰富的临床阅片经验。
你目前正在构建一个高质量的医学VLM（视觉语言模型）训练数据集，目标是通过"硬负采样（Hard Negative Sampling）"策略，生成具有高辨析难度的干扰选项，以测试并提升模型对病灶存在性、形态特征及定量数值的识别能力。
任务描述：给定一组PET/CT报告中的正确描述（Correct Descriptions），请针对每一条描述生成一个对应的干扰项（Distractor）。
生成策略（需平衡使用）：形态特征反转（Qualitative Inversion）： 不仅是简单的"有/无"取反，更侧重于对病灶边缘、密度、强化性质等视觉特征的改写。示例："结节边缘规则"->"结节边缘毛刺状"，注意，只能根据给定线索改写，不能凭空生成，例如"脑结构未见异常"不能生成"脑结构紊乱，灰白质分界不清"，因为灰白质并非原描述中存在。视觉矛盾取反（Strict Negation）：直接否定存在性或改变解剖位置的描述。示例： "骨质破坏伴软组织形成"->"骨质形态异常，未见软组织肿块"，注意，仅需否定描述中最关键的论断即可，不需要全盘否定。临床意义级数值偏差（Clinical-Scale Distortion）： 针对病灶大小或SUV值。修改后的数值必须跨越临床判读的临界点（如从微小结节变为巨大肿块）。示例："长径约0.5cm"->"长径约5.2cm"。
输出要求：一一对应：干扰项列表必须与原始正确描述列表在索引位置上完全匹配。多样性：在整个列表中，均匀分布上述三种策略，避免单一使用"未见"式取反。医学真实性：干扰项虽然是错误的，但在医学术语表达上必须符合诊断报告规范，不能出现语法破碎。
核心约束（必须严格遵守）：
1. 仅输出干扰项列表：不要任何开场白、解释、诊断建议（如"提示...可能"）、括号标注或总结。列表格式为标准python列表，且每个干扰项必须用双引号括起，条目间用逗号分隔。示例输出：["干扰项1","干扰项2","干扰项3",...]
2. 原子化修改：每个干扰项仅针对其对应的正确项进行特征改写。严禁提及或修改正确列表中其他条目所涵盖的解剖部位或特征。
3. 纯形态学描述：禁止输出任何诊断结论。只能描述视觉上的形态、密度、大小、边界或代谢值的异常。
4. 策略应用：平衡使用三种策略，根据每个描述自身特点选择最为合适。
5. 列表对应：输出的干扰项顺序必须与输入列表严格一一对应。
例外情况处理：若给定待处理列表中存在重复项，请确保生成的干扰项在语义上具有区分度。
待处理列表：""" + str(desc_list)
        response = self.deepseek_api(prompt)
        return json.loads(response)

    def generate_opposing_description_2c(self, desc_list: list):
        """Generate distractor descriptions for question type 2c (FDG metabolism dimension)"""
        prompt = """Role: 你是一位资深的核医学科医师，擅长编写 PET/CT 放射报告的多选题干扰项。
Task: 给定一条关于某个器官或病灶的"正确描述"，请根据"保持形态学描述不变，篡改 FDG 代谢描述"的原则，生成 3 个具有迷惑性的干扰项（错误答案）。
生成规则:
1.代谢状态对冲：如果原句是"代谢增高"，干扰项应改为"代谢缺如"、"代谢降低"、"代谢未见明显异常"等。
2.定量数值偏差：如果原句包含 SUVmax 值，生成两个结论相同但数值具有显著差异的选项。
3.词汇精准替换：灵活使用FDG代谢缺如、摄取缺失、代谢轻度增高、未见明显异常等术语。
文风保持一致：干扰项的句式结构必须与原句高度一致。
！极度重要：输入列表的原句索引和输出列表的错误答案子列表索引保持高度一致！
Input Format: ["正确描述文本1","正确文本2",...]
Output Format: [["错误答案1", "错误答案2", "错误答案3"],["...","...","..."],...]
下面请处理如下句子:""" + str(desc_list)
        response = self.deepseek_api(prompt)
        return json.loads(response)

    def gen_qa_type_2ab(self, annotations, question_pattern_a, question_pattern_b, next_q_id=None):
        """Batch generate type 2a (multi-select 4-choose-2) and 2b (true/false 2-choose-1) questions"""
        lang = _lang_key()
        all_desc_list = []
        for item in annotations:
            if item["morphological_descriptions"] == "None":
                continue
            for desc in item["morphological_descriptions"]:
                all_desc_list.append(desc)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                all_opdesc_list = self.generate_opposing_description_2ab(all_desc_list)
                if len(all_opdesc_list) != len(all_desc_list):
                    if attempt < max_retries - 1:
                        print(f"Warning: attempt {attempt + 1}, count mismatch, retrying...")
                        continue
                    else:
                        raise ValueError(f"Distractor description count mismatch")
                break
            except (json.JSONDecodeError, SyntaxError) as e:
                if attempt < max_retries - 1:
                    continue
                else:
                    raise

        CT_path = annotations[0]["CT_path"]
        generated_qas = []
        for item in annotations:
            if item["morphological_descriptions"] == "None":
                continue
            desc_list = [d for d in item["morphological_descriptions"] if d != "None"]
            desc_num = len(desc_list)
            entity_name = item["entity_name"]
            segmentation_path = item["CT_seg_path"]
            if len(all_opdesc_list) < desc_num:
                raise IndexError(f"Not enough distractor descriptions remaining")
            opdesc_list = all_opdesc_list[:desc_num]
            all_opdesc_list = all_opdesc_list[desc_num:]

            def _make_2a(d1, d2, od1, od2):
                keys = ["A", "B", "C", "D"]
                r1 = random.choice(keys); keys.remove(r1)
                r2 = random.choice(keys); keys.remove(r2)
                w1 = random.choice(keys); w2 = keys[0]
                opts = {"A": "", "B": "", "C": "", "D": ""}
                opts[r1], opts[r2], opts[w1], opts[w2] = d1, d2, od1, od2
                qa = {"_entity_name": entity_name, "q_type": "2a",
                      "content": random.choice(question_pattern_a)[lang].format(entity=entity_name),
                      "options": opts, "answer": r1 + r2,
                      "CT_path": CT_path, "segmentation_path": segmentation_path}
                if next_q_id is not None: qa["q_id"] = next_q_id()
                return qa

            if desc_num % 2 == 0:
                for i in range(0, desc_num, 2):
                    generated_qas.append(_make_2a(desc_list[i], desc_list[i+1], opdesc_list[i], opdesc_list[i+1]))
            else:
                for i in range(0, desc_num - 1, 2):
                    generated_qas.append(_make_2a(desc_list[i], desc_list[i+1], opdesc_list[i], opdesc_list[i+1]))
                # True/false: correct description
                qa_true = {"_entity_name": entity_name, "q_type": "2b",
                           "content": random.choice(question_pattern_b)[lang].format(entity=entity_name, description=desc_list[-1]),
                           "options": ["A", "B"], "answer": "A",
                           "CT_path": CT_path, "segmentation_path": segmentation_path}
                if next_q_id is not None: qa_true["q_id"] = next_q_id()
                generated_qas.append(qa_true)
                # True/false: incorrect description
                qa_false = {"_entity_name": entity_name, "q_type": "2b",
                            "content": random.choice(question_pattern_b)[lang].format(entity=entity_name, description=opdesc_list[-1]),
                            "options": ["A", "B"], "answer": "B",
                            "CT_path": CT_path, "segmentation_path": segmentation_path}
                if next_q_id is not None: qa_false["q_id"] = next_q_id()
                generated_qas.append(qa_false)
        return generated_qas

    def gen_qa_type_2c(self, annotations, question_pattern_c, next_q_id=None):
        """Batch generate type 2c (FDG metabolism single-choice 4-choose-1) questions"""
        lang = _lang_key()
        all_desc_list = []
        for item in annotations:
            if item["fdg_uptake_descriptions"] == "None":
                continue
            for desc in item["fdg_uptake_descriptions"]:
                all_desc_list.append(desc)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                all_opdesc_list = self.generate_opposing_description_2c(all_desc_list)
                if not all(isinstance(s, list) for s in all_opdesc_list):
                    if attempt < max_retries - 1: continue
                    else: raise ValueError(f"Invalid format")
                if len(all_opdesc_list) != len(all_desc_list):
                    if attempt < max_retries - 1: continue
                    else: raise ValueError(f"Count mismatch")
                if not all(len(s) == 3 for s in all_opdesc_list):
                    if attempt < max_retries - 1: continue
                    else: raise ValueError(f"Sub-list length incorrect")
                break
            except (json.JSONDecodeError, SyntaxError):
                if attempt < max_retries - 1: continue
                else: raise

        CT_path = annotations[0]["CT_path"]
        PET_path = annotations[0]["PET_reg2CT_path"]
        generated_qas = []
        for item in annotations:
            if item["fdg_uptake_descriptions"] == "None":
                continue
            desc_list = [d for d in item["fdg_uptake_descriptions"] if d != "None"]
            desc_num = len(desc_list)
            entity_name = item["entity_name"]
            segmentation_path = item["CT_seg_path"]
            if len(all_opdesc_list) < desc_num:
                raise IndexError(f"Not enough distractor descriptions remaining")
            opdesc_list = all_opdesc_list[:desc_num]
            all_opdesc_list = all_opdesc_list[desc_num:]
            for i in range(desc_num):
                keys = ["A", "B", "C", "D"]
                right = random.choice(keys); keys.remove(right)
                w1 = random.choice(keys); keys.remove(w1)
                w2 = random.choice(keys); w3 = keys[0]
                opts = {"A": "", "B": "", "C": "", "D": ""}
                opts[right], opts[w1], opts[w2], opts[w3] = desc_list[i], opdesc_list[i][0], opdesc_list[i][1], opdesc_list[i][2]
                qa = {"_entity_name": entity_name, "q_type": "2c",
                      "content": random.choice(question_pattern_c)[lang].format(entity=entity_name),
                      "options": opts, "answer": right,
                      "CT_path": CT_path, "PET_path": PET_path, "segmentation_path": segmentation_path}
                if next_q_id is not None: qa["q_id"] = next_q_id()
                generated_qas.append(qa)
        return generated_qas


# ==================== Helper Functions ====================

def get_last_question_id(id_record_path):
    if not os.path.exists(id_record_path):
        return -1
    try:
        with open(id_record_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        if not records:
            return -1
        return int(records[-1]["q_id"])
    except Exception as e:
        print(f"Error reading ID record file: {e}")
        return -1

def _load_id_records(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        return []
    records = json.loads(content)
    if not isinstance(records, list):
        raise RuntimeError(f"ID record file format error: {path}")
    return records

def _atomic_write_json(path, data):
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=dir_name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
            json.dump(data, tmp_file, ensure_ascii=False, indent=2)
        last_err = None
        for attempt in range(5):
            try:
                os.replace(tmp_path, path)
                break
            except PermissionError as e:
                last_err = e
                time.sleep(0.2 * (attempt + 1))
        if last_err is not None:
            raise last_err
    finally:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass

def generate_next_question_id(last_id):
    return f"{last_id + 1:06d}"

def get_img_id_from_ct_path(ct_path):
    if not ct_path: return ""
    base_name = os.path.basename(ct_path.replace("/", "\\"))
    return base_name.split("_")[0] if "_" in base_name else ""

def get_anno_id_from_filename(filename):
    name = os.path.splitext(filename)[0]
    if name.endswith("_filtered"):
        name = name[:-len("_filtered")]
    return name

def register_question_id(id_record_path, q_id, entity_name, anno_id, img_id):
    records = _load_id_records(id_record_path)
    records.append({"q_id": q_id, "anno_id": anno_id, "img_id": img_id, "entity_name": entity_name})
    _atomic_write_json(id_record_path, records)


def _load_anno_to_ptxh():
    """Load anno_id -> Image ID (PTXH) mapping table"""
    mapping = {}
    with open(_project_path("anno_mapping_table.csv"), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ptxh_id = row.get('Image ID', '').strip()
            anno_id = row.get('anno_id', '').strip()
            if anno_id and ptxh_id:
                mapping[anno_id] = ptxh_id
    return mapping


def _load_qa_pairs(path):
    """Safely load existing QA pairs; returns empty list if file not found or parse fails"""
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.loads(f.read().strip()) or []
        except json.JSONDecodeError:
            return []


# ==================== Question Type Generation Entry Points ====================

def gen_type_1a():
    """Generate type 1a: organ recognition questions"""
    id_record_path = _project_path("MetaStructVQA_Dataset", "1a_ids.json")
    last_id = get_last_question_id(id_record_path)

    with open(os.path.join(BASE_DIR, "question_patterns", "1a-entity_name_recognition.json"), 'r', encoding='utf-8') as f:
        question_patterns = json.load(f)
    with open(os.path.join(BASE_DIR, "1a_organs_groups.json"), 'r', encoding='utf-8') as f:
        organ_groups = json.load(f)
    with open(_project_path("entities_nonsub_mapping_table.json"), 'r', encoding='utf-8') as f:
        entity_name_mapping_table = json.load(f)

    anno_to_ptxh = _load_anno_to_ptxh()

    qa_pairs_path = _project_path("MetaStructVQA_Dataset", "QA_pairs", "type_1a.json")
    qa_pairs = _load_qa_pairs(qa_pairs_path)

    import threading
    id_lock = threading.Lock()
    current_id_num = [last_id]

    def _generate_qa_for_anno(anno_id, ptxh_id):
        anno_qa_batch, id_records = [], []
        ct_path = f"imgs_data/CT/{ptxh_id}_0000.nii.gz"
        for entity_name, wrong_pool in organ_groups.items():
            if not isinstance(wrong_pool, list) or len(wrong_pool) < 3:
                raise ValueError(f"Organ group table error: {entity_name}")
            with id_lock:
                current_id_num[0] += 1
                q_id = f"{current_id_num[0]:06d}"
            question_pattern = random.choice(question_patterns)[_lang_key()]
            wrong_options = random.sample(wrong_pool, 3)
            english_name = entity_name_mapping_table.get(entity_name, "")
            seg_path = f"imgs_data/CT_seg/{ptxh_id}/{english_name}.nii.gz" if english_name else ""
            qa = {"q_id": q_id, "q_type": "1a", "content": question_pattern,
                  "options": {"A": "", "B": "", "C": "", "D": ""}, "answer": "",
                  "CT_path": ct_path, "segmentation_path": seg_path}
            opt_list = ["A", "B", "C", "D"]
            ans = random.choice(opt_list); opt_list.remove(ans)
            qa["answer"] = ans
            qa["options"][ans] = entity_name
            for i, k in enumerate(opt_list):
                qa["options"][k] = wrong_options[i]
            id_records.append({"q_id": q_id, "entity_name": entity_name, "anno_id": anno_id, "img_id": ptxh_id})
            anno_qa_batch.append(qa)
        return anno_qa_batch, id_records

    print("=" * 80)
    print("Starting concurrent generation of type 1a questions")
    results_by_anno, all_id_records = {}, []
    with ThreadPoolExecutor(max_workers=25) as executor:
        future_map = {executor.submit(_generate_qa_for_anno, aid, pid): aid for aid, pid in anno_to_ptxh.items()}
        for i, future in enumerate(as_completed(future_map), 1):
            anno_id = future_map[future]
            batch, records = future.result()
            results_by_anno[anno_id] = batch
            all_id_records.extend(records)
            print(f"  [{i}/{len(anno_to_ptxh)}] {anno_id}: {len(batch)} questions")

    for anno_id in sorted(anno_to_ptxh.keys()):
        if anno_id in results_by_anno:
            qa_pairs.extend(results_by_anno[anno_id])
    for record in all_id_records:
        register_question_id(id_record_path, record["q_id"], record["entity_name"], record["anno_id"], record["img_id"])
    with open(qa_pairs_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"\nGenerated {current_id_num[0] - last_id} new questions")


def gen_type_1b():
    """Generate type 1b: modality recognition questions"""
    lang = _lang_key()
    last_id = get_last_question_id(_project_path("MetaStructVQA_Dataset", "1b_ids.json"))
    with open(os.path.join(BASE_DIR, "question_patterns", "1b-modality_recognition.json"), 'r', encoding='utf-8') as f:
        question_patterns = json.load(f)

    anno_to_ptxh = _load_anno_to_ptxh()

    qa_pairs_path = _project_path("MetaStructVQA_Dataset", "QA_pairs", "type_1b.json")
    qa_pairs = []
    for anno_id, ptxh_id in anno_to_ptxh.items():
        ct_path = f"imgs_data/CT/{ptxh_id}_0000.nii.gz"
        pet_path = f"imgs_data/PET/{ptxh_id}_0000.nii.gz"
        qa_ct = {"q_id": None, "q_type": "1b", "content": random.choice(question_patterns)[lang],
                 "options": {"A": "CT", "B": "PET"}, "answer": "A", "img_path": ct_path, "_anno_id": anno_id}
        qa_pairs.append(qa_ct)
        qa_pet = {"q_id": None, "q_type": "1b", "content": random.choice(question_patterns)[lang],
                  "options": {"A": "CT", "B": "PET"}, "answer": "B", "img_path": pet_path, "_anno_id": anno_id}
        qa_pairs.append(qa_pet)

    current_id_num = last_id
    for qa in qa_pairs:
        current_id_num += 1
        qa["q_id"] = f"{current_id_num:06d}"
        register_question_id(_project_path("MetaStructVQA_Dataset", "1b_ids.json"),
                             qa["q_id"], None, qa.pop("_anno_id"), None)
    with open(qa_pairs_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"\nType 1b generation complete: {len(qa_pairs)} questions total")


def gen_type2ab(test_mode=False):
    """Generate type 2a/2b: morphological multi-select / true-false questions"""
    qa_gen = QAGenerator(_project_path("entities_nonsub_mapping_table.json"))
    id_record_path = _project_path("MetaStructVQA_Dataset", "all_question_ids.json")
    progress_path = _project_path("MetaStructVQA_Dataset", "type2_progress.json")
    last_id = get_last_question_id(id_record_path) if not test_mode else -1
    current_id_num = last_id

    with open(os.path.join(BASE_DIR, "question_patterns", "2a-select_the_correct_description.json"), 'r', encoding='utf-8') as f:
        qp_2a = json.load(f)
    with open(os.path.join(BASE_DIR, "question_patterns", "2b-identify_true_or_false.json"), 'r', encoding='utf-8') as f:
        qp_2b = json.load(f)

    processed_files = set()
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                processed_files = set(json.load(f).get("processed_files", []))
        except (json.JSONDecodeError, OSError):
            pass

    anno_dir = _project_path("MetaStructVQA_Dataset", "annotations")
    files = sorted(os.listdir(anno_dir))
    if test_mode: files = files[:1]
    files_to_process = [f for f in files if f not in processed_files]

    qa_pairs_path = _project_path("MetaStructVQA_Dataset", "QA_pairs", "type_2.json")
    qa_pairs = _load_qa_pairs(qa_pairs_path)

    for file_name in files_to_process:
        print(f"\nProcessing: {file_name}")
        with open(os.path.join(anno_dir, file_name), 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        qa_batch = qa_gen.gen_qa_type_2ab(annotations, qp_2a, qp_2b)
        anno_id = get_anno_id_from_filename(file_name)
        for qa in qa_batch:
            current_id_num += 1
            qa["q_id"] = f"{current_id_num:06d}"
            register_question_id(id_record_path, qa["q_id"], qa.pop("_entity_name", ""), anno_id,
                                 get_img_id_from_ct_path(qa.get("CT_path", "")))
        qa_pairs.extend(qa_batch)
        with open(qa_pairs_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        processed_files.add(file_name)
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump({"processed_files": list(processed_files)}, f, ensure_ascii=False, indent=2)
        print(f"Written {len(qa_batch)} questions")


def gen_type2c(test_mode=False):
    """Generate type 2c: FDG metabolism single-choice questions"""
    qa_gen = QAGenerator(_project_path("entities_nonsub_mapping_table.json"))
    id_record_path = _project_path("MetaStructVQA_Dataset", "all_question_ids.json")
    progress_path = _project_path("MetaStructVQA_Dataset", "type2c_progress.json")
    qa_pairs_path = _project_path("MetaStructVQA_Dataset", "QA_pairs", "type_2c.json")
    last_id = get_last_question_id(id_record_path) if not test_mode else -1
    current_id_num = last_id

    with open(os.path.join(BASE_DIR, "question_patterns", "2c-select_the_correct_description.json"), 'r', encoding='utf-8') as f:
        qp_2c = json.load(f)

    if test_mode:
        print("TEST MODE: validation only, no file writes")
        anno_dir = _project_path("MetaStructVQA_Dataset", "annotations")
        files = sorted(os.listdir(anno_dir))[:1]
        for fn in files:
            with open(os.path.join(anno_dir, fn), 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            qa_batch = qa_gen.gen_qa_type_2c(annotations, qp_2c)
            print(f"Successfully generated {len(qa_batch)} questions")
        return

    print("PRODUCTION MODE: concurrent processing")
    processed_files = set()
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                processed_files = set(json.load(f).get("processed_files", []))
        except (json.JSONDecodeError, OSError):
            pass

    anno_dir = _project_path("MetaStructVQA_Dataset", "annotations")
    files = sorted(os.listdir(anno_dir))
    files_to_process = [f for f in files if f not in processed_files]

    qa_pairs = _load_qa_pairs(qa_pairs_path)

    def _process(file_name):
        with open(os.path.join(anno_dir, file_name), 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        return file_name, qa_gen.gen_qa_type_2c(annotations, qp_2c)

    results = {}
    with ThreadPoolExecutor(max_workers=30) as executor:
        futs = {executor.submit(_process, fn): fn for fn in files_to_process}
        for i, fut in enumerate(as_completed(futs), 1):
            fn, batch = fut.result()
            results[fn] = batch
            print(f"  [{i}/{len(files_to_process)}] {fn}: {len(batch)} questions")

    for fn in files_to_process:
        batch = results.get(fn, [])
        anno_id = get_anno_id_from_filename(fn)
        for qa in batch:
            current_id_num += 1
            qa["q_id"] = f"{current_id_num:06d}"
            register_question_id(id_record_path, qa["q_id"], qa.pop("_entity_name", ""), anno_id,
                                 get_img_id_from_ct_path(qa.get("CT_path", "")))
        qa_pairs.extend(batch)
        with open(qa_pairs_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        processed_files.add(fn)
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump({"processed_files": list(processed_files)}, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(qa_pairs)} type 2c questions total")


def gen_type_3(test_mode=0):
    """Generate type 3: report-level comprehensive questions"""
    lang = _lang_key()
    qa_gen = QAGenerator(_project_path("entities_nonsub_mapping_table.json"))
    id_record_path = _project_path("MetaStructVQA_Dataset", "type3_ids.json")
    progress_path = _project_path("MetaStructVQA_Dataset", "type3_progress.json")
    qa_pairs_path = _project_path("MetaStructVQA_Dataset", "QA_pairs", "type_3.json")
    reports_path = _project_path("reports_prepare", "origin_reports", _report_file())
    last_id = get_last_question_id(id_record_path) if not test_mode else -1
    current_id_num = last_id

    _PROMPTS = {
        "chinese": '''你是一位资深的核医学影像专家与 AI 数据科学家，擅长将复杂的 PET/CT 诊断报告拆解为可供计算机视觉模型学习的细粒度视觉问答（VQA）对。
请阅读提供的 [PET/CT 报告全文]，按照以下维度生成 Q&A 选择题数据：属性识别、定量测量、空间关系、比较逻辑、阴性识别。
输出 JSON 格式：[{"type": "...", "question": "...", "answer": "...", "wrong_answer": ["...","...","..."], "rationale": "..."}]
禁止臆造，仅根据报告内容生成。正负样本平衡。
# PET/CT 报告全文：
''',
        "english": '''You are a senior nuclear medicine imaging expert and AI data scientist, skilled in decomposing complex PET/CT diagnostic reports into fine-grained visual question answering (VQA) pairs for computer vision model training.
Please read the provided [PET/CT Report] and generate Q&A multiple-choice questions across the following dimensions: attribute recognition, quantitative measurement, spatial relationships, comparative reasoning, and negative finding identification.
Output JSON format: [{"type": "...", "question": "...", "answer": "...", "wrong_answer": ["...","...","..."], "rationale": "..."}]
Do not fabricate information; generate only based on the report content. Balance positive and negative samples.
# PET/CT Report:
'''
    }
    prompt = _PROMPTS[lang]
    report_encoding = 'utf-8' if lang == 'chinese' else 'latin-1'

    def _call_api_with_retry(report_text, max_retries=3):
        for attempt in range(max_retries):
            try:
                raw = qa_gen.deepseek_api(prompt + report_text)
                batch = raw if isinstance(raw, list) else json.loads(raw)
                if not isinstance(batch, list):
                    raise ValueError("Invalid response format")
                return batch
            except Exception as e:
                if attempt < max_retries - 1: continue
                raise

    def _format_item(raw_item, ptxh_id):
        keys = ["A", "B", "C", "D"]
        right = random.choice(keys); keys.remove(right)
        w1 = random.choice(keys); keys.remove(w1)
        w2 = random.choice(keys); w3 = keys[0]
        opts = {"A": "", "B": "", "C": "", "D": ""}
        opts[right] = raw_item["answer"]
        opts[w1], opts[w2], opts[w3] = raw_item["wrong_answer"][0], raw_item["wrong_answer"][1], raw_item["wrong_answer"][2]
        ct_path = f"imgs_data/CT/{ptxh_id}_0000.nii.gz" if ptxh_id else ""
        pet_path = f"imgs_data/PET/{ptxh_id}_0000.nii.gz" if ptxh_id else ""
        return {"q_type": "3", "content": raw_item["question"], "options": opts, "answer": right,
                "CT_path": ct_path, "PET_path": pet_path}

    with open(reports_path, 'r', encoding=report_encoding) as f:
        reader = csv.reader(f)
        reports_tuples = [(row[0], row[1]) for row in reader][1:]

    anno_to_ptxh = _load_anno_to_ptxh()

    processed_reports = set()
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                processed_reports = set(json.load(f).get("processed_reports", []))
        except (json.JSONDecodeError, OSError):
            pass

    reports_to_process = [r for r in reports_tuples if r[0] not in processed_reports]
    if test_mode and int(test_mode) > 0:
        reports_to_process = reports_to_process[:int(test_mode)]

    qa_pairs = _load_qa_pairs(qa_pairs_path)

    def _process_report(anno_id, report_text):
        return anno_id, _call_api_with_retry(report_text)

    results = {}
    print(f"Starting concurrent processing of {len(reports_to_process)} report(s)...")
    with ThreadPoolExecutor(max_workers=30) as executor:
        futs = {executor.submit(_process_report, aid, rt): aid for aid, rt in reports_to_process}
        for i, fut in enumerate(as_completed(futs), 1):
            aid, batch = fut.result()
            results[aid] = batch
            print(f"  [{i}/{len(reports_to_process)}] {aid}: {len(batch)} questions")

    for anno_id, _ in reports_to_process:
        batch = results.get(anno_id, [])
        ptxh_id = anno_to_ptxh.get(anno_id, "")
        formatted = []
        for raw_item in batch:
            item = _format_item(raw_item, ptxh_id)
            current_id_num += 1
            item["q_id"] = f"{current_id_num:06d}"
            register_question_id(id_record_path, item["q_id"], "*", anno_id, ptxh_id)
            formatted.append(item)
        qa_pairs.extend(formatted)
        with open(qa_pairs_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        processed_reports.add(anno_id)
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump({"processed_reports": list(processed_reports)}, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(qa_pairs)} type 3 questions total")


def main():
    parser = argparse.ArgumentParser(description="MetaStructVQA question generator")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed (default: 42)")
    parser.add_argument("--types", nargs="+", default=[],
                        choices=["1a", "1b", "2ab", "2c", "3"],
                        help="Question types to generate (multi-select; none generated if not specified)")
    parser.add_argument("--test_mode", action="store_true", help="Test mode (process only a few samples)")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Random seed: {args.seed}")

    if "1a" in args.types:
        gen_type_1a()
    if "1b" in args.types:
        gen_type_1b()
    if "2ab" in args.types:
        gen_type2ab(test_mode=args.test_mode)
    if "2c" in args.types:
        gen_type2c(test_mode=args.test_mode)
    if "3" in args.types:
        gen_type_3(int(args.test_mode) if args.test_mode else 0)

    if not args.types:
        print("No question type specified. Use --types 1a 1b 2ab 2c 3 to specify types to generate.")


if __name__ == "__main__":
    main()
