# MetaStructVQA Dataset Pipeline

Automated PET-CT Visual Question Answering dataset construction pipeline for medical Vision-Language Model (VLM) training.

## Overview

This repository contains a 7-stage fully automated pipeline that starts from raw PET-CT imaging reports and goes through NER recognition, data fusion, entity filtering, QA generation, and more to produce a structured multiple-choice VQA dataset.

- **Data Scale**: 490 subjects, 100,000+ questions, 103 standard anatomical structures
- **LLM**: DeepSeek-R1 (deepseek-reasoner, via OpenAI SDK)
- **Segmentation Tool**: TotalSegmentator

## Directory Structure

```
repo/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── run_pipeline.py                       # Full pipeline automation script (not recommended, see below)
├── anno_mapping_table.csv                # anno_id ↔ Image ID mapping table
├── entities_nonsub_mapping_table.json    # Chinese entity name → TotalSegmentator English name mapping
│
├── deepseekNER/                          # Stages 1-3: NER recognition module
│   ├── deepseek_api.py                   # Single-threaded NER processing
│   ├── deepseek_api_concurrent.py        # Concurrent NER processing
│   ├── clean_NER_format.py               # NER result format check & fix
│   ├── split_entities.py                 # Compound entity splitting tool
│   ├── find_entity_location.py           # Entity location search tool
│   ├── ner_prompt.txt                       # NER extraction prompt
│   └── anatomy_hierarchy_reference.txt      # Anatomical entity hierarchy reference (for NER constraints)
│
├── merge_seg_and_reports/                # Stage 4: Data fusion module
│   ├── main_merger.py                    # Integrate images/masks/NER/reports into unified metadata
│   └── anno_converter.py                 # anno_id ↔ Image ID conversion
│
├── QA_pipeline/                          # Stages 5-7: QA generation pipeline
│   ├── QA_generator.py                   # Core QA generator (all question types)
│   ├── entities_filter.py                # Entity filtering (match segmentation masks)
│   ├── filter_absent_organs.py           # Absent organ filtering
│   ├── split_type2.py                    # Type 2 split (2a/2b)
│   ├── convert_type_2a_to_mf.py          # 2a → 2a-enhanced (mask-free) conversion
│   ├── convert_type_2b_to_mf.py          # 2b → 2b-enhanced (mask-free) conversion
│   ├── convert_type_2c_to_mf.py          # 2c → 2c-enhanced (mask-free) conversion
│   ├── delete_entity_questions.py        # Interactive question deletion tool
│   ├── configs.yaml                      # Pipeline configuration file
│   ├── 1a_organs_groups.json             # Type 1a organ distractor group table
│   └── question_patterns/                # Question templates (bilingual CN/EN)
│       ├── 1a-entity_name_recognition.json
│       ├── 1b-modality_recognition.json
│       ├── 2a-select_the_correct_description.json
│       ├── 2b-identify_true_or_false.json
│       ├── 2c-select_the_correct_description.json
│       └── *-enhanced.json              # Mask-free version templates
│
└── examples/                             # Examples for each question type (one per type, 9 total)
    │                                         # Content language follows report_language in configs.yaml
    ├── type_1a_example.json
    ├── type_1b_example.json
    ├── type_2a_example.json
    ├── type_2a-enhanced_example.json    # 2a-enhanced (mask-free version)
    ├── type_2b_example.json
    ├── type_2b-enhanced_example.json    # 2b-enhanced (mask-free version)
    ├── type_2c_example.json
    ├── type_2c-enhanced_example.json    # 2c-enhanced (mask-free version)
    └── type_3_example.json
```

## Prerequisites & Data Requirements

This repository does not include any raw data. The following external resources must be prepared before running the pipeline. All paths can be configured via command-line arguments or `configs.yaml`.

### 1. DeepSeek API Key

This project uses the DeepSeek-R1 (`deepseek-reasoner`) model for NER recognition and distractor generation. Please apply for an API key at [DeepSeek Platform](https://platform.deepseek.com/) and set the environment variable:

```bash
# Linux / macOS
export DEEPSEEK_API_KEY="your-api-key-here"

# Windows PowerShell
$env:DEEPSEEK_API_KEY = "your-api-key-here"
```

> **Note**: Generating the full dataset (490 subjects × 9 question types) will consume significant API quota. Please ensure sufficient account balance.

### 2. Original Report Data

Both Chinese and English reports are supported, switchable via `report_language` in `configs.yaml` (default: `chinese`). Reports are associated with subjects in `anno_mapping_table.csv` via the Image ID column.

**Chinese reports (`report_zh.csv`)**, encoded as `utf-8`:

| Column | Type | Description |
|------|------|------|
| Image Number | str | Subject anno_id, e.g., `sub_001` |
| Image Description | str | Full report findings section |
| Report Diagnosis | str | Report diagnosis/conclusion section |

```csv
Image Number,Image Description,Report Diagnosis
sub_001,Brain morphology and structure are normal...,1. Multiple intrahepatic...
sub_002,...,...
```

**English reports (`report_en.csv`)**, encoded as `latin-1`:

| Column | Type | Description |
|------|------|------|
| `Image ID` | str | Subject anno_id, e.g., `sub_001` |
| `Image Description` | str | English report findings section |
| `Report Diagnosis` | str | English report diagnosis/conclusion |

```csv
Image ID,Image Description,Report Diagnosis
sub_001,After intravenous...,1.a. Multiple intrahepatic...
sub_002,...,...
```

**Report text list (`report_text_list.txt`)**:

A separate list of report texts needed for the NER stage, formatted as a JSON string array. Element order corresponds to report CSV row order:

```json
["Brain morphology and structure are normal, small patchy low-density shadows in deep brain...", "Second report text...", ...]
```

> **Language switching**: Set `report_language: "english"` in `configs.yaml` to switch to English reports + English question templates + English prompts.

### 3. PET-CT Image Data

Both CT and PET images are in NIfTI format (`.nii.gz`), with naming convention `{Image_ID}_0000.nii.gz`, where `Image_ID` comes from the first column of `anno_mapping_table.csv`.

```
imgs_data/
├── CT/
│   ├── PTXH220103024_0000.nii.gz
│   ├── PTXH220104042_0000.nii.gz
│   └── ...
└── PET/                          # PET registered to CT space
    ├── PTXH220103024_0000.nii.gz
    ├── PTXH220104042_0000.nii.gz
    └── ...
```

### 4. TotalSegmentator Segmentation Masks

CT images are segmented using TotalSegmentator, with one subdirectory per subject containing individual mask files for each organ. Mask file names must use TotalSegmentator standard English anatomical names (e.g., `brain.nii.gz`, `liver.nii.gz`).

```
imgs_data/
└── CT_seg/
    ├── PTXH220103024/
    │   ├── brain.nii.gz
    │   ├── liver.nii.gz
    │   ├── lung_upper_lobe_left.nii.gz
    │   └── ...                   # One file per organ
    ├── PTXH220104042/
    │   └── ...
    └── ...
```

> See `entities_nonsub_mapping_table.json` in the repository for the mapping between English and Chinese organ names:
> ```json
> {"Left Adrenal Gland": "adrenal_gland_left", "Brain": "brain", "Liver": "liver", ...}
> ```

### 5. anno_mapping_table.csv

Subject ID mapping table, included in the repository:

| Column | Description |
|------|------|
| `Image ID` | Original image ID, e.g., `PTXH220103024` |
| `anno_id` | Subject number, e.g., `sub_001` |

```csv
Image ID,anno_id
PTXH220103024,sub_001
PTXH220104042,sub_002
```

### 6. Recommended Directory Layout

Place the above data alongside the repository root:

```
Project Root/
├── repo/                            # This repository
├── imgs_data/
│   ├── CT/                          # CT NIfTI images
│   ├── PET/                         # PET (registered to CT) NIfTI images
│   └── CT_seg/                      # TotalSegmentator segmentation masks
├── reports_prepare/
│   └── origin_reports/
│       ├── report_zh.csv            # Chinese reports
│       └── report_en.csv            # English reports
└── ...
```

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Refer to section 1 above ("Prerequisites") to set `DEEPSEEK_API_KEY`.

## Usage

### Stages 1-3: NER Recognition (deepseekNER/)

Use the DeepSeek API to perform named entity recognition on PET-CT reports, extracting anatomical entities with their morphological and FDG metabolism descriptions.

```bash
# Concurrent batch processing
cd deepseekNER/
python deepseek_api_concurrent.py \
    --reports_csv ../reports_prepare/origin_reports/report_zh.csv \
    --prompt_file ner_prompt.txt \
    --hierarchy_file anatomy_hierarchy_reference.txt \
    --reports_list_file report_text_list.txt \
    --output_dir descriptions/ \
    --start 0 --end 490 --max_workers 10

# Check and fix NER result format
python clean_NER_format.py

# (Optional) Split compound entities, e.g., "renal pelvis and calyx" -> "renal pelvis" + "renal calyx"
python find_entity_location.py "renal pelvis and calyx"   # Search for target entity locations
# Paste results into TARGET_FILES variable in split_entities.py, then run:
python split_entities.py run_all
```

### Stage 4: Data Fusion (merge_seg_and_reports/)

Integrate CT/PET images, TotalSegmentator segmentation masks, NER results, and bilingual reports into unified metadata files.

```bash
cd merge_seg_and_reports/
python main_merger.py \
    --ct_folder ../imgs_data/CT \
    --pet_folder ../imgs_data/PET \
    --ct_seg_folder ../imgs_data/CT_seg \
    --ct_reg2pet_seg_folder ../imgs_data/CT_reg2PET_seg \
    --reports_en ../reports_prepare/origin_reports/report_en.csv \
    --reports_zh ../reports_prepare/origin_reports/report_zh.csv \
    --ner_folder ../deepseekNER/descriptions \
    --start 1 --end 491
```

Output: `VQA_pre_version/sub_xxx/metadata.json`

### Stage 5: Entity Filtering (QA_pipeline/)

Filter NER results based on segmentation mask existence, selecting only entities with corresponding mask files.

```bash
cd QA_pipeline/
python entities_filter.py
```

Edit `configs.yaml` to configure input/output paths before running.

### Stage 6: QA Generation (QA_pipeline/)

Generate multiple-choice questions for each question type:

```bash
cd QA_pipeline/

# Type 1a: Organ recognition (4-choose-1, static grouping based on anatomical homology)
# Type 1b: Modality recognition (2-choose-1, CT vs PET)
# Type 2a/2b: Morphological multi-select / true-false (DeepSeek API dynamic distractor generation)
# Type 2c: FDG metabolism single-choice (DeepSeek API dynamic distractor generation)
# Type 3: Report-level comprehensive questions

# Generate all question types (default random seed 42)
python QA_generator.py --types 1a 1b 2ab 2c 3

# Custom random seed (for reproducible option ordering)
python QA_generator.py --types 1a 1b --seed 123

# Test mode (process only a few samples)
python QA_generator.py --types 2ab 2c --test_mode
```

### Stage 7: Post-processing

```bash
# Split type 2 into separate 2a and 2b files
python split_type2.py

# Generate mask-free versions (enhanced)
python convert_type_2a_to_mf.py
python convert_type_2b_to_mf.py
python convert_type_2c_to_mf.py

# Filter questions for absent organs
python filter_absent_organs.py
```

### Full Pipeline Automation (Not Recommended)

> **⚠️ Warning**: The one-click script below is provided for debugging convenience only. **In production, it is strongly recommended to run each stage manually.**
> The full pipeline script cannot inspect data quality at intermediate steps, handle API exceptions, or resume from breakpoints, which may lead to hard-to-diagnose batch errors.

```bash
# Run from repository root
python run_pipeline.py \
    --reports_zh ../reports_prepare/origin_reports/report_zh.csv \
    --reports_en ../reports_prepare/origin_reports/report_en.csv \
    --reports_list ../deepseekNER/report_text_list.txt \
    --ct_folder ../imgs_data/CT \
    --pet_folder ../imgs_data/PET \
    --ct_seg_folder ../imgs_data/CT_seg \
    --ct_reg2pet_seg_folder ../imgs_data/CT_reg2PET_seg

# Skip completed stages
python run_pipeline.py \
    --reports_zh ... --reports_en ... --reports_list ... \
    --ct_folder ... --pet_folder ... --ct_seg_folder ... --ct_reg2pet_seg_folder ... \
    --skip_ner --skip_merge

# Run post-processing only (skip NER, fusion, QA generation)
python run_pipeline.py \
    --reports_zh ... --reports_en ... --reports_list ... \
    --ct_folder ... --pet_folder ... --ct_seg_folder ... --ct_reg2pet_seg_folder ... \
    --skip_ner --skip_merge --skip_qa
```

## Question Types

| Type | Category | Options | Input | Description |
|------|----------|---------|-------|-------------|
| **1a** | Organ recognition | 4-choose-1 | CT + mask | Identify organ name from CT image and segmentation mask |
| **1b** | Modality recognition | 2-choose-1 | CT or PET | Determine whether the image is CT or PET |
| **2a** | Morphological multi-select | 4-choose-2 | CT + mask | Select two correct descriptions of organ morphology |
| **2b** | Morphological true/false | 2-choose-1 | CT + mask | Determine if a given morphological description is correct |
| **2c** | FDG metabolism selection | 4-choose-1 | CT + PET + mask | Select the correct FDG metabolism description |
| **2a-enhanced** | Same as 2a | 4-choose-2 | CT | Mask-free version |
| **2b-enhanced** | Same as 2b | 2-choose-1 | CT | Mask-free version |
| **2c-enhanced** | Same as 2c | 4-choose-1 | CT + PET | Mask-free version |
| **3** | Report-level comprehensive | 4-choose-1 | CT + PET | Comprehensive understanding based on full report |

## Data Format

Each question is a JSON object with common fields:

```json
{
  "q_id": "000001",           // 6-digit unique ID
  "q_type": "1a",             // Question type identifier
  "content": "Question text...", // Question content
  "options": {                // Options (2b uses array ["A","B"])
    "A": "Option A content",
    "B": "Option B content",
    "C": "Option C content",
    "D": "Option D content"
  },
  "answer": "B",              // Correct answer letter (2a uses two letters like "CD")
  "CT_path": "imgs_data/CT/PTXH220103024_0000.nii.gz",
  "PET_path": "...",          // Included in types 1b/2c/3
  "segmentation_path": "...", // Included in types 1a/2a/2b/2c
  "entity": "brain"           // Corresponding anatomical entity (English)
}
```

> **Optional fields**: Some post-processing steps (e.g., abnormality classification) may add extra fields such as `"abnormality": true/false` to indicate whether the question describes a pathological finding.

## License

This project is open-sourced under the [MIT License](LICENSE) for academic research use only.
