"""
Data fusion module: integrates CT/PET images, segmentation masks, NER results, and reports
into a unified metadata file. Generates one VQA_pre_version/sub_xxx/metadata.json per subject.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anno_converter import anno_converter
import json
import logging
import pandas as pd


class mainMerger:
    def __init__(self, CT_folder, PET_folder,
                 CTseg_folder, CTreg2PETseg_folder,
                 reports_csv_en, reports_csv_zh,
                 NER_folder,
                 entities_mapping_path=None,
                 output_root=None):

        self.CT_folder = CT_folder
        self.PET_reg2CT_folder = PET_folder
        self.CTseg_folder = CTseg_folder
        self.reports_csv_en = reports_csv_en
        self.reports_csv_zh = reports_csv_zh
        self.NER_folder = NER_folder

        # Entity mapping table path (default: relative to project root)
        if entities_mapping_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            entities_mapping_path = os.path.join(base_dir, '..', 'entities_nonsub_mapping_table.json')
        self.en_zh_mapping_table_path = entities_mapping_path
        with open(self.en_zh_mapping_table_path, "r", encoding="utf-8") as f:
            self.en_zh_mapping_js = json.load(f)

        # Output root directory (default: VQA_pre_version/ under project root)
        if output_root is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            output_root = os.path.join(base_dir, '..', 'VQA_pre_version')
        self.output_root = output_root

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def init_vqa_folder(self, anno_id: str) -> None:
        """Create the VQA preparation folder for a given report based on anno_id"""
        vqa_folder = os.path.join(self.output_root, anno_id)
        os.makedirs(vqa_folder, exist_ok=True)

        report_zh = ["", ""]
        report_en = ["", ""]

        try:
            df_zh = pd.read_csv(self.reports_csv_zh)
        except Exception as e:
            logging.error(f"file_error: Failed to read Chinese reports CSV: {e}")
        else:
            row = df_zh.loc[df_zh['影像号'] == anno_id]
            if not row.empty:
                report_zh[0] = row['影像描述'].values[0]
                report_zh[1] = row['报告诊断'].values[0]
            else:
                logging.warning(f"report_warning: No Chinese report found for Image ID: {anno_id}")

        try:
            df_en = pd.read_csv(self.reports_csv_en, encoding='latin-1')
        except Exception as e:
            logging.error(f"file_error: Failed to read English reports CSV: {e}")
        else:
            row = df_en.loc[df_en['Image ID'] == anno_id]
            if not row.empty:
                report_en[0] = row['Image Description'].values[0]
                report_en[1] = row['Report Diagnosis'].values[0]
            else:
                logging.warning(f"report_warning: No English report found for Image ID: {anno_id}")

        metadata_path = os.path.join(vqa_folder, "metadata.json")
        metadata = {
            "anno_id": anno_id,
            "CT_path": os.path.join(self.CT_folder, f"{anno_converter(anno_id)}_0000.nii.gz"),
            "PET_reg2CT_path": os.path.join(self.PET_reg2CT_folder, f"{anno_converter(anno_id)}_0000.nii.gz"),
            "CT_seg_folder": os.path.join(self.CTseg_folder, f"{anno_converter(anno_id)}"),
            "report_zh": {
                "description": report_zh[0],
                "diagnosis": report_zh[1]
            },
            "report_en": {
                "description": report_en[0],
                "diagnosis": report_en[1]
            },
            "NER_path": os.path.join(self.NER_folder, f"{anno_id}.txt")
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data fusion: integrate CT/PET/masks/NER/reports into unified metadata")
    parser.add_argument("--ct_folder", required=True, help="CT image folder path")
    parser.add_argument("--pet_folder", required=True, help="PET (registered to CT) image folder path")
    parser.add_argument("--ct_seg_folder", required=True, help="CT segmentation mask folder path")
    parser.add_argument("--ct_reg2pet_seg_folder", required=True, help="CT registered to PET segmentation mask path")
    parser.add_argument("--reports_en", required=True, help="English report CSV file path")
    parser.add_argument("--reports_zh", required=True, help="Chinese report CSV file path")
    parser.add_argument("--ner_folder", required=True, help="NER results folder path")
    parser.add_argument("--entities_mapping", default=None, help="Entity name mapping table JSON path")
    parser.add_argument("--output_root", default=None, help="Output root directory (default: VQA_pre_version/)")
    parser.add_argument("--start", type=int, default=1, help="Start subject number (default: 1)")
    parser.add_argument("--end", type=int, default=491, help="End subject number (exclusive, default: 491)")
    args = parser.parse_args()

    merger = mainMerger(
        CT_folder=args.ct_folder,
        PET_folder=args.pet_folder,
        CTseg_folder=args.ct_seg_folder,
        CTreg2PETseg_folder=args.ct_reg2pet_seg_folder,
        reports_csv_en=args.reports_en,
        reports_csv_zh=args.reports_zh,
        NER_folder=args.ner_folder,
        entities_mapping_path=args.entities_mapping,
        output_root=args.output_root,
    )

    for i in range(args.start, args.end):
        merger.init_vqa_folder(f"sub_{i:03d}")
