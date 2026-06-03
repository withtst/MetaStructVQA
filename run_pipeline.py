"""
MetaStructVQA full pipeline automation script
One-click run from NER to post-processing.

Warning: This script is provided for debugging convenience only.
In production, it is strongly recommended to run each stage manually
to inspect intermediate results, handle exceptions, and resume from breakpoints.
"""
import argparse
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default paths (relative to the parent of the project root)
_DEFAULT_DATA_ROOT = os.path.dirname(BASE_DIR)


def _run(cmd, stage_name):
    """Execute a subprocess command with stage markers"""
    print("\n" + "=" * 72)
    print(f"  Stage: {stage_name}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 72 + "\n")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\n❌ Stage [{stage_name}] failed, exit code: {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n✅ Stage [{stage_name}] completed")


def main():
    parser = argparse.ArgumentParser(
        description="MetaStructVQA full pipeline automation (⚠️ For debugging only; run stages individually in production)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run_pipeline.py \\
    --reports_zh ../reports_prepare/origin_reports/report_zh.csv \\
    --reports_en ../reports_prepare/origin_reports/report_en.csv \\
    --reports_list ../deepseekNER/report_text_list.txt \\
    --ct_folder ../imgs_data/CT \\
    --pet_folder ../imgs_data/PET \\
    --ct_seg_folder ../imgs_data/CT_seg \\
    --ct_reg2pet_seg_folder ../imgs_data/CT_reg2PET_seg
        """)

    # Report data
    parser.add_argument("--reports_zh", required=True, help="Chinese report CSV path")
    parser.add_argument("--reports_en", required=True, help="English report CSV path")
    parser.add_argument("--reports_list", required=True, help="Report text list (JSON array) path")

    # Image data
    parser.add_argument("--ct_folder", required=True, help="CT image folder path")
    parser.add_argument("--pet_folder", required=True, help="PET image folder path")
    parser.add_argument("--ct_seg_folder", required=True, help="CT segmentation mask folder path")
    parser.add_argument("--ct_reg2pet_seg_folder", required=True, help="CT registered to PET segmentation mask path")

    # NER parameters
    parser.add_argument("--ner_workers", type=int, default=10, help="NER concurrent thread count (default: 10)")
    parser.add_argument("--ner_start", type=int, default=0, help="NER start index (default: 0)")
    parser.add_argument("--ner_end", type=int, default=490, help="NER end index (default: 490)")

    # Fusion parameters
    parser.add_argument("--merge_start", type=int, default=1, help="Fusion start number (default: 1)")
    parser.add_argument("--merge_end", type=int, default=491, help="Fusion end number (default: 491)")

    # QA generation
    parser.add_argument("--qa_types", nargs="+", default=["1a", "1b", "2ab", "2c", "3"],
                        choices=["1a", "1b", "2ab", "2c", "3"],
                        help="Question types to generate (default: all)")
    parser.add_argument("--qa_seed", type=int, default=42, help="QA random seed (default: 42)")
    parser.add_argument("--qa_test_mode", action="store_true", help="QA test mode")

    # Pipeline control
    parser.add_argument("--skip_ner", action="store_true", help="Skip NER stage (when NER results already exist)")
    parser.add_argument("--skip_merge", action="store_true", help="Skip data fusion stage")
    parser.add_argument("--skip_qa", action="store_true", help="Skip QA generation stage")

    args = parser.parse_args()

    py = sys.executable
    t0 = time.time()

    print("🚀 MetaStructVQA full pipeline started")
    print(f"   Report language: see configs.yaml")
    print(f"   QA types:  {args.qa_types}")
    print(f"   QA seed:  {args.qa_seed}")

    # ────────── Stages 1-3: NER Recognition ──────────
    if not args.skip_ner:
        prompt_file = os.path.join(BASE_DIR, "deepseekNER", "ner_prompt.txt")
        hierarchy_file = os.path.join(BASE_DIR, "deepseekNER", "anatomy_hierarchy_reference.txt")
        ner_output = os.path.join(BASE_DIR, "deepseekNER", "descriptions")

        # Select report file based on language config (NER scripts support auto-detection of CN/EN column names)
        import yaml
        with open(os.path.join(BASE_DIR, "QA_pipeline", "configs.yaml"), 'r', encoding='utf-8') as _f:
            _cfg = yaml.safe_load(_f)
        _lang = _cfg.get('report_language', 'chinese').lower()
        ner_reports_csv = args.reports_zh if _lang == 'chinese' else args.reports_en

        _run([py, os.path.join("deepseekNER", "deepseek_api_concurrent.py"),
              "--reports_csv", ner_reports_csv,
              "--prompt_file", prompt_file,
              "--hierarchy_file", hierarchy_file,
              "--reports_list_file", args.reports_list,
              "--output_dir", ner_output,
              "--start", str(args.ner_start),
              "--end", str(args.ner_end),
              "--max_workers", str(args.ner_workers)],
             "1-3. NER Recognition")

        _run([py, os.path.join("deepseekNER", "clean_NER_format.py")],
             "NER Format Fix")
    else:
        print("\n⏭️  Skipped NER stage (--skip_ner)")

    # ────────── Stage 4: Data Fusion ──────────
    if not args.skip_merge:
        ner_folder = os.path.join(BASE_DIR, "deepseekNER", "descriptions")

        _run([py, os.path.join("merge_seg_and_reports", "main_merger.py"),
              "--ct_folder", args.ct_folder,
              "--pet_folder", args.pet_folder,
              "--ct_seg_folder", args.ct_seg_folder,
              "--ct_reg2pet_seg_folder", args.ct_reg2pet_seg_folder,
              "--reports_en", args.reports_en,
              "--reports_zh", args.reports_zh,
              "--ner_folder", ner_folder,
              "--start", str(args.merge_start),
              "--end", str(args.merge_end)],
             "4. Data Fusion")
    else:
        print("\n⏭️  Skipped data fusion stage (--skip_merge)")

    # ────────── Stage 5: Entity Filtering ──────────
    _run([py, os.path.join("QA_pipeline", "entities_filter.py")],
         "5. Entity Filtering")

    # ────────── Stage 6: QA Generation ──────────
    if not args.skip_qa:
        qa_cmd = [py, os.path.join("QA_pipeline", "QA_generator.py"),
                  "--types"] + args.qa_types + ["--seed", str(args.qa_seed)]
        if args.qa_test_mode:
            qa_cmd.append("--test_mode")

        _run(qa_cmd, "6. QA Generation")
    else:
        print("\n⏭️  Skipped QA generation stage (--skip_qa)")

    # ────────── Stage 7: Post-processing ──────────
    _run([py, os.path.join("QA_pipeline", "split_type2.py")],
         "7a. Type 2 Split")

    _run([py, os.path.join("QA_pipeline", "convert_type_2a_to_mf.py")],
         "7b. 2a -> 2a-enhanced Conversion")

    _run([py, os.path.join("QA_pipeline", "convert_type_2b_to_mf.py")],
         "7c. 2b -> 2b-enhanced Conversion")

    _run([py, os.path.join("QA_pipeline", "convert_type_2c_to_mf.py")],
         "7d. 2c -> 2c-enhanced Conversion")

    _run([py, os.path.join("QA_pipeline", "filter_absent_organs.py")],
         "7e. Absent Organ Filtering")

    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  🎉 Pipeline completed! Elapsed: {elapsed / 60:.1f} minutes")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
