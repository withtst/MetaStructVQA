"""
DeepSeek NER API - Concurrent version
Performs batch NER recognition on PET-CT reports using multi-threaded DeepSeek API calls.
"""
import json
import os
from openai import OpenAI
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


class DeepSeekNER:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com")

    def process_1_report(self, prompt: str, anatomy_hierarchy: str, report: str) -> str:
        """Process a single report using the DeepSeek API"""
        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "你是一个PET-CT影像分析专家！"},
                {"role": "user", "content": f"{prompt},请参考以下解剖学实体层级结构:\n{anatomy_hierarchy},对这个影像报告所见部分进行处理:\n{report}"},
            ],
            stream=False
        )
        return response.choices[0].message.content

    def process_report_with_index(self, row_id: int, annoid: str, prompt: str,
                                  anatomy_hierarchy: str, report: str, output_dir: str) -> tuple:
        """Process a single report and return the index and result"""
        try:
            print(f"[Thread] Processing {row_id}th report (annoid: {annoid})...")
            result = self.process_1_report(prompt, anatomy_hierarchy, report)

            output_file = os.path.join(output_dir, f"{annoid}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)

            print(f"[Thread] Done {row_id}th report, saved to {output_file}")
            return (row_id, True, None)
        except Exception as e:
            print(f"[Thread] Error processing {row_id}th report: {e}")
            return (row_id, False, str(e))


def main():
    """Main function: concurrent report processing"""
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek NER concurrent batch processing")
    parser.add_argument("--reports_csv", required=True, help="Report CSV file path")
    parser.add_argument("--prompt_file", required=True, help="Prompt template file path")
    parser.add_argument("--hierarchy_file", required=True, help="Anatomical entity hierarchy reference file path")
    parser.add_argument("--reports_list_file", required=True, help="Report text list file path")
    parser.add_argument("--output_dir", default="descriptions", help="Output directory")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=100, help="End index (exclusive)")
    parser.add_argument("--max_workers", type=int, default=10, help="Number of concurrent threads")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ner = DeepSeekNER()

    print("Loading configuration files...")
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()
    with open(args.hierarchy_file, "r", encoding="utf-8") as f:
        anatomy_hierarchy = f.read()
    with open(args.reports_list_file, "r", encoding="utf-8") as f:
        reports = json.load(f)

    df = pd.read_csv(args.reports_csv)
    # Support both Chinese and English column names
    if '影像号' in df.columns:
        annoids = df['影像号'].tolist()
    elif 'Image ID' in df.columns:
        annoids = df['Image ID'].tolist()
    else:
        print(f"Error: '影像号' or 'Image ID' column not found in CSV")
        return

    start, end, max_workers = args.start, args.end, args.max_workers

    print(f"\nConfiguration:")
    print(f"  - Processing range: [{start}, {end})")
    print(f"  - Concurrent workers: {max_workers}")
    print(f"  - Total reports to process: {min(end - start, len(reports) - start)}\n")

    tasks = []
    for row_id in range(start, min(end, len(reports))):
        tasks.append((row_id, annoids[row_id], reports[row_id]))

    if not tasks:
        print("No reports to process!")
        return

    print(f"Starting concurrent processing with {max_workers} workers...\n")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                ner.process_report_with_index,
                row_id, annoid, prompt, anatomy_hierarchy, report, args.output_dir
            ): row_id
            for row_id, annoid, report in tasks
        }

        success_count = 0
        failed_count = 0
        failed_reports = []

        for future in as_completed(futures):
            row_id, success, error = future.result()
            if success:
                success_count += 1
            else:
                failed_count += 1
                failed_reports.append((row_id, error))

        print("\n" + "=" * 60)
        print("Processing Summary:")
        print(f"  Total:   {len(tasks)}")
        print(f"  Success: {success_count}")
        print(f"  Failed:  {failed_count}")

        if failed_reports:
            print("\nFailed Reports:")
            for row_id, error in failed_reports:
                print(f"  - Report {row_id}: {error}")

        print("=" * 60)


if __name__ == "__main__":
    main()
