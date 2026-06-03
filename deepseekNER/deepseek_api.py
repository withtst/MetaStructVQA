"""
DeepSeek NER API - Single-threaded version
Performs named entity recognition (NER) on PET-CT reports using the DeepSeek API.
"""
import json
import os
from openai import OpenAI
import pandas as pd


class DeepSeekNER:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com")

    def process_1_report(self, prompt: str, anatomy_hierarchy: str, report: str) -> str:
        """Process a single report using the DeepSeek API for NER extraction"""
        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "你是一个PET-CT影像分析专家！"},
                {"role": "user", "content": f"{prompt},请参考以下解剖学实体层级结构:\n{anatomy_hierarchy},对这个影像报告所见部分进行处理:\n{report}"},
            ],
            stream=False
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek NER single-threaded processing")
    parser.add_argument("--reports_csv", required=True, help="Report CSV file path (with '影像号' or 'Image ID' column)")
    parser.add_argument("--prompt_file", required=True, help="Prompt template file path")
    parser.add_argument("--hierarchy_file", required=True, help="Anatomical entity hierarchy reference file path")
    parser.add_argument("--reports_list_file", required=True, help="Report text list file path (Python list format)")
    parser.add_argument("--output_dir", default="descriptions", help="Output directory (default: descriptions/)")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=10, help="End index (exclusive)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ner = DeepSeekNER()
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
        exit(1)

    for row_id, report in enumerate(reports, start=0):
        if row_id < args.start:
            continue
        if row_id >= args.end:
            break
        print(f"Processing {row_id}th report...")
        result = ner.process_1_report(prompt, anatomy_hierarchy, report)
        annoid = annoids[row_id]
        output_file = os.path.join(args.output_dir, f"{annoid}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
            print(f"Processed {row_id}th report, result saved to {output_file}")
