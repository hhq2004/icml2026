# /root/hhq/main_code/utils/metrics.py
import json
import re
import sys
import glob
import os

def extract_option(text):
    text = str(text).strip()
    if not text: return "C"
    if len(text) == 1 and text.upper() in "ABCD": return text.upper()
    match = re.search(r'\(([ABCD])\)', text)
    if match: return match.group(1)
    match = re.search(r'Option ([ABCD])', text, re.IGNORECASE)
    if match: return match.group(1)
    match = re.match(r'^([ABCD])\s', text)
    if match: return match.group(1)
    return "C"

def evaluate_files(file_pattern_list):
    # 处理 Shell 通配符传递进来可能是 List 的情况
    files = []
    for pattern in file_pattern_list:
        # 如果 pattern 包含通配符，glob 展开；如果是具体文件，直接加
        found = glob.glob(pattern)
        if not found and not '*' in pattern and os.path.exists(pattern):
            found = [pattern]
        files.extend(found)
        
    if not files:
        # 这里是关键修改：没找到文件不报错，而是打印提示
        print(f"⚠️  [Metrics] No result files found matching: {file_pattern_list}")
        print("   -> Did the inference script crash? Check the .log files!")
        return

    all_data = []
    print(f"📊 Found {len(files)} result files. Merging...")
    
    for f_path in files:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                all_data.extend(chunk_data)
        except Exception as e:
            print(f"  ⚠️ Error reading {f_path}: {e}")

    if not all_data:
        print("No valid data loaded.")
        return

    correct = 0
    total = len(all_data)
    for item in all_data:
        pred_raw = item.get('pred', '')
        gt = item.get('gt', 'C')
        if extract_option(pred_raw) == gt:
            correct += 1

    acc = correct / total * 100
    print("-" * 30)
    print(f"🏆 Final Results:")
    print(f"   Total:    {total}")
    print(f"   Accuracy: {acc:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        evaluate_files(sys.argv[1:])
    else:
        print("Usage: python metrics.py results/*.json")