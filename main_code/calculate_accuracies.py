#!/usr/bin/env python3
"""
计算所有测试结果的准确率
扫描result目录下所有子文件夹，查找*_merged.json文件并计算准确率

JSON格式: [{"id": "...", "pred": "A", "gt": "B"}, ...]

使用方法:
    cd /root/hhq/main_code
    python calculate_accuracies.py
    
    # 保存到文件
    python calculate_accuracies.py > accuracy_report.txt
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

# 定义结果目录（相对于当前脚本位置）
SCRIPT_DIR = Path(__file__).parent
RESULT_DIR = SCRIPT_DIR / "result"

# 方法名称映射（用于美化输出）
METHOD_MAPPING = {
    "dycoke": "DyCoke",
    "eventgraph": "EventGraph-LMM",
    "fastv": "FastV",
    "no_compression": "No-Compression",
    "scenegraph": "SceneGraph-Cap",
    "test_50": "Q-Frame",  # test_50samples_7b_parallel 是 Q-Frame
    "tome": "ToMe",
}

def identify_method(folder_name: str) -> str:
    """根据文件夹名称识别测试方法"""
    folder_lower = folder_name.lower()
    
    for key, method in METHOD_MAPPING.items():
        if key in folder_lower:
            return method
    
    return "Unknown"

def estimate_samples(folder_name: str) -> str:
    """
    从文件夹名称估计样本数量
    
    已知样本数:
    - eventgraph_test: 600 samples
    - eventgraph_full: 2697 samples (全量)
    - *_200samples_*: 200 samples
    - *_50samples_*: 50 samples
    - smoke_test: 通常是 50 samples
    """
    folder_lower = folder_name.lower()
    
    # 特殊情况
    if "eventgraph_test" in folder_lower:
        return "600"
    elif "full" in folder_lower and "eventgraph" in folder_lower:
        return "2697"
    
    # 从名称中提取
    if "200samples" in folder_lower or "200_samples" in folder_lower:
        return "200"
    elif "50samples" in folder_lower or "50_samples" in folder_lower:
        return "50"
    elif "smoke_test" in folder_lower or "smoke" in folder_lower:
        return "50"
    
    return "?"

def calculate_accuracy(json_file: Path, debug=False) -> Tuple[int, int, float]:
    """
    计算单个merged.json文件的准确率
    
    JSON格式: [{"id": "...", "pred": "A", "gt": "B"}, ...]
    
    Returns:
        (correct_count, total_count, accuracy_percentage)
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return 0, 0, 0.0
        
        total = len(data)
        
        # 调试：打印第一个样本的结构
        if debug and total > 0:
            print(f"\n[DEBUG] First sample in {json_file.name}:")
            print(f"  Keys: {list(data[0].keys())}")
            print(f"  Sample: {data[0]}")
        
        # 比较pred和gt字段
        correct = 0
        for item in data:
            pred = item.get('pred', '').strip()
            gt = item.get('gt', '').strip()
            
            if pred and gt and pred == gt:
                correct += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        return correct, total, accuracy
    
    except Exception as e:
        print(f"  ❌ Error reading {json_file}: {e}")
        return 0, 0, 0.0

def find_merged_json(folder: Path) -> Path:
    """在文件夹中查找*_merged.json文件"""
    merged_files = list(folder.glob("*_merged.json"))
    
    if len(merged_files) == 0:
        return None
    elif len(merged_files) == 1:
        return merged_files[0]
    else:
        # 如果有多个，优先选择最大的（可能是最完整的）
        return max(merged_files, key=lambda x: x.stat().st_size)

def main():
    result_path = Path(RESULT_DIR)
    
    if not result_path.exists():
        print(f"❌ Result directory not found: {RESULT_DIR}")
        print(f"   Please run this script from: {SCRIPT_DIR}")
        return
    
    print("=" * 100)
    print("📊 All Test Results Accuracy Report")
    print("=" * 100)
    print()
    
    # 存储所有结果
    results = []
    
    # 遍历所有子文件夹
    for folder in sorted(result_path.iterdir()):
        if not folder.is_dir():
            continue
        
        # 查找merged.json文件
        merged_file = find_merged_json(folder)
        
        if merged_file is None:
            print(f"⚠️  {folder.name:<45} - No merged.json found")
            continue
        
        # 计算准确率
        correct, total, accuracy = calculate_accuracy(merged_file)
        
        if total == 0:
            print(f"⚠️  {folder.name:<45} - Empty or invalid JSON")
            continue
        
        # 识别方法
        method = identify_method(folder.name)
        expected_samples = estimate_samples(folder.name)
        
        # 存储结果
        results.append({
            'folder': folder.name,
            'method': method,
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'file': merged_file.name,
            'expected_samples': expected_samples
        })
        
        # 打印结果
        status = "✅" if accuracy > 25 else "⚠️ "
        samples_match = "✓" if str(total) == expected_samples or expected_samples == "?" else "⚠"
        print(f"{status} {folder.name:<45} | {method:<20} | {correct:>4}/{total:<4} = {accuracy:>6.2f}% {samples_match}")
    
    print()
    print("=" * 100)
    print("📈 Summary by Method")
    print("=" * 100)
    print()
    
    # 按方法分组
    method_groups = {}
    for result in results:
        method = result['method']
        if method not in method_groups:
            method_groups[method] = []
        method_groups[method].append(result)
    
    # 打印每个方法的所有测试
    for method in sorted(method_groups.keys()):
        tests = method_groups[method]
        print(f"\n🔹 {method}")
        print(f"{'─' * 100}")
        
        for test in sorted(tests, key=lambda x: x['total'], reverse=True):
            samples_info = f"{test['total']} samples"
            print(f"  • {test['folder']:<45} | {samples_info:<15} | {test['accuracy']:>6.2f}%")
        
        # 如果有多个测试，显示最佳结果
        if len(tests) > 1:
            best = max(tests, key=lambda x: x['total'])
            print(f"  └─ Best (largest dataset): {best['accuracy']:.2f}% ({best['correct']}/{best['total']})")
    
    print()
    print("=" * 120)
    print("🏆 Complete Ranking (All Tests)")
    print("=" * 120)
    print()
    
    # 为每个结果添加描述性标签
    for result in results:
        folder = result['folder'].lower()
        
        # 添加描述标签
        if 'full' in folder:
            result['label'] = f"{result['method']} (Full Dataset)"
        elif '200samples' in folder or '200_samples' in folder:
            if 'parallel' in folder:
                result['label'] = f"{result['method']} (200 samples, parallel)"
            else:
                result['label'] = f"{result['method']} (200 samples)"
        elif '50samples' in folder or '50_samples' in folder:
            result['label'] = f"{result['method']} (50 samples)"
        elif 'test' in folder and result['total'] == 600:
            result['label'] = f"{result['method']} (600 samples, Shot Detection fixed)"
        elif 'smoke' in folder:
            result['label'] = f"{result['method']} (Smoke test)"
        else:
            result['label'] = f"{result['method']} ({result['total']} samples)"
    
    # 按准确率排序（所有结果）
    ranked_all = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    for i, result in enumerate(ranked_all, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:>2}."
        print(f"{medal:>4} {result['label']:<45} | {result['accuracy']:>6.2f}% ({result['correct']:>4}/{result['total']:<4})")
    
    print()
    print("=" * 120)
    print(f"📁 Total folders scanned: {len(list(result_path.iterdir()))}")
    print(f"✅ Valid results found: {len(results)}")
    print("=" * 120)
    print()
    print("Legend:")
    print("  ✓ = Sample count matches expected")
    print("  ⚠ = Sample count mismatch (check if test was interrupted)")
    print()
    print("Note:")
    print("  - Results with more samples are generally more reliable")
    print("  - Compare methods on the same sample count for fair evaluation")

if __name__ == "__main__":
    main()
