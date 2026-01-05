import json
import os
import glob

# 配置
OUTPUT_DIR = "../result/eventgraph_full"
METHOD = "EventGraph-LMM"
BACKBONE = "Video-LLaVA-7B"
NUM_CHUNKS = 108

print(f"🔄 开始合并 {NUM_CHUNKS} 个chunk结果...")
print(f"📂 目录: {OUTPUT_DIR}")

# 读取所有chunk文件
all_results = []
missing_chunks = []

for i in range(NUM_CHUNKS):
    chunk_file = f'{OUTPUT_DIR}/VideoMME_{METHOD}_all_{BACKBONE}_chunk{i}.json'
    if os.path.exists(chunk_file):
        try:
            with open(chunk_file, 'r') as f:
                data = json.load(f)
                all_results.extend(data)
        except Exception as e:
            print(f"❌ 读取错误 Chunk {i}: {e}")
            missing_chunks.append(i)
    else:
        print(f"⚠️  缺失 Chunk {i}")
        missing_chunks.append(i)

# 检查完整性
if missing_chunks:
    print(f"\n❌ 合并失败！缺失 {len(missing_chunks)} 个chunk: {missing_chunks}")
    print("💡 请补跑这些chunk后再试。")
    exit(1)

# 保存合并结果
output_file = f'{OUTPUT_DIR}/VideoMME_{METHOD}_all_{BACKBONE}_merged.json'
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"\n✅ 合并成功！")
print(f"📄 结果文件: {output_file}")

# 计算准确率
correct = sum(1 for r in all_results if r.get('pred') == r.get('gt'))
total = len(all_results)
if total > 0:
    accuracy = 100 * correct / total
    print(f'\n📊 最终结果:')
    print(f'  - 样本数: {total}')
    print(f'  - 正确数: {correct}')
    print(f'  - 准确率: {accuracy:.2f}%')
else:
    print('\n⚠️  结果为空！')
