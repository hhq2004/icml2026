#!/bin/bash

# TransNet V2 Shot Detection 修复后测试脚本
# 对比修复前后的EventGraph-LMM性能

echo "========================================================================"
echo "EventGraph-LMM - TransNet V2修复后完整测试"
echo "========================================================================"
echo "修复内容："
echo "  1. ✅ 修复TransNet V2 API调用"
echo "  2. ✅ 添加NMS避免过度碎片化"
echo "  3. ✅ 改进输入预处理（48x27标准尺寸）"
echo "  4. ✅ 添加device设置支持GPU加速"
echo ""
echo "预期改进："
echo "  - Shot detection更精确（专业级）"
echo "  - Event切分更合理（语义完整）"
echo "  - 准确率提升3-10个百分点"
echo "========================================================================"

# 第一步：测试TransNet V2是否正常工作
echo ""
echo "🧪 Step 1: 测试TransNet V2功能..."
echo ""

cd /root/hhq/main_code

python test_transnet_v2.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ TransNet V2测试失败！"
    echo "请检查："
    echo "  1. pip install transnetv2-pytorch"
    echo "  2. CUDA是否可用"
    exit 1
fi

echo ""
echo "✅ TransNet V2测试通过！"
echo ""

# 第二步：运行EventGraph-LMM完整测试
echo "========================================================================"
echo "🚀 Step 2: 运行EventGraph-LMM (200样本，快速验证)..."
echo "========================================================================"
echo ""

cd /root/hhq/main_code/Script

# 使用200样本版本（快速验证）
bash run_eventgraph_200samples.sh

echo ""
echo "========================================================================"
echo "📊 测试完成！结果分析："
echo "========================================================================"

# 等待片刻让文件写入完成
sleep 2

# 查找最新的merged结果文件（200样本版本）
RESULT_FILE=$(ls -t ../result/*/VideoMME_EventGraph-LMM_all_Video-LLaVA-7B_merged.json 2>/dev/null | head -1)

if [ -z "$RESULT_FILE" ]; then
    echo "⚠️  未找到结果文件，请手动检查 result/ 目录"
    exit 1
fi

echo ""
echo "结果文件: $RESULT_FILE"
echo ""

# 计算准确率
python -c "
import json
import sys

try:
    with open('$RESULT_FILE', 'r') as f:
        results = json.load(f)
    
    total = len(results)
    correct = sum(1 for r in results if r.get('pred') == r.get('gt'))
    accuracy = 100 * correct / total if total > 0 else 0
    
    print('=' * 72)
    print('📈 EventGraph-LMM 性能报告 (TransNet V2修复后 - 200样本)')
    print('=' * 72)
    print(f'  总样本数: {total}')
    print(f'  正确数:   {correct}')
    print(f'  准确率:   {accuracy:.2f}%')
    print('=' * 72)
    print('')
    print('🔍 验证修复效果:')
    print('  - 200样本基准（修复前）: 需要对比之前的结果')
    print(f'  - 200样本当前（修复后）: {accuracy:.2f}% ({correct}/{total})')
    print('')
    print('💡 提示：')
    print('  - 如果准确率合理（30-40%），说明TransNet V2工作正常')
    print('  - 查看日志确认是否使用了TransNet V2（而非fallback）')
    print('  - 如果效果好，可以运行完整2697样本测试')
    print('')
    
    print('=' * 72)
    
except Exception as e:
    print(f'❌ 错误: {e}')
    sys.exit(1)
"

echo ""
echo "========================================================================"
echo "📝 下一步建议："
echo "========================================================================"
echo ""
echo "如果准确率提升 >= 3%:"
echo "  ✅ 修复成功！Shot Detection已解决"
echo "  → 继续优化CELF和Graph-CoT"
echo ""
echo "如果准确率提升 < 3%:"
echo "  ⚠️  需要进一步排查："
echo "  1. 查看日志确认TransNet V2是否真正运行（而非fallback）"
echo "  2. 检查shot数量是否合理（3-100个）"
echo "  3. 检查utils/celf_solver.py的选择逻辑"
echo "  4. 检查utils/graph_builder.py的PageRank计算"
echo ""
echo "查看详细日志:"
echo "  tail -f ../result/*/gpu*.log"
echo ""
echo "========================================================================"
