
# 测试

## 性能测试

测试 Silent Screen Label 的性能

```bash
cd {project_root}

# 监控 10 分钟
sa-monitor --duration 600
```

报告将生成在 `test/performance/performance_log.csv`

## 模型评估

测试 Silent Screen Label 的模型性能，这里的评估主要针对 prompt 的效果。

```bash
# 务必先到项目根目录
cd {project_root}

# 直接评估（读取 config.json 默认配置）
sa-eval clip

# 指定评估数据集
sa-eval clip \
  --data-root dataset/eval \
  --batch-size 32 \
  --clip-weights checkpoints/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin \
  --clip-agg mean \
  --threshold 0.3
```

参数：

- `--data-root`: 测试数据集根目录
- `--batch-size`: 推理批量大小
- `--clip-weights`: CLIP 模型权重路径, 默认使用 config.json 的配置
- `--clip-agg`: CLIP 特征聚合方式，可选 `mean` 或 `max`
- `--threshold`: 分类阈值，默认 0.3
