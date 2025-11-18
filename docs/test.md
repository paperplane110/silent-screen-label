
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

测试 Silent Screen Label 的模型性能

```bash
# 务必先到项目根目录
cd {project_root}

# 直接评估当前使用的模型
sa-eval

# 手动配置测试数据集、测试模型路径
sa-eval \
    --data-root dataset/eval \
    --clip-weights models/clip.pt \
    --probe test/probe.pt \
    --batch-size 32 \
    --report-dir test/linear_probe
```

参数说明：

- `--data-root`: 测试数据集根目录
- `--clip-weights`: CLIP 模型权重路径, 默认使用 config.json 的配置
- `--probe`: 训练好的线性探针模型路径，默认使用 config.json 的配置
- `--batch-size`: 推理批量大小
- `--report-dir`: 评估报告输出目录，默认为 `test/linear_probe/`

报告将生成在 `test/linear_probe/{probe_name}/`
