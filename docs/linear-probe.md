# 线性探针训练与评估

<!-- 请在这里稍微解释线性探针的原理，并列需要用户操作的主要步骤 -->

线性探针是一个简单的分类器，它将 CLIP 特征映射到类别标签。

大致流程如下：
1. 准备数据集，每个类别一个文件夹，文件夹名作为类别标签。
2. 运行线性探针训练脚本，指定数据集路径、批量大小、训练轮数、学习率、正则化系数、验证集比例、随机种子和输出路径。
3. 训练完成后，运行线性探针评估脚本，指定数据集路径、批量大小、探针路径和报告目录。
4. 评估完成后，会在报告目录下生成 `summary.json`、`per_label.csv`、`badcases.csv` 三个文件，分别包含模型在验证集上的准确率、每个类别的准确率、错误分类样本的详细信息。
5. 选择满意的模型，修改 `config.json` 中的 `linear_probe` 字段为模型路径。


## 标注

在项目根目录下创建 `dataset/training` 文件夹，每个类别一个子文件夹，文件夹名作为类别标签。

```
dataset
├── eval
└── training
   ├── chat
   ├── game
   ├── idle
   ├── music
   ├── programming
   ├── search
   ├── sheet
   ├── unknown
   ├── video
   └── writing
      └── 17_25_00_2.png
```

## 训练线性探针

```bash
cd {project_root_dir}

sa-train \
  --data-root dataset/training \
  --batch-size 32 \
  --epochs 20 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --val-split 0.2 \
  --seed 42 \
  --model-name linear_probe_balanced
```

模型将生成在 `checkpoints/{model-name}/model.pth`。

以上训练会同时生成：`checkpoints/{model-name}/model.meta.json`（训练超参与数据统计），`checkpoints/{model-name}/history.csv`（训练曲线）。

### 训练参数含义

- --batch-size : 每次参数更新前参与计算的样本数量。影响吞吐、显存占用和梯度噪声水平。
- --epochs : 完整遍历训练集的次数。更高通常提升拟合度，但过高可能过拟合。
- --lr : 学习率， nn.Linear 权重每步更新的步长。过大易震荡，过小收敛慢或停滞。
- --weight-decay : 权重衰减系数（L2 正则），抑制过拟合并鼓励小权重。
- --val-split : 训练/验证划分比例（分层按类别拆分）。越大验证集越多、训练样本越少。
- --seed : 随机种子。固定数据拆分与初始化，保证复现实验结果。

### 训练影响

- batch-size
  - 大批次降低梯度方差、训练更稳定，速度快；但可能降低泛化，且占用更多显存。
  - 小批次增加梯度噪声，有时能提高泛化，但训练波动更大。
- epochs
  - 线性探针一般几十轮即可收敛；结合 history.csv 中的 val_acc 观察是否已饱和。
- lr
  - 线性分类器在 CLIP 特征上通常对 1e-3 ~ 5e-4 区间敏感。
  - 现象判断：震荡/精度不上升→降 lr ；收敛过慢→升 lr 。
- weight-decay
  - 轻度正则（如 1e-4 ）通常有效；过大导致欠拟合， val_acc 上不去。
  - 类间边界清晰时可适当减小；数据噪声大/特征冗余时可适当增大。
- val-split
  - 分层划分可避免类别偏差。验证比例过大→训练数据不足；过小→评估不稳定。
  - 数据量较小建议 0.1~0.2 ，保证评估可靠性与充足训练样本。
- seed
  - 保证拆分和训练过程一致，便于横向对比。更换 seed 可评估结果稳定性。

### 调参建议

- 初始配置
  - --batch-size 32 、 --epochs 20 、 --lr 5e-4 、 --weight-decay 1e-4 、 --val-split 0.2 、 --seed 42
- 观察与迭代
  - 查看 *.history.csv 的 train_loss 与 val_acc 曲线，若前期震荡明显且未提升，降低 lr 或增大 batch-size 。
  - 若训练集拟合很好但验证精度下降，增大 weight-decay 或减小 epochs 。
  - 类别不平衡时可保持现有“类别加权”设置不变，同时适度提高 val-split 以稳定评估。
- 横向对比
  - 使用 *.meta.json 的 best_val_acc 与评估报告中的 overall_acc 、 per_label.csv ，对比不同超参组合在整体与各类别的表现。
  - 固定 seed 后只改动一个参数，更易定位影响来源

## 评估线性探针


```bash
cd {project_root}

sa-eval

# or

sa-eval \
  --data-root dataset/eval \
  --batch-size 32 \
  --probe checkpoints/linear_probe_balanced/model.pth \
  --report-dir test/linear_probe
```

评估会在 `test/linear_probe/<probe 名称>/` 下生成 `summary.json`、`per_label.csv`、`badcases.csv`，便于横向对比。

## 应用

在配置文件 `config.json` 中，将 `linear_probe` 字段设置为模型路径 `checkpoints/linear_probe_balanced/model.pth`。

```json
{
  ...
  "linear_probe": "./checkpoints/linear_probe_balanced/model.pth",
  ...
}
```