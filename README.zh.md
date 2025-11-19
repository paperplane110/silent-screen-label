# Silent Screen Label

自然而然地、不被打扰的屏幕使用统计

- 🤫 在后台每分钟自动截图，静默的记录屏幕活动；
- 🔐 所有数据与分析均在本地完成，不经网络，避免泄露风险。
- 🔍 使用轻量级 CLIP 在本地判断当下活动并自动贴标签。
- 🔧 可配置分类标签，支持自定义。

![streamlit.png](assets/streamlit.png)

## 一、安装

安装分为两个步骤：

1. 安装 python 依赖
2. 下载预训练模型

### 1.1 安装 python 依赖

#### 使用 `uv`
- 创建虚拟环境并安装
  - `uv venv`
  - `source .venv/bin/activate`
  - `uv pip install -e .`

#### 使用 `pip`
- 创建虚拟环境并安装
  - `python -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -e .`

### 1.2 下载预训练模型

- CLIP-ViT-B-32-laion2B-s34B-b79K
  - [下载](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin)
  - 放置在 `checkpoints/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/`

## 二、使用方法

使用步骤：

1. 配置文件（可以先不做，使用默认配置）
2. 启动后台截图进程
3. 可视化

### 2.1 配置文件

配置文件决定了

1. 屏幕截图保存在哪
2. 分类标签是哪些
3. 模型参数

路径：项目根目录 `config.json`
- 定时截图相关配置项
  - `cycle`：截图间隔秒数（默认 `60` 秒）
  - `dir`：输出根目录（默认项目根目录）
- 模型配置：
  - `batch_size`：推理批大小
  - `clip_weights`：CLIP 权重文件本地路径
  - `clip_agg`: CLIP 特征聚合方式（默认 `mean`，可选 `mean`、`sum` 或 `max`）
  - `linear_probe`：线性探针模型路径
  - `clip_prompts`：`{prompt, label}` 列表

可参考 [`config.json`](config.json) 中的默认值。

【重要】

- 想要实现分类功能，需要配置 `clip_prompts`：`{prompt, label}` 列表
- 每个 `{prompt, label}` 对表示一个分类类别，
  - `prompt` 是对一张截图的描述，
  - `label` 是该描述所对应的类别名称。
  - 例如：
    - `{"prompt": "a screenshot of an IDE or code editor", "label": "编程"}`
    - `{"prompt": "a screenshot of a video player", "label": "看视频"}`
- 支持多个描述对应同一个类别，例如：
  - `{"prompt": "a screenshot of a video player", "label": "看视频"}`
  - `{"prompt": "a screenshot of a bilibili website", "label": "看视频"}`

### 2.2 启动后台截图进程

启动后台截图进程，可能需要通过隐私设置允许屏幕截图。

- 截图将保存到 `{project_dir}/screenshots/<YYYYMMDD>`
- `sa run` 或 `python -m screen_analysis.main run`
- 参数：
  - `--cycle <seconds>` 截图间隔
  - `--dir <path>` 输出根目录
  - `--menubar` 显示 macOS 菜单栏指示器

```bash
sa run
```

### 2.3 可视化

使用 Streamlit 展示屏幕使用的时间线

- 在命令行中执行 `sa-studio` 或 `python -m screen_analysis.commands.studio.cli`
- 访问 `http://localhost:8501` 查看时间线

```bash
sa-studio
```

### 2.4 手动分析

对截图进行分析并写出报告到 `reports/<YYYYMMDD>`

- 在命令行中执行 `sa analyze [YYYYMMDD]` 或 `python -m screen_analysis.main analyze [YYYYMMDD]`
- 参数：
  - [YYYYMMDD] 如 20251119，可不填写，不填写时默认是今天
  - `--debug` 输出逐图分类信息
  - `--overwrite` 强制全量分析（忽略增量）

增量分析说明：
- 当存在 `reports/<YYYYMMDD>/executed_at.log` 与当日 `timeline.csv` 时，只分析“上次执行时间”之后的新截图。
- 仅覆盖新增分钟的 `category`，其他行保持不变。

### 2.5 手动清理

清理某日的截图

- 在命令行中执行 `sa del -d 20251118` 或 `python -m screen_analysis.main del -d 20251118`

```bash
sa del -d 20251118
```

## 三、测试

- 性能测试请参考 [测试-性能测试](docs/test.md#性能测试)
- 模型效果测试请参考：[测试-模型评估](docs/test.md#模型评估)

## 四、分类优化

如果分类效果不理想，可尝试以下优化：

1. 调整分类标签
   - 检查 `config.json` 中的 `clip_prompts` 是否准确描述了屏幕活动。
   - 可以添加、删除或修改标签，根据实际情况调整。
2. 越过 prompt 使用线性探针，
   - 我们提供了一个简单的训练脚本，来训练线性探针模型。只需要极少的训练数据（100 张截图） 就可以显著提升分类效果。
   - 请参考这个文档 [优化-训练线性探针模型](docs/linear-probe.md) 

