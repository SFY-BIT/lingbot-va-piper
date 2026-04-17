<h1 align="center">LingBot-VA：用于机器人控制的因果世界建模</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2601.21998"><img src="https://img.shields.io/static/v1?label=论文&message=PDF&color=red&logo=arxiv"></a>
  <a href="https://technology.robbyant.com/lingbot-va"><img src="https://img.shields.io/badge/项目-主页-blue"></a>
  <a href="https://huggingface.co/collections/robbyant/lingbot-va"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20模型&message=HuggingFace&color=orange"></a>
  <a href="https://modelscope.cn/collections/Robbyant/LingBot-VA"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%96%20模型&message=ModelScope&color=purple"></a>
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/许可证-Apache--2.0-green"></a>
</p>

<p align="center">
  <img src="assets/teaser_v3.png" width="100%">
</p>

https://github.com/user-attachments/assets/cec7b7a6-953b-4fa4-8f1a-47efc1fce547

## 💫 认识 **LingBot-VA**：一个同时进行世界建模与动作生成的自回归扩散框架

**LingBot-VA** 主要关注以下方向：
- **自回归视频-动作世界建模**：在统一的交错序列中同时建模视觉动态预测与动作推断，同时保持两者在概念上的区别。
- **高效执行**：采用双流 Mixture-of-Transformers（MoT）架构，并结合异步执行与 KV Cache。
- **长时程性能与泛化能力**：在样本效率、长时程任务成功率以及新场景泛化上均带来显著提升。

# 🚀 更新动态
- **[2026-02-17]** 发布后训练代码与数据集，支持在自定义机器人操作数据集上微调 LingBot-VA。
- **[2026-01-29]** 发布共享骨干网络版本的权重与代码，拆分版本敬请期待。

---

# 📦 模型下载
- **用于后训练的预训练检查点**

| 模型名称 | Huggingface 仓库 | ModelScope 仓库 | 说明 |
| :--- | :--- | :--- | :--- |
| lingbot-va-base | [🤗 robbyant/lingbot-va-base](https://huggingface.co/robbyant/lingbot-va-base) | [🤖 Robbyant/lingbot-va-base](https://modelscope.cn/models/Robbyant/lingbot-va-base) | 使用共享骨干网络的 LingBot-VA |
| lingbot-va-posttrain-robotwin | [🤗 robbyant/lingbot-va-posttrain-robotwin](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin) | [🤖 Robbyant/lingbot-va-posttrain-robotwin](https://modelscope.cn/models/Robbyant/lingbot-va-posttrain-robotwin) | 使用共享骨干网络的 LingBot-VA-Posttrain-Robotwin |

- **后训练数据集**

| 数据集名称 | 仓库 | 说明 |
| :--- | :--- | :--- |
| robotwin-clean-and-aug-lerobot | [🤗 robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot) | 清洗并增强后的 RoboTwin 数据集，采用 LeRobot 格式，可用于后训练 |

---

# 🛠️ 快速开始

## 安装
**环境要求**
- Python == 3.10.16
- PyTorch == 2.9.0
- CUDA 12.6

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install flash-attn --no-build-isolation
```

## ⚠️ 重要：`attn_mode` 配置

> **你必须根据当前是训练还是推理，修改 `attn_mode` 设置。**
> 由于 LingBot-VA 是通过 `from_pretrained` 加载的，这个参数会从模型目录下的 **`transformer/config.json`** 读取。
> 因此在启动前，你需要 **手动编辑** 这个文件。
>
> | 模式 | `attn_mode` 值 | 说明 |
> |---|---|---|
> | **训练** | `"flex"` | 训练必须使用该值。**不能**用于推理。 |
> | **推理 / 评估** | `"torch"` 或 `"flashattn"` | 推理必须使用该值。若仍使用 `"flex"`，评估时会报错。 |
>
> **修改方式：** 打开 `<your-model-path>/transformer/config.json`，找到 `"attn_mode"` 字段，并将其设置为对应模式所需的值。

---

## 部署 LingBot-VA 进行推理

LingBot-VA 同时支持独立运行模式和 Server-Client 架构。后者将模型环境与仿真环境解耦，能够避免依赖冲突，并支持在 GPU、集群及其他设备上进行分布式推理。

### 在 RoboTwin-2.0 上评估

**准备环境**

你可以参考 RoboTwin-2.0 原始仓库提供的官方说明：  
[https://robotwin-platform.github.io/doc/usage/robotwin-install.html](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)

简要步骤如下：

1.
```bash
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
```

2.
```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git && cd RoboTwin && git checkout 2eeec322
```

3. 修改 `script/requirements.txt`
```bash
transforms3d==0.4.2
sapien==3.0.0b1
scipy==1.10.1
mplib==0.2.1
gymnasium==0.29.1
trimesh==4.4.3
open3d==0.18.0
imageio==2.34.2
pydantic
zarr
openai
huggingface_hub==0.36.2
h5py
# For Description Generation
azure==4.0.0
azure-ai-inference
pyglet<2
wandb
moviepy
imageio
termcolor
av
matplotlib
ffmpeg
```

4. 修改 `script/_install.sh` 的第 8 行：
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
```

5. 运行：
```bash
bash script/_install.sh
```

6. 运行：
```bash
bash script/_download_assets.sh
```

**部署推理服务端**
```bash
# 单 GPU
bash evaluation/robotwin/launch_server.sh

# 多 GPU
bash evaluation/robotwin/launch_server_multigpus.sh
```

**执行推理客户端**
```bash
# 单 GPU
task_name="adjust_bottle";
save_root="results/";
bash evaluation/robotwin/launch_client.sh ${save_root} ${task_name}

# 多 GPU
save_root="results/"
task_group_id=0;
bash evaluation/robotwin/launch_client_multigpus.sh ${save_root} ${task_group_id}
```

实验结果会保存在 `/path/to/your/RoboTwin/${save_root}`。需要注意的是，系统还会额外生成一个 `eval_result` 文件夹，这是 RoboTwin 原生输出，与 `results` 中内容一致，可以忽略。

还需要注意：
- 推理服务端与客户端必须部署在同一台机器上。
- 在多 GPU 客户端启动脚本中，作者将原始 50 个任务通过复制填充到 56 个，并划分为 7 组，以适配其 8-GPU 推理节点。
- 你可以通过 `task_group_id`（0-6）选择某一组任务执行推理。
- 具体分组方式可参考 `evaluation/robotwin/launch_client_multigpus.sh`。

> **显存需求**：在启用 offload 模式（VAE 与 text_encoder 卸载到 CPU）时，单 GPU RoboTwin 评估约需 **24GB VRAM**。

### 运行图像到视频-动作生成

我们也提供了一个图像到视频-动作生成脚本：

```bash
NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh
```

> **显存需求**：在启用 offload 模式（VAE 与 text_encoder 卸载到 CPU）时，单 GPU 的 i2av 推理约需 **18GB VRAM**。

## 对 LingBot-VA 进行后训练

我们支持在自定义机器人操作数据集上对 LingBot-VA 进行后训练（微调）。训练流程使用 FSDP 做分布式训练，并兼容 [LeRobot](https://github.com/huggingface/lerobot) 数据集格式。

### 额外依赖

在基础安装之外，后训练还需要：

```bash
pip install lerobot==0.3.3 scipy wandb --no-deps
```

### 数据准备

从 HuggingFace 下载后训练数据集：

```bash
huggingface-cli download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot --local-dir /path/to/your/dataset
```

### 自定义数据集准备

如果你希望在自己的机器人操作数据上微调 LingBot-VA，请按下面步骤准备数据。

#### 示例数据集

我们提供了一个基于 [Issue #29](https://github.com/Robbyant/lingbot-va/issues/29) 数据转换而来的示例数据集。该数据集已经被转换为训练所需格式，可直接作为参考来理解目标数据结构。

- **下载地址**：[Example Dataset](https://drive.google.com/file/d/1D52nK4ZOJmWBXKv1nWrLb9YBwq8nKa_b/view?usp=sharing)

你可以将它作为模板，参考如何把自己的机器人操作数据转换成正确格式。

#### 数据处理流程概览

准备自定义数据集时，整体流程如下：

1. **原始数据** → 转换成 LeRobot 格式（包含元数据和视频文件）
2. **添加动作分段** → 在 `episodes.jsonl` 中加入 `action_config`
3. **提取 latent** → 根据视频规格用 VAE 处理视频
4. **数据集加载** → 按训练要求的动作维度加载处理后的数据

最终数据需要满足以下规范：

**动作格式：**
- 输出维度固定为 **30 维**，结构如下：
  - 左臂末端执行器（EEF）：7 维
  - 右臂末端执行器（EEF）：7 维
  - 左臂关节：7 维
  - 右臂关节：7 维
  - 左夹爪：1 维
  - 右夹爪：1 维
- 在数据集类加载器中，需要把你自己的机器人动作维度映射到这个标准的 30 维格式。
- 缺失的维度用 **0** 补齐。

**视频格式：**
- 在 VAE latent 提取阶段，建议把视频缩放到 **约 256 × 256 像素**，并下采样到 **5-15 fps**，可根据任务实际需求调整。

#### 实现步骤

**步骤 1：将原始数据转换为 LeRobot 格式**

参考官方 [LeRobot 数据集文档](https://github.com/huggingface/lerobot/tree/v0.3.3)，把原始数据（例如 HDF5、视频文件等）转换为标准 LeRobot 数据集格式。确保每个 episode 都包含所需的观测视频、动作和元数据。

**步骤 2：为 `episodes.jsonl` 添加 `action_config` 字段**

转换完成后，你需要修改 `meta/episodes.jsonl`，为其中每一行增加一个 `action_config` 字段。这个字段用于描述该 episode 内机器人的动作时间分段，以及每一段对应的自然语言描述。

`episodes.jsonl` 中每一行应符合如下格式：

```json
{
  "episode_index": 0,
  "tasks": ["task description"],
  "length": 450,
  "action_config": [
    {
      "start_frame": 0,
      "end_frame": 450,
      "action_text": "Natural language description of the robot action in this segment."
    }
  ]
}
```

- `start_frame` / `end_frame`：该动作分段在 episode 中的帧范围（从 0 开始计数）。
- `action_text`：该动作分段对应的自然语言描述。

如果一个 episode 只有单一连续动作，那么 `start_frame` 应为 `0`，`end_frame` 应等于该 episode 的 `length`。如果数据中包含连续子任务，也可以为同一个 episode 定义多个分段。

**步骤 3：使用 Wan2.2 VAE 提取视频 latent**

LingBot-VA 使用的是视频 latent 表示，而不是原始像素。因此你需要使用 Wan2.2 的 VAE 编码器提取 latent 特征，并将其放到转换后的 LeRobot 数据集目录中。具体如何运行 VAE 编码器，请参考 [Wan-Video 文档](https://github.com/Wan-Video)。

提取出的 latent 文件应存放在数据集目录中的 `latents/` 下，并与 `videos/` 的目录结构保持一致：

```text
your_dataset/
├── videos/
│   └── chunk-000/
│       └── observation.images.cam_high/
│           ├── episode_000000.mp4
│           └── ...
├── latents/
│   └── chunk-000/
│       └── observation.images.cam_high/
│           ├── episode_000000_0_450.pth
│           └── ...
└── meta/
    └── episodes.jsonl
```

每个 `.pth` 文件都是一个字典，包含以下字段：

| Key | 类型 | 说明 |
| :--- | :--- | :--- |
| `latent` | `Tensor [N, C]` (bfloat16) | 展平后的 VAE latent 特征，例如形状可为 `[latent_num_frames * latent_height * latent_width, C]` |
| `latent_num_frames` | `int` | latent 空间中的时间帧数 |
| `latent_height` | `int` | latent 空间中的高度 |
| `latent_width` | `int` | latent 空间中的宽度 |
| `video_num_frames` | `int` | 采样后源视频的帧数 |
| `video_height` | `int` | 原始视频高度（像素） |
| `video_width` | `int` | 原始视频宽度（像素） |
| `text_emb` | `Tensor [L, D]` (bfloat16) | 动作描述文本的 embedding，由 Wan2.2 text encoder 编码得到 |
| `text` | `str` | 原始动作描述文本 |
| `frame_ids` | `list[int]` | 按目标 fps 从原始 episode 中采样得到的帧索引 |
| `start_frame` | `int` | 与 `episodes.jsonl` 中 `action_config` 对齐的起始帧 |
| `end_frame` | `int` | 与 `episodes.jsonl` 中 `action_config` 对齐的结束帧 |
| `fps` | `int` | latent 提取时使用的目标采样 fps |
| `ori_fps` | `int` | episode 原始 fps |

latent 文件名格式 `episode_{index}_{start_frame}_{end_frame}.pth` 与 `episodes.jsonl` 中 `action_config` 定义的分段一一对应。例如，一个 episode 中若 `"start_frame": 0, "end_frame": 450`，则会生成名为 `episode_000000_0_450.pth` 的 latent 文件。

### 训练

```bash
NGPU=8 bash script/run_va_posttrain.sh
```

为了获得更好的训练效果，建议使用更大的全局 batch size，例如 32 或 64。如果 GPU 资源有限，可以通过增大 `gradient_accumulation_steps` 获得更大的等效 batch size。

---

# 📊 性能表现

我们在仿真基准和真实场景中都对模型进行了评估，并取得了当前最优性能。

## 仿真评估

- **RoboTwin 2.0**

我们是第一个将 RoboTwin 2.0 指标推进到 90+ 水平的方法。

<table style="border-collapse: collapse; width: auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; font-size: 13px; line-height: 1.2;">
  <p style="font-size: 12px; color: #666; margin-bottom: 5px;">* 所有指标均为百分比（%），数值越高越好，加粗表示最优。</p>
  <thead>
    <tr style="border-top: 2px solid black; border-bottom: 1px solid black;">
      <th align="left" style="padding: 6px 12px; white-space: nowrap;">方法（50 个任务平均）</th>
      <th align="center" style="padding: 6px 12px;">Easy SR (%)</th>
      <th align="center" style="padding: 6px 12px;">Hard SR (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 4px 12px; white-space: nowrap;">X-VLA</td>
      <td align="center">72.9</td>
      <td align="center">72.8</td>
    </tr>
    <tr>
      <td style="padding: 4px 12px; white-space: nowrap;">&pi;<sub>0</sub></td>
      <td align="center">65.9</td>
      <td align="center">58.4</td>
    </tr>
    <tr>
      <td style="padding: 4px 12px; white-space: nowrap;">&pi;<sub>0.5</sub></td>
      <td align="center">82.7</td>
      <td align="center">76.8</td>
    </tr>
    <tr>
      <td style="padding: 4px 12px; white-space: nowrap;">Motus</td>
      <td align="center"><u>88.7</u></td>
      <td align="center"><u>87.0</u></td>
    </tr>
    <tr style="border-top: 1px solid black; border-bottom: 2px solid black;">
      <td style="padding: 6px 12px; white-space: nowrap;"><b>LingBot-VA（我们的方法）</b></td>
      <td align="center"><b>92.9</b> <small>(+4.2)</small></td>
      <td align="center"><b>91.6</b> <small>(+4.6)</small></td>
    </tr>
  </tbody>
</table>

- **LIBERO**

<table style="border-collapse: collapse; width: auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; font-size: 13px; line-height: 1.2;">
  <p style="font-size: 12px; color: #666; margin-bottom: 5px;">* 所有指标均为百分比（%），数值越高越好，加粗表示最优。</p>
  <thead>
    <tr style="border-top: 2px solid black; border-bottom: 1px solid black;">
      <th align="left" style="padding: 6px 10px; border-right: 1px solid black; white-space: nowrap;">方法</th>
      <th align="center" style="padding: 6px 8px;">Spatial</th>
      <th align="center" style="padding: 6px 8px;">Object</th>
      <th align="center" style="padding: 6px 8px;">Goal</th>
      <th align="center" style="padding: 6px 8px;">Long</th>
      <th align="center" style="padding: 6px 8px;">Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 4px 10px; border-right: 1px solid black; white-space: nowrap;">&pi;<sub>0</sub></td>
      <td align="center">96.8</td><td align="center">98.8</td><td align="center">95.8</td><td align="center">85.2</td><td align="center">94.1</td>
    </tr>
    <tr>
      <td style="padding: 4px 10px; border-right: 1px solid black; white-space: nowrap;">&pi;<sub>0.5</sub></td>
      <td align="center">98.8</td><td align="center">98.2</td><td align="center">98.0</td><td align="center">92.4</td><td align="center">96.9</td>
    </tr>
    <tr>
      <td style="padding: 4px 10px; border-right: 1px solid black; white-space: nowrap;">OpenVLA</td>
      <td align="center">84.7</td><td align="center">88.4</td><td align="center">79.2</td><td align="center">53.7</td><td align="center">76.5</td>
    </tr>
    <tr>
      <td style="padding: 4px 10px; border-right: 1px solid black; white-space: nowrap;">X-VLA</td>
      <td align="center">98.2</td><td align="center">98.6</td><td align="center">97.8</td><td align="center">97.6</td><td align="center">98.1</td>
    </tr>
    <tr style="border-top: 1.5px solid black; border-bottom: 2px solid black;">
      <td style="padding: 5px 10px; border-right: 1px solid black; white-space: nowrap;"><b>LingBot-VA（我们的方法）</b></td>
      <td align="center"><b>98.5 &plusmn; 0.3</b></td>
      <td align="center"><b>99.6 &plusmn; 0.3</b></td>
      <td align="center"><b>97.2 &plusmn; 0.2</b></td>
      <td align="center"><b>98.5 &plusmn; 0.5</b></td>
      <td align="center"><b>98.5</b></td>
    </tr>
  </tbody>
</table>

&nbsp;

## 真实世界部署

我们在三类共六个真实操作任务上进行了评估，包括：
- 长时程任务：Make Breakfast、Pick Screws
- 精细操作任务：Insert Tube、Unpack Delivery
- 可变形 / 关节物体操作：Fold Clothes、Fold Pants

在每个任务仅使用 <b>50 次试验</b> 的前提下，我们的方法在 Progress Rate 和 Success Rate 两项指标上都取得了当前最优表现，并显著超过强基线 &pi;<sub>0.5</sub>。

<div style="text-align: left; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; line-height: 1.6;">
  <div style="margin-bottom: 5px;"><strong>Progress Score (PS)：</strong> 所有试验的平均进度分数除以最大可能分数，再转成百分比：</div>
  PS = Average_Progress / Max_Steps &times; 100%

  <div style="margin-bottom: 5px;"><strong>Success Rate (SR)：</strong> 成功试验次数除以总试验次数，再转成百分比：</div>
  SR = Successful_Trials / N &times; 100%
</div>

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;">
  <p style="font-size: 12px; color: #666; margin-bottom: 5px;">* 所有指标均为百分比（%），数值越高越好，加粗表示最优。</p>

  <table style="border-collapse: collapse; width: auto; font-size: 13px; line-height: 1.2;">
    <thead>
      <tr style="border-top: 2px solid black;">
        <th rowspan="2" align="left" style="padding: 4px 10px; border-bottom: 1px solid black; white-space: nowrap;"><b>任务</b></th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">Make Breakfast</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">Pick Screws</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">Insert Tube</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">Unpack Delivery</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">Fold Clothes</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">Fold Pants</th>
      </tr>
      <tr style="border-bottom: 1px solid black;">
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding: 6px 10px; white-space: nowrap;">&pi;<sub>0.5</sub></td>
        <td align="center">73.0</td><td align="center">70.0</td>
        <td align="center">74.0</td><td align="center">50.0</td>
        <td align="center">79.2</td><td align="center">30.0</td>
        <td align="center">73.0</td><td align="center">25.0</td>
        <td align="center"><b>62.9</b></td><td align="center">30.0</td>
        <td align="center">30.0</td><td align="center">30.0</td>
      </tr>
      <tr style="border-bottom: 2px solid black;">
        <td style="padding: 6px 10px; white-space: nowrap;"><b>LingBot-VA（我们的方法）</b></td>
        <td align="center"><b>97.0</b></td><td align="center"><b>75.0</b></td>
        <td align="center"><b>82.5</b></td><td align="center"><b>70.0</b></td>
        <td align="center"><b>85.8</b></td><td align="center"><b>40.0</b></td>
        <td align="center"><b>84.5</b></td><td align="center"><b>65.0</b></td>
        <td align="center">48.8</td><td align="center"><b>35.0</b></td>
        <td align="center"><b>76.7</b></td><td align="center"><b>70.0</b></td>
      </tr>
    </tbody>
  </table>
</div>

# 🪪 许可证

本项目基于 Apache License 2.0 发布。详情请见 [LICENSE](LICENSE.txt)。

# 📚 引用

```bibtex
@article{lingbot-va2026,
  title={Causal World Modeling for Robot Control},
  author={Li, Lin and Zhang, Qihang and Luo, Yiming and Yang, Shuai and Wang, Ruilin and Han, Fei and Yu, Mingrui and Gao, Zelin and Xue, Nan and Zhu, Xing and Shen, Yujun and Xu, Yinghao},
  journal={arXiv preprint arXiv:2601.21998},
  year={2026}
}
```
