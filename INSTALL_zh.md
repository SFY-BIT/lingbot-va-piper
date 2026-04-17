# 安装指南

## 使用 pip 安装

```bash
pip install .
pip install .[dev]  # 同时安装开发工具
```

## 使用 Poetry 安装

请先确认你的系统中已经安装了 [Poetry](https://python-poetry.org/docs/#installation)。

安装全部依赖：

```bash
poetry install
```

### 处理 `flash-attn` 安装问题

如果 `flash-attn` 因 **PEP 517 构建问题** 安装失败，可以尝试以下方法。

#### 关闭 Build Isolation 安装（推荐）

```bash
poetry run pip install --upgrade pip setuptools wheel
poetry run pip install flash-attn --no-build-isolation
poetry install
```

#### 从 Git 直接安装（备选方案）

```bash
poetry run pip install git+https://github.com/Dao-AILab/flash-attention.git
```
find /mnt/hdd/sfy/models/lingbot-va-base -path "*/transformer/config.json"

---

### 运行模型

安装完成后，你可以通过下面的命令运行 **Wan2.2**：

```bash
poetry run python generate.py --task t2v-A14B --size '1280*720' --ckpt_dir ./Wan2.2-T2V-A14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

#### 测试

```bash
bash tests/test.sh
```

#### 代码格式化

```bash
black .
isort .
```
