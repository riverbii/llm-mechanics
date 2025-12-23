# llm-mechanics

从0开始实现LLM 大模型核心机制，包含自动求导引擎、MLP 组件，以及后续将逐步扩展的注意力、RoPE、KV Cache 等模块及可视化。

## 功能概览
- 轻量级自动求导引擎 `Value`，支持加减乘除、幂运算、ReLU 与反向传播。
- PyTorch 风格的 `Module`/`Neuron`/`Layer`/`MLP` 实现，可直接构建多层感知机。
- 配套测试用例，方便验证前向/反向传播的正确性。

## 快速开始
1) 准备环境（Python 3.12+）
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

2) 运行测试
```bash
pytest -q
```

3) 体验自动求导与 MLP
```python
from llm_mechanics.autograd.engine import Value
from llm_mechanics.nn import MLP

# 自动求导示例
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
z = a * b + c
z.backward()
print(z.data, a.grad, b.grad, c.grad)  # -> 4.0 -3.0 2.0 1.0

# MLP 前向与反向
model = MLP(3, [4, 4, 1])
x = [Value(2.0), Value(3.0), Value(-1.0)]
pred = model(x)
pred.backward()
print(pred.data, model.layers[0].neurons[0].w[0].grad)
```

## 项目结构
- `src/llm_mechanics/autograd/engine.py`：核心 `Value` 自动求导逻辑。
- `src/llm_mechanics/nn.py`：基础神经网络模块与 MLP 实现。
- `tests/`：针对自动求导与 MLP 的最小化回归测试。

## 开发与规范
- 代码格式与静态检查：`ruff check .`、`ruff format .`
- 提交前可运行 `pre-commit run --all-files`（若已安装 hook）
