import random

from llm_mechanics.autograd.engine import Value
from llm_mechanics.nn import MLP


def test_mlp():
    random.seed(42)

    # 1. 定义一个小型网络
    # 输入维度 3 (例如：身高、体重、年龄)
    # 隐藏层有两个，分别是 4 个神经元、4 个神经元
    # 输出层 1 个神经元 (例如：预测寿命)
    model = MLP(3, [4, 4, 1])

    # 2. 伪造一条输入数据
    # 注意：我们这里没用 Tensor，直接用 list of values
    x = [Value(2.0), Value(3.0), Value(-1.0)]

    # 3. 前向传播 (Forward Pass)
    pred = model(x)

    print("模型结构:", model)
    print("预测结果:", pred)

    # 4. 反向传播 (Backward Pass)
    # 这一步会瞬间计算出网络中所有权重 (w) 和偏置 (b) 的梯度！
    pred.backward()

    # 5. 随便挑一个参数看看它的梯度
    first_layer_weight = model.layers[0].neurons[0].w[0]
    print(f"第一个权重的数值: {first_layer_weight.data}")
    print(f"第一个权重的梯度: {first_layer_weight.grad}")

    assert first_layer_weight.data == 0.2788535969157675
    assert first_layer_weight.grad == 0.0
