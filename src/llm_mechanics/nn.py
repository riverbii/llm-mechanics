import random

from llm_mechanics.autograd.engine import Value


class Module:
    """所有神经网络模块的基类 (PyTorch 风格)"""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """返回该模块包含的所有可训练参数 (Value 对象)"""
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        """
        nin: number of inputs (输入维数)
        nonlin: 是否应用非线性激活 (ReLU)
        """
        # 1. 随机初始化权重 (-1 到 1 之间)
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # 2. 初始化偏置 (通常初始化为 0 或很小的数)
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        """前向传播逻辑: w * x + b"""
        # 这里的 zip(self.w, x) 把权重和输入配对
        # sum(...) 会调用我们写的 Value.__add__
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # 应用非线性激活
        return act.relu() if self.nonlin else act

    def parameters(self):
        # 所有的 w 和 b 都是需要更新的参数
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        """
        nin: 输入维数 (即上一层有多少个神经元)
        nout: 输出维数 (即这一层有多少个神经元)
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        # 让每个神经元都处理一下输入 x
        out = [n(x) for n in self.neurons]
        # 如果只有一个输出，直接返回标量，方便处理
        return out[0] if len(out) == 1 else out

    def parameters(self):
        # 收集这一层所有神经元的参数
        # 列表推导式写法: [p for n in neurons for p in n.parameters()]
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts):
        """
        nin: 输入数据的维数
        nouts: 一个列表，定义每一层的大小。例如 [16, 16, 1]
        """
        sz = [nin] + nouts  # [输入维, 第一层, 第二层, ...]

        # 这里的逻辑很精妙：
        # 第 i 层的输入是 sz[i]，输出是 sz[i+1]
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]
        # 注意：最后一层通常是输出层，如果是回归任务，有时候不加 ReLU，
        # 所以我在 Layer/Neuron 里加了个 nonlin 参数控制。
        # 这里为了简化，我们假设除了最后一层，前面都加 ReLU。

    def __call__(self, x):
        # 数据像流水一样流过每一层
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        return f"MLP of [{', '.join(str(la) for la in self.layers)}]"
