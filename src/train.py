from llm_mechanics.autograd.engine import Value
from llm_mechanics.nn import MLP

# 训练数据 (Input)
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

# 目标结果 (Target)
ys = [1.0, -1.0, -1.0, 1.0]  # 期望输出

# 输入层 3 (对应 xs 的维度)，两个隐藏层 [4, 4]，输出层 1
n = MLP(3, [4, 4, 1])

# 训练 20 轮 (Epochs)
for k in range(500):
    # 1. Forward
    ypred = [n(x) for x in xs]
    ys = [Value(y) for y in ys]
    loss = sum([(yout - ygt) * (yout - ygt) for yout, ygt in zip(ypred, ys)])

    # 2. Backward
    n.zero_grad()  # 🔥 必须清零！
    loss.backward()

    # 3. Update
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(f"Epoch {k}: Loss = {loss.data:.4f}")

print("最终预测:", [y.data for y in ypred])
