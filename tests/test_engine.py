from llm_mechanics.autograd.engine import Value


def test_engine():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    e = a * b
    z = e + c

    # 运行反向传播
    z.backward()

    assert z.data == 4.0
    assert a.grad == -3.0
    assert b.grad == 2.0
    assert c.grad == 1.0


def test_relu():
    # Case 1: 正数，梯度应该透传
    a = Value(2.0)
    b = a.relu()
    b.backward()
    assert b.data == 2.0
    assert a.grad == 1.0

    # Case 2: 负数，梯度应该为0
    c = Value(-10.0)
    d = c.relu()
    d.backward()
    assert d.data == 0.0
    assert c.grad == 0.0  # 梯度被“杀”死了
