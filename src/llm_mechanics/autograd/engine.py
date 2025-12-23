# Autograd Engine
# This engine is a simple implementation of the autograd engine.


class Value:
    def __init__(self, data, _children=(), _op=""):
        if isinstance(data, Value):
            data = data.data
        self.data = data
        self._children = set(_children)
        self._op = _op
        self.grad = 0.0
        self.variables = {}
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, children={self._children}, op={self._op})"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    # 当出现 other + self 时调用（例如 0 + Value）
    def __radd__(self, other):
        return self + other  # 加法满足交换律，直接调换顺序调用 __add__

    # 实现 -self
    def __neg__(self):
        return self * -1

    # 实现 self - other
    def __sub__(self, other):
        return self + (-other)  # 变成 self + (other * -1)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __pow__(self, other):
        # 处理 x**2 这种情况，暂时只支持作为常数的 exponent
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            # 链式法则：grad += (n * x^(n-1)) * out.grad
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def relu(self):
        out_data = self.data if self.data > 0 else 0.0
        out = Value(out_data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out
