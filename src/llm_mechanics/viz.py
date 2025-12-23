from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        display_label = "{ data %.4f | grad %.4f }" % (n.data, n.grad)
        dot.node(name=uid, label=display_label, shape="record")
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        uid1 = str(id(n1))
        uid2 = str(id(n2))
        if n2._op:
            dot.edge(uid1, uid2 + n2._op)
        else:
            dot.edge(uid1, uid2)

    return dot


# ✨ 支持高亮的绘图函数
def draw_step(root, active_node=None, visited_set=None, topo_list=None):
    """
    active_node: 当前正在递归访问的节点 (标红)
    visited_set: 已经进入递归栈的节点 (标灰)
    topo_list:   已经完成拓扑排序的节点 (标绿)
    """
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = trace(root)
    visited_set = visited_set or set()
    topo_list = topo_list or []

    for n in nodes:
        uid = str(id(n))

        # --- 🎨 颜色逻辑 ---
        fillcolor = "white"  # 默认：白色
        style = "filled"

        if n in topo_list:
            fillcolor = "#90ee90"  # 绿色：已完成排序 (Done)
        elif n == active_node:
            fillcolor = "#ffcccb"  # 红色：当前正在处理 (Active)
        elif n in visited_set:
            fillcolor = "#d3d3d3"  # 灰色：已访问但孩子还没处理完 (In Stack)

        display_label = "{ data %.2f | grad %.2f }" % (n.data, n.grad)

        dot.node(
            name=uid,
            label=display_label,
            shape="record",
            style=style,
            fillcolor=fillcolor,
        )

        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        uid1 = str(id(n1))
        uid2 = str(id(n2))

        # 如果是正在处理的边，也可以加粗 (可选)
        edge_color = "black"
        penwidth = "1"
        if n2 == active_node and n1 in visited_set:
            edge_color = "red"

        if n2._op:
            dot.edge(uid1, uid2 + n2._op, color=edge_color, penwidth=penwidth)
        else:
            dot.edge(uid1, uid2, color=edge_color, penwidth=penwidth)

    return dot
