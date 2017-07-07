from unittest import SkipTest

import numpy as np

from test.util import generate_kernel_test_case
from webdnn.graph.graph import Graph
from webdnn.graph.operators.leaky_relu import LeakyRelu
from webdnn.graph.order import OrderNHWC, OrderNCHW
from webdnn.graph.variable import Variable
from webdnn.graph.variables.constant_variable import ConstantVariable

condition_default = {
    "x_order": OrderNHWC,
    "y_order": OrderNHWC,
    "backend": ["webgpu", "webassembly", "fallback"],
    "slope": 0.5
}


def test_general():
    for condition_custom in [
        {},
        {"slope": 0},
        {"slope": 1},
        {"x_order": OrderNCHW},
    ]:
        condition = dict(condition_default)
        condition.update(condition_custom)

        vx = np.random.rand(2, 3, 4, 5) - 0.5
        vy = np.maximum(vx, vx * condition["slope"])

        x = Variable(vx.shape, order=OrderNHWC)
        y, = LeakyRelu(None, slope=condition["slope"])(x)

        x.change_order(condition["x_order"])
        y.change_order(condition["y_order"])

        generate_kernel_test_case(
            description=f"ScalarAffine: " + (", ".join([f"{k}={v}" for k, v in condition_custom.items()])),
            backend=condition["backend"],
            graph=Graph([x], [y]),
            inputs={x: ConstantVariable(vx, OrderNHWC).change_order(x.order).data},
            expected={y: ConstantVariable(vy, OrderNHWC).change_order(y.order).data},
            raise_skip=False
        )

    raise SkipTest
