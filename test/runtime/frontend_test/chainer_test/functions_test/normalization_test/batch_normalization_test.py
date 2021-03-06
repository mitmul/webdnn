from unittest import SkipTest

import chainer
import numpy as np

from test.util import generate_kernel_test_case, wrap_template
from webdnn.frontend.chainer.converter import ChainerConverter


@wrap_template
def template(description=""):
    if chainer.__version__ >= "3.":
        raise SkipTest("Since Chainer 3.0.0, L.BatchNormalization use F.fixed_batch_normalization when 'chainer.config.train == False'.")

    link = chainer.links.BatchNormalization(size=4)
    vx = chainer.Variable(np.random.rand(2, 4, 6, 8).astype(np.float32))

    if chainer.__version__ >= "2.":
        with chainer.using_config('train', False):
            vy = link(vx)

    else:
        vy = link(vx, test=True)

    graph = ChainerConverter().convert([vx], [vy])

    x = graph.inputs[0]
    y = graph.outputs[0]

    generate_kernel_test_case(
        description=f"[chainer] L.BatchNormalization {description}",
        graph=graph,
        inputs={x: vx.data},
        expected={y: vy.data}
    )


def test():
    template()
