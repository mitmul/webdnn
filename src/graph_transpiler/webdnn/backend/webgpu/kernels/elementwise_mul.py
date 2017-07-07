from webdnn.backend.webgpu.kernels.elementwise import register_elementwise_kernel
from webdnn.graph.operators.elementwise_mul import ElementwiseMul

register_elementwise_kernel(ElementwiseMul, "y = x0 * x1;")
