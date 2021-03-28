# Creating the TensorRT engine from ONNX

import tensorrt as trt


def build_engine(TRT_LOGGER, onnx_path, shape = [1,224,224,3]):
    """
    This is the function to create the TensorRT engine
    Args:
       onnx_path : Path to onnx_file. 
       shape : Shape of the input of the ONNX file. 
   """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = (256 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        # use FP16 mode if possible
        if builder.platform_has_fast_fp16:
            builder.fp16_mode = True

        # generate TensorRT engine optimized for the target platform
        engine = builder.build_cuda_engine(network)
        return engine

def save_engine(engine, engine_file_name):
    buf = engine.serialize()
    with open(engine_file_name, 'wb') as f:
        f.write(buf)
       
def load_engine(trt_runtime, engine_file_name):
    with open(engine_file_name, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

