

from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw, Image
from torchvision import transforms

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1

            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 1232, 1232]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path =  "./models_trained/797-AG-BC.onnx"
    engine_file_path =  "./models_trained/797-AG-BC.trt"
    onnx_file_path =  "./models_trained/544-CH-CA.onnx"
    engine_file_path =  "./models_trained/544-CH-CA.trt"


    # Download a dog image and save it to the following file path:

    input_image_path = "./imgs_prueba_clasificacion/AG_BC1.png"
    input_image_path = "./imgs_prueba_deteccion/CH_CA.png"


    imagen = Image.open(input_image_path)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    #loader = transforms.Compose(
        #transforms.Resize(128), transforms.ToTensor(), transforms.Normalize(std, mean)])
    loader = transforms.Compose([transforms.ToTensor(), transforms.Normalize(std, mean)])
    imagen = loader(imagen).float()
    #imagen = imagen.unsqueeze(0) 
    image = imagen.numpy()

      
    # Do inference with TensorRT
    trt_outputs = []

    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
       inputs, outputs, bindings, stream = common.allocate_buffers(engine)
       # Do inference
       print('Running inference on image {}...'.format(input_image_path))
       # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
       inputs[0].host = image
       trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
       print(trt_outputs)


if __name__ == '__main__':
    main()


