from face_analysis import FaceAnalysis
from swapper import INSwapper
from arcface import ArcFaceONNX
from retina_face import RetinaFace

import time
import numpy as np
from pyodide.ffi import create_proxy, to_js
from js import js_init_session, js_run_retina_session, js_run_arcface_session, js_run_swap_session


class FaceAnalysisBrowse(FaceAnalysis):
    async def __init__(self,  model_path_det='models/det_10g.onnx', session_det=None, model_path_recog='models/w600k_r50.onnx', session_recog=None):
        self.init_session(model_path_det, model_path_recog)
        super().__init__(model_path_det, self.session_det, model_path_recog, self.session_recog)

    async def init_session(self, model_path_det, model_path_recog):
        self.session_det = await js_init_session(model_path_det)
        self.session_recog = await js_init_session(model_path_recog)


class INSwapperBrowse(INSwapper):
    def __init__(self, model_file, session=None):
        super().__init__(model_file, [])

    async def init_session(self):
        self.session = await js_init_session(self.model_file)

    async def run_session(self, target_tensor, source_tensor):
        start = time.perf_counter()
        outputs_jsproxy = await js_run_swap_session(to_js(np.array(target_tensor)), to_js(np.array(source_tensor)))
        output_info = {k: np.array(v) for k, v in outputs_jsproxy.to_py().items()}
        outputs = output_info['output'].reshape(output_info['dims'])
        end = time.perf_counter()
        print('Swapper onnx session time: {}'.format(end - start))
        return outputs


class ArcFaceONNXBrowse(ArcFaceONNX):
    def __init__(self, model_file, session=None):
        # self.init_session(model_file)
        super().__init__(model_file, [])

    async def init_session(self):
        self.session = await js_init_session(self.model_file)
        await self.init()

    async def run_session(self, input_tensor):
        start = time.perf_counter()
        outputs_jsproxy = await js_run_arcface_session(to_js(np.array(input_tensor)))
        output_info = {k: np.array(v) for k, v in outputs_jsproxy.to_py().items()}
        outputs = output_info['683'].reshape(output_info['dims'])
        end = time.perf_counter()
        print('Arcface onnx session time: {}'.format(end - start))
        return outputs


class RetinaFaceBrowse(RetinaFace):
    def __init__(self, model_file, session=None):
        # self.init_session(model_file)
        super().__init__(model_file, [])

    async def init_session(self):
        _session = await js_init_session(self.model_file)
        self.session = _session.to_py()
        await self._init_vars()

    async def run_session(self, input_tensor):
        start = time.perf_counter()
        outputs_jsproxy = await js_run_retina_session(to_js(np.array(input_tensor)))
        output_dict = outputs_jsproxy.to_py()
        tensor_order = ['448', '471', '494', '451', '474', '497', '454', '477', '500']
        # output_info = {k: np.array(v) for k, v in output_dict.items() if k != 'dims'}
        dim_dict = output_dict['dims']
        # outputs = [np.array(v) for v in outputs_jsproxy.to_py().values()]
        outputs = []
        for key in tensor_order:
            outputs.append(np.array(output_dict[key]).reshape(dim_dict[key]))
        end = time.perf_counter()
        print('Retina onnx session time: {}'.format(end - start))
        return outputs
