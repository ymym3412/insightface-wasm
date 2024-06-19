import numpy as np
import cv2
from utils import norm_crop
# from pyodide.ffi import create_proxy, to_js
try:
    import onnxruntime
except:
    # print('arcface')
    print('no onnxruntime')


class ArcFaceONNX:
    def __init__(self, model_file=None, session=None):
        print('arcface init')
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        self.taskname = "recognition"
        # find_sub = False
        # find_mul = False
        # model = onnx.load(self.model_file)
        # graph = model.graph
        # for nid, node in enumerate(graph.node[:8]):
            # print(nid, node.name)
            # if node.name.startswith("Sub") or node.name.startswith("_minus"):
                # find_sub = True
            # if node.name.startswith("Mul") or node.name.startswith("_mul"):
                # find_mul = True
        if False:
            # mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        # print('input mean and std:', self.input_mean, self.input_std)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)

    async def init(self):
        # input_cfg = self.session.get_inputs()[0]
        # 先頭がNoneなのはバッチサイズが不定ということ
        # 1枚ずつ必ず処理するなら1でもよい(はず)
        input_shape = [None, 3, 112, 112]
        input_name = 'input.1'
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        # outputs = self.session.get_outputs()
        output_names = ['683']
        # for out in outputs:
            # output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1
        # self.output_shape = outputs[0].shape
        self.output_shape = [1, 512]

    async def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            await self.session.set_providers(["CPUExecutionProvider"])

    async def get(self, img, face):
        aimg = norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        v = await self.get_feat(aimg)
        face.embedding = v.flatten()
        return face.embedding

    async def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm

        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return await sim

    async def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size

        blob = cv2.dnn.blobFromImages(
            imgs,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        # onnxruntime-web
        # net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        net_out = await self.run_session(blob)
        return net_out

    async def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        # onnxruntime-web
        # net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        net_out = await self.run_session(blob)
        return net_out

    async def run_session(self, input_tensor):
        net_out = await self.session.run(self.output_names, {self.input_name: input_tensor})
        return net_out

print('arcface final')
