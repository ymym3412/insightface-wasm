import numpy as np
import cv2
from utils import norm_crop2
import pickle
# from pyodide.ffi import create_proxy, to_js
try:
    import onnxruntime
except:
    # print('swapper')
    print('no onnxruntime')


class INSwapper:
    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        # model = onnx.load(self.model_file)
        # graph = model.graph  
        with open('emap.pkl', 'rb') as f:
            emap = pickle.load(f)
        # self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.emap = emap
        self.input_mean = 0.0
        self.input_std = 255.0
        # print('input mean and std:', model_file, self.input_mean, self.input_std)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        # inputs = self.session.get_inputs()
        self.input_names = ['target', 'source']
        # for inp in inputs:
            # self.input_names.append(inp.name)
        # outputs = self.session.get_outputs()
        # for out in outputs:
            # output_names.append(out.name)
        self.output_names = ['output']
        assert len(self.output_names) == 1
        # output_shape = outputs[0].shape
        # input_cfg = inputs[0]
        input_shape = [1, 3, 128, 128]
        self.input_shape = input_shape
        print("inswapper-shape:", self.input_shape)
        self.input_size = tuple(input_shape[2:4][::-1])

    async def forward(self, img, latent):
        img = (img - self.input_mean) / self.input_std

        # onnxruntime-web
        pred = await self.run_session(img, latent)
        # pred = self.session.run(
            # self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent}
        # )[0]
        return pred

    async def get(self, img, target_face, source_face, paste_back=True):
        aimg, M = norm_crop2(img, target_face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)

        # onnxruntime-web
        pred = await self.run_session(blob, latent)
        # pred = self.session.run(
            # self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent}
        # )[0]

        # print(latent.shape, latent.dtype, pred.shape)
        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        if not paste_back:
            return bgr_fake, M
        else:
            target_img = img
            fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
            fake_diff = np.abs(fake_diff).mean(axis=2)
            fake_diff[:2, :] = 0
            fake_diff[-2:, :] = 0
            fake_diff[:, :2] = 0
            fake_diff[:, -2:] = 0
            IM = cv2.invertAffineTransform(M)
            img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
            bgr_fake = cv2.warpAffine(
                bgr_fake,
                IM,
                (target_img.shape[1], target_img.shape[0]),
                borderValue=0.0,
            )
            img_white = cv2.warpAffine(
                img_white,
                IM,
                (target_img.shape[1], target_img.shape[0]),
                borderValue=0.0,
            )
            fake_diff = cv2.warpAffine(
                fake_diff,
                IM,
                (target_img.shape[1], target_img.shape[0]),
                borderValue=0.0,
            )
            img_white[img_white > 20] = 255
            fthresh = 10
            fake_diff[fake_diff < fthresh] = 0
            fake_diff[fake_diff >= fthresh] = 255
            img_mask = img_white
            mask_h_inds, mask_w_inds = np.where(img_mask == 255)
            mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
            mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
            mask_size = int(np.sqrt(mask_h * mask_w))
            k = max(mask_size // 10, 10)
            # k = max(mask_size//20, 6)
            # k = 6
            kernel = np.ones((k, k), np.uint8)
            img_mask = cv2.erode(img_mask, kernel, iterations=1)
            kernel = np.ones((2, 2), np.uint8)
            fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
            k = max(mask_size // 20, 5)
            # k = 3
            # k = 3
            kernel_size = (k, k)
            blur_size = tuple(2 * i + 1 for i in kernel_size)
            img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
            k = 5
            kernel_size = (k, k)
            blur_size = tuple(2 * i + 1 for i in kernel_size)
            fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
            img_mask /= 255
            fake_diff /= 255
            # img_mask = fake_diff
            img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
            fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(
                np.float32
            )
            fake_merged = fake_merged.astype(np.uint8)
            return fake_merged

    async def run_session(self, target_tensor, source_tensor):
        pred = await self.session.run(
            self.output_names, {self.input_names[0]: target_tensor, self.input_names[1]: source_tensor}
        )
        return pred

print('swapper done')
