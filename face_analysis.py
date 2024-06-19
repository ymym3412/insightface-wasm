from face import Face


class FaceAnalysis:
    def __init__(
        self,
        # model_path_det='models/det_10g.onnx',
        # session_det=None,
        # model_path_recog='models/w600k_r50.onnx',
        # session_recog=None,
        model_det,
        model_recog,
        **kwargs
    ):
        # onnxruntime.set_default_logger_severity(3)
        self.models = {}
        # self.models['detection'] = RetinaFace(model_file=model_path_det, session=session_det)
        # self.models['recognition'] = ArcFaceONNX(model_path_recog, session=session_recog)
        self.models['detection'] = model_det
        self.models['recognition'] = model_recog
        self.det_model = self.models['detection']
        # self.det_model = self.models["detection"]

    async def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print("set det-size:", det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == 'detection':
                await model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                await model.prepare(ctx_id)

    async def get(self, img, max_num=0):
        bboxes, kpss = await self.det_model.detect(img, max_num=max_num, metric="default")
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue
                await model.get(img, face)
            ret.append(face)
        return ret

    async def draw_on(self, img, faces):
        import cv2

        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(int)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
            if face.gender is not None and face.age is not None:
                cv2.putText(
                    dimg,
                    "%s,%d" % (face.sex, face.age),
                    (box[0] - 1, box[1] - 4),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 255, 0),
                    1,
                )

            # for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return await dimg

print('fa final')
