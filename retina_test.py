from retina_face import RetinaFace
import onnxruntime

det = RetinaFace(model_file='models/det_10g.onnx')
det._init_vars()