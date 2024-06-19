import cv2
from face import Face
import numpy
from typing import Any, List, Optional
import threading

# import insightface

# import globals
import os
from face_analysis import FaceAnalysis
from swapper import INSwapper
from front_model import RetinaFaceBrowse, ArcFaceONNXBrowse, INSwapperBrowse
from retina_face import RetinaFace
from arcface import ArcFaceONNX

Frame = numpy.ndarray[Any, Any]

FACE_SWAPPER = None
FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()
SIMILAR_FACE_DISTANCE = 0.85
ALLOW_MANY_FACE = False


# Face analyzer
def get_face_analyser() -> Any:
    global FACE_ANALYSER
    from local import ENV

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            # FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=globals.execution_providers)
            if ENV == 'LOCAL':
                det = RetinaFace(model_file='models/det_10g.onnx')
                recog = ArcFaceONNX('models/w600k_r50.onnx')
                FACE_ANALYSER = FaceAnalysis(det, recog)
            else:
                det = RetinaFaceBrowse(model_file='models/det_10g.onnx')
                recog = ArcFaceONNXBrowse('models/w600k_r50.onnx')
                FACE_ANALYSER = FACE_ANALYSER(det, recog)
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER


def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        return get_face_analyser().get(frame)
    except ValueError:
        return None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None


def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            if hasattr(face, "normed_embedding") and hasattr(
                reference_face, "normed_embedding"
            ):
                distance = numpy.sum(
                    numpy.square(
                        face.normed_embedding - reference_face.normed_embedding
                    )
                )
                if distance < SIMILAR_FACE_DISTANCE:
                    return face
    return None


# util
def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


# face_swapper
def get_face_swapper() -> Any:
    global FACE_SWAPPER
    from local import ENV

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path("models/inswapper_128.onnx")
            if ENV == 'LOCAL':
            # FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=globals.execution_providers)
                FACE_SWAPPER = INSwapper(model_path)
            else:
                FACE_SWAPPER = INSwapperBrowse(model_path)
    return FACE_SWAPPER


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    target_face = find_similar_face(temp_frame, reference_face)
    if target_face:
        temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if ALLOW_MANY_FACE else get_one_face(target_frame, 0)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)


def process_image2(source: str, target: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source))
    # target_frame = cv2.imread(target)
    target_frame = target
    reference_face = None if ALLOW_MANY_FACE else get_one_face(target_frame, 0)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)
