import cv2
from face import Face
import numpy
from typing import Any, List, Optional
import threading
import traceback

import os
from face_analysis import FaceAnalysis
from swapper import INSwapper
from retina_face import RetinaFace
from arcface import ArcFaceONNX
from front_model import RetinaFaceBrowse, ArcFaceONNXBrowse, INSwapperBrowse


Frame = numpy.ndarray[Any, Any]

FACE_SWAPPER = None
FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()
SIMILAR_FACE_DISTANCE = 0.85
ALLOW_MANY_FACE = False

# Face analyzer
async def get_face_analyser() -> Any:
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
                await det.init_session()
                recog = ArcFaceONNXBrowse('models/w600k_r50.onnx')
                await recog.init_session()
                FACE_ANALYSER = FaceAnalysis(det, recog)
            await FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER


async def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        analyzer = await get_face_analyser()
        result = await analyzer.get(frame)
        return result
    except Exception as e:
        traceback.print_exc()
        await 1 == 1
        raise e
        # return None


async def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    try:
        many_faces = await get_many_faces(frame)
        if many_faces:
            return many_faces[position]
    except IndexError:
        traceback.print_exc()
        await 1 == 1
        return many_faces[-1]
    except ValueError as e:
        traceback.print_exc()
        await 1 == 1
        raise e
    return None


async def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    try:
        many_faces = await get_many_faces(frame)
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
    except Exception as e:
        traceback.print_exc()
        raise e


# util
def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


# face_swapper
async def get_face_swapper() -> Any:
    global FACE_SWAPPER
    from local import ENV

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = "models/inswapper_128.onnx"
            if ENV == 'LOCAL':
            # FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=globals.execution_providers)
                FACE_SWAPPER = INSwapper(model_path)
            else:
                FACE_SWAPPER = INSwapperBrowse(model_path)
                await FACE_SWAPPER.init_session()
    return FACE_SWAPPER


async def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    swapper = await get_face_swapper()
    res = await swapper.get(temp_frame, target_face, source_face, paste_back=True)
    return res


async def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    target_face = await find_similar_face(temp_frame, reference_face)
    if target_face:
        temp_frame = await swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if ALLOW_MANY_FACE else get_one_face(target_frame, 0)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)


async def process_image2(source: numpy.array, target: numpy.array) -> None:
    source_face = await get_one_face(source)
    # target_frame = cv2.imread(target)
    target_frame = target
    reference_face = await get_one_face(target_frame, 0)
    result = await process_frame(source_face, reference_face, target_frame)
    return result
