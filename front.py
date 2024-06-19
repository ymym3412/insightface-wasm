import asyncio

async def main():
    # setup ----------------------------------------------------------------
    print('importing python packages...')
    import pyodide_js
    # import onnxruntime
    await pyodide_js.loadPackage(['opencv-python', "matplotlib", 'scikit-image'])
    import micropip
    await micropip.install('scikit-image')

    from pyodide.ffi import create_proxy
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from js import document, readFileAsDataURL
    import base64
    import traceback
    from faceswap_front import process_image2

    # functions ----------------------------------------------------------------
    async def swap_from_upload(e):
        print('detect_from_upload...')
        try:
            dataurl = await readFileAsDataURL()
            await swap_from_dataURL(dataurl)
        except e:
            traceback.print_stack()
            raise e

    async def swap_from_dataURL(dataURL: str):
        binary_data = base64.b64decode(dataURL.split(",")[1])
        img_data = np.frombuffer(binary_data, np.uint8)
        source = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        target = cv2.imread('images/out.jpg')
        print('swap from datauri')
        result = await process_image2(source, target)
        print('complete swap phase')
        print('Ploting image...')
        _, buffer = cv2.imencode('.jpg', result)
        jpg_as_text = base64.b64encode(buffer).decode()
        img_str = f'data:image/png;base64,{jpg_as_text}'
        img_tag = document.getElementById('image')
        img_tag.src = img_str
        print('Swap completed!')

    document.getElementById("fileInput").addEventListener(
        "change", create_proxy(swap_from_upload))

    # process_image(img1_path, img2_path, result_path)
    print("done")