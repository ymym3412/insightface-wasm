img1_path = "images/kimutaku.jpg"
img2_path = "images/out.jpg"
result_path = "images/result.jpg"
ENV = 'front'

if __name__ == '__main__':
    from faceswap import process_image
    process_image(img1_path, img2_path, result_path)
