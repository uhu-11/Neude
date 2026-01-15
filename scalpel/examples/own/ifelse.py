import numpy as np
import numpy as np
def img_info(img: list):
    assert isinstance(img, list) and len(img) == 255 and isinstance(img[0], list) and len(img[0]) == 255
    img = np.array(img)
    return img.shape

example_img = [[[0] for _ in range(255)] for _ in range(255)]


if __name__ == "__main__":
    print(img_info(example_img))