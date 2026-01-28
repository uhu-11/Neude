import numpy as np
import cv2
import random
import albumentations as A


from skimage import util
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian
from io import BytesIO
import ctypes
from wand.image import Image as WandImage
from wand.api import library as wandlibrary

class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
class ImageMutationOperator:


    def ops1(self, buf):
        arr = np.array(buf)
        # 获取图像的形状
        height, width, channels = arr.shape  # 假设 buf 是一个形状为 (height, width, channels) 的数组
        # 计算总像素数量
        total_pixels = height * width
        # 确保 m 不超过总像素数量
        num = random.randint(total_pixels // 60, total_pixels // 10)

        # 生成不重复的随机索引
        random_indices = random.sample(range(total_pixels), num)
        # 修改选定的像素
        for index in random_indices:
            # 计算行和列
            row = index // width
            col = index % width

            # 重新赋值为随机值（0 到 255 之间）
            arr[row, col] = np.random.randint(0, 256, size=channels)
        return arr.tolist()  # 返回修改后的数组

 

    # '''
    # 将图片小角度旋转，从而变异
    # '''

    # def ops2(self, buf):
    #     res = buf[:]
    #     arr = np.array(res, dtype=np.uint8)  # 确保图像类型为 np.uint8
    #     height, width = arr.shape[:2]

    #     angle = random.uniform(0, 360) 
    #     M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    #     # 计算旋转后的图像大小的最大边界框
    #     corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]])
    #     transformed_corners = np.dot(M, corners.T).T

    #     # 计算旋转后的图像的最小边界框
    #     min_x = int(np.min(transformed_corners[:, 0]))
    #     max_x = int(np.max(transformed_corners[:, 0]))
    #     min_y = int(np.min(transformed_corners[:, 1]))
    #     max_y = int(np.max(transformed_corners[:, 1]))

    #     # 计算旋转后的图像的大小
    #     rotated_width = max_x - min_x
    #     rotated_height = max_y - min_y

    #     # 调整输出图像的大小，稍微大一点以容纳旋转后的图像
    #     output_width = rotated_width + 10
    #     output_height = rotated_height + 10

    #     # 执行旋转
    #     rotated = cv2.warpAffine(arr, M, (output_width, output_height))

    #     # 裁剪到原始图像大小
    #     rotated = rotated[:height, :width]

    #     return rotated.tolist()

    # '''
    # 将图片进行小范围平移
    # '''
    # def ops3(self, buf):
    #     res = buf[:]
    #     arr = np.array(res, dtype=np.float32)  # 确保输入图像数据类型为 float32
    #     height, width = arr.shape[:2]

    #     # 生成随机的平移量
    #     tx = random.uniform(-0.5 * width, 0.5 * width)
    #     ty = random.uniform(-0.5 * height, 0.5 * height)

    #     # 创建平移矩阵
    #     M = np.float32([[1, 0, tx], [0, 1, ty]])

    #     # 执行图像平移
    #     rotated = cv2.warpAffine(arr, M, (width, height))

    #     return rotated.tolist()  # 将结果转换为列表返回

    '''
    亮度和对比度调整
    '''
    def ops4(self, buf):
        # 亮度调整
        res = buf[:]
        arr = np.array(res)
        alpha = random.uniform(0.3, 3.0)  # 亮度调整
        image = cv2.convertScaleAbs(arr, alpha=alpha)
        return image.tolist()
    '''
    亮度和对比度调整
    '''
    def ops5(self, buf):
        # 对比度调整
        res = buf[:]
        arr = np.array(res)
        beta = random.uniform(-100, 100)  # 对比度调整
        image = cv2.convertScaleAbs(arr, beta=beta)
        return image.tolist()


    def add_rain(self,buf):
        """给输入图像添加雨效果"""
        image = np.array(buf, dtype=np.uint8)
        rain_transform = A.RandomRain(brightness_coefficient=0.4, drop_width=2, blur_value=3, p=1)
        return rain_transform(image=image)['image'].tolist()

    def add_snow(self,buf):
        """给输入图像添加雪效果"""
        image = np.array(buf, dtype=np.uint8)
        snow_transform = A.RandomSnow(brightness_coeff=0.8, snow_point_lower=0.3, snow_point_upper=0.6, p=1)
        return snow_transform(image=image)['image'].tolist()

    def add_fog(self,buf):
        image = np.array(buf, dtype=np.uint8)
        """给输入图像添加雾效果"""
        fog_transform = A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.6, alpha_coef=0.3, p=1)
        return fog_transform(image=image)['image'].tolist()

    def image_blur(self,buf):
        img = np.array(buf, dtype=np.uint8)
        blur = []
        params = random.randint(1, 9)
        if params == 1:
            blur = cv2.blur(img, (3, 3))
        if params == 2:
            blur = cv2.blur(img, (5, 5))
        if params == 3:
            blur = cv2.blur(img, (7, 7))
        if params == 4:
            blur = cv2.GaussianBlur(img, (5, 5), 0)
        if params == 5:
            blur = cv2.GaussianBlur(img, (7, 7), 0)
        if params == 6:
            blur = cv2.GaussianBlur(img, (9, 9), 0)
        if params == 7:
            blur = cv2.medianBlur(img, 5)
        if params == 8:
            blur = cv2.medianBlur(img, 7)
        if params == 9:
            blur = cv2.bilateralFilter(img, 9, 75, 75)
        return blur.tolist()
    

    wandlibrary.MagickMotionBlurImage.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double]

    

    # --- Defocus Blur (失焦模糊) accepts np.ndarray ---
    def defocus_blur(self, x, severity: int = 1) -> np.ndarray:
        """
        Apply defocus (out-of-focus) blur to an image array.
        x: HxWx3 uint8 array
        severity: 1-5
        Returns HxWx3 uint8 array.
        """
        # parameters per severity
        x=np.array(x, dtype=np.uint8)
        params = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)]
        radius, alias = params[severity-1]
        # build disk kernel
        L = np.arange(-radius, radius+1)
        X, Y = np.meshgrid(L, L)
        kernel = ((X**2 + Y**2) <= radius**2).astype(np.float32)
        kernel /= kernel.sum()
        kernel = cv2.GaussianBlur(kernel, ksize=(3,3) if radius<=8 else (5,5), sigmaX=alias)

        x_f = x.astype(np.float32)/255.0
        out = np.zeros_like(x_f)
        for c in range(3):
            out[...,c] = cv2.filter2D(x_f[...,c], -1, kernel)
        return (np.clip(out,0,1)*255).astype(np.uint8).tolist()

    # --- Motion Blur (运动模糊) accepts np.ndarray ---
    def motion_blur(self, x, severity: int = 1) -> np.ndarray:
        """
        Apply motion blur to an image array.
        x: HxWx3 uint8 array
        severity: 1-5
        Returns HxWx3 uint8 array.
        """
        x=np.array(x, dtype=np.uint8)
        params = [(10,3), (15,5), (15,8), (15,12), (20,15)]
        radius, sigma = params[severity-1]
        # convert to PIL via blob
        is_gray = (x.ndim==2 or x.shape[2]==1)
        mode = 'L' if is_gray else 'RGB'
        # encode to PNG
        img = WandImage(width=x.shape[1], height=x.shape[0], depth=8, background=None)
        # alternative: use PIL internally
        from PIL import Image
        x= np.array(x, dtype=np.uint8)
        pil = Image.fromarray(x) if not is_gray else Image.fromarray(x.squeeze(), mode='L')
        buf = BytesIO()
        pil.save(buf, format='PNG')
        wimg = MotionImage(blob=buf.getvalue())
        wimg.motion_blur(radius=radius, sigma=sigma, angle=np.random.uniform(-45,45))
        arr = np.frombuffer(wimg.make_blob(), dtype=np.uint8)
        img_cv = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img_cv.ndim==2:
            img_cv = np.stack([img_cv]*3, axis=-1)
        else:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return img_cv.tolist()

    # --- Gaussian Noise (高斯噪声) accepts np.ndarray ---
    def gaussian_noise(self, x, severity: int = 1) -> np.ndarray:
        """
        Add Gaussian noise to an image array.
        x: HxWx3 uint8 array
        severity: 1-5
        Returns HxWx3 uint8 array.
        """
        x=np.array(x, dtype=np.uint8)
        levels = [0.08, 0.12, 0.18, 0.26, 0.38]
        scale = levels[severity-1]
        x_f = x.astype(np.float32)/255.0
        noise = np.random.normal(0, scale, x_f.shape)
        out = x_f + noise
        return (np.clip(out,0,1)*255).astype(np.uint8).tolist()

    # --- Impulse Noise (脉冲噪声) accepts np.ndarray ---
    def impulse_noise(self, x, severity: int = 1) -> np.ndarray:
        """
        Add impulse (salt-and-pepper) noise to an image array.
        x: HxWx3 uint8 array
        severity: 1-5
        Returns HxWx3 uint8 array.
        """
        x=np.array(x, dtype=np.uint8)
        amounts = [0.03, 0.06, 0.09, 0.17, 0.27]
        amount = amounts[severity-1]
        x_f = x.astype(np.float32)/255.0
        noisy = util.random_noise(x_f, mode='s&p', amount=amount)
        return (np.clip(noisy,0,1)*255).astype(np.uint8).tolist()



    def getOps(self):
        return [self.ops1,self.ops4,self.ops5,self.add_fog, self.add_rain, self.add_snow, self.defocus_blur, self.motion_blur,
                self.impulse_noise, self.gaussian_noise]

    def ops4_d(self, buf):
        # 亮度调整
        res = buf[:]
        arr = np.array(res)
        alpha = random.uniform(0.9, 1.1)  # 亮度调整
        image = cv2.convertScaleAbs(arr, alpha=alpha)
        return image.tolist()

    def ops5_d(self, buf):
        # 对比度调整
        res = buf[:]
        arr = np.array(res)
        beta = random.uniform(-5, 5)  # 对比度调整
        image = cv2.convertScaleAbs(arr, beta=beta)
        return image.tolist()
    

    # --- Defocus Blur (失焦模糊) accepts np.ndarray ---
    def defocus_blur_d(self, x, severity: int = 1) -> np.ndarray:
        x = np.array(x, dtype=np.uint8)
        params = [(2, 0.05), (3, 0.2), (4, 0.3), (5, 0.3), (6, 0.4)]
        radius, alias = params[severity-1]
        
        L = np.arange(-radius, radius+1)
        X, Y = np.meshgrid(L, L)
        kernel = ((X**2 + Y**2) <= radius**2).astype(np.float32)
        kernel /= kernel.sum()  # 归一化
        blur_ksize = (2, 2) if radius <= 4 else (3, 3)
        kernel = cv2.GaussianBlur(kernel, ksize=blur_ksize, sigmaX=alias)

        x_f = x.astype(np.float32) / 255.0
        out = np.zeros_like(x_f)
        for c in range(3):
            out[..., c] = cv2.filter2D(x_f[..., c], -1, kernel)
        return (np.clip(out, 0, 1) * 255).astype(np.uint8).tolist()

    # --- Motion Blur (运动模糊) accepts np.ndarray ---
    def motion_blur_d(self, x, severity: int = 1) -> np.ndarray:
        """
        Apply motion blur to an image array .
        x: HxWx3 uint8 array
        severity: 1-5 
        Returns HxWx3 uint8 array (转为list输出)
        """
        x = np.array(x, dtype=np.uint8)
        params = [(5, 1), (8, 2), (10, 3), (12, 4), (15, 5)]
        radius, sigma = params[severity-1]
        
        is_gray = (x.ndim == 2 or x.shape[2] == 1)
        mode = 'L' if is_gray else 'RGB'
        
        pil = Image.fromarray(x) if not is_gray else Image.fromarray(x.squeeze(), mode='L')
        buf = BytesIO()
        pil.save(buf, format='PNG')
        wimg = MotionImage(blob=buf.getvalue())
        wimg.motion_blur(radius=radius, sigma=sigma, angle=np.random.uniform(-45, 45))
        
        arr = np.frombuffer(wimg.make_blob(), dtype=np.uint8)
        img_cv = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img_cv.ndim == 2:
            img_cv = np.stack([img_cv]*3, axis=-1)
        else:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return img_cv.tolist()

    # --- Gaussian Noise (高斯噪声) accepts np.ndarray ---
    def gaussian_noise_d(self, x, severity: int = 1) -> np.ndarray:
        """
        Add Gaussian noise to an image array .
        x: HxWx3 uint8 array
        severity: 1-5
        Returns HxWx3 uint8 array (转为list输出)
        """
        x = np.array(x, dtype=np.uint8)
        levels = [0.02, 0.04, 0.07, 0.11, 0.16]
        scale = levels[severity-1]
        
        x_f = x.astype(np.float32) / 255.0
        noise = np.random.normal(0, scale, x_f.shape)
        out = x_f + noise 
        return (np.clip(out, 0, 1) * 255).astype(np.uint8).tolist()

    # --- Impulse Noise (脉冲噪声) accepts np.ndarray ---
    def impulse_noise_d(self, x, severity: int = 1) -> np.ndarray:
        """
        Add impulse (salt-and-pepper) noise to an image array 
        x: HxWx3 uint8 array
        severity: 1-5
        Returns HxWx3 uint8 array (转为list输出)
        """
        x = np.array(x, dtype=np.uint8)
        amounts = [0.005, 0.01, 0.02, 0.04, 0.07]
        amount = amounts[severity-1]
        
        x_f = x.astype(np.float32) / 255.0
        noisy = util.random_noise(x_f, mode='s&p', amount=amount)
        return (np.clip(noisy, 0, 1) * 255).astype(np.uint8).tolist()

    
    def ops1_p(self, buf):
        arr = np.array(buf)
        # 获取图像的形状
        height, width, channels = arr.shape  # 假设 buf 是一个形状为 (height, width, channels) 的数组
        # 计算总像素数量
        total_pixels = height * width
        # 确保 m 不超过总像素数量
        num = random.randint(total_pixels // 60, total_pixels // 10)

        # 生成不重复的随机索引
        random_indices = random.sample(range(total_pixels), num)
        # 修改选定的像素
        for index in random_indices:
            # 计算行和列
            row = index // width
            col = index % width
        return arr.tolist()  # 返回修改后的数组

    def getOps_deephunter(self):
        return [self.ops4_d,self.ops5_d, self.defocus_blur, self.motion_blur, self.gaussian_noise, self.impulse_noise]

    def getOps_pythonfuzz(self):
        return [self.ops1_p]

from PIL import Image
import os
if __name__ == "__main__":
    # print("运行开始")
    # mo = ImageMutationOperator()
    # img = Image.open('/media/lzq/D/lzq/pylot_test/pylot/15096.png')  # 打开图片
    # img = img.convert('RGB')  # 确保是 RGB 格式
    # # img = img.resize(target_size)  # 调整图片大小
    # img_array = np.array(img).tolist()
    # newimg = Image.fromarray(np.array(mo.add_rain(img_array), dtype=np.uint8))
    # newimg.save('/media/lzq/D/lzq/pylot_test/pylot/hhhhjjj.png')
    # print('保存完成')

    
    mo = ImageMutationOperator()
    folder_path='/home/lzq/experiment_datatset/inputs/traffic_light'
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])

    images = []
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)
        img = img.convert('RGB')  # 确保是 RGB 格式
        # img = img.resize(target_size)  # 调整图片大小
        img_array = np.array(img).tolist()

        ind = random.randint(0, len(mo.getOps()) - 1)
        ops = mo.getOps()[ind]
        newimg = Image.fromarray(np.array(ops(img_array), dtype=np.uint8))
        newimg.save(image_path)
        

# from PIL import Image
# if __name__ == "__main__":
#     print("运行开始")
#     mo = ImageMutationOperator()
#     img = Image.open('/media/lzq/D/lzq/fuzz_tool/demo/bird.png')  # 打开图片
#     img = img.convert('RGB')  # 确保是 RGB 格式
#     # img = img.resize(target_size)  # 调整图片大小
#     img_array = np.array(img).tolist()
#     newimg = Image.fromarray(np.array(mo.motion_blur(img_array), dtype=np.uint8))
#     newimg.show()