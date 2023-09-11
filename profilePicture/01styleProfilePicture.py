# 01风格头像生成器
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
# n是采样区域的大小，a是第一个区间的最小值，b是第一个区间的最大值，c是第二个区间的最小值，d是第二个区间的最大值，x是字体大小
# 因为字体如果和采样区域一样大的话，有时候显示的字不够大，因此可以设置一个x，使得实际字体大小是n+x，取x为0.3n到0.8n之间的整数
class ImageProcessor:
    def __init__(self, input_image='input.jpg', n=5, a=0, b=128, d=196, x=3, font_path='C:\\Windows\\Fonts\\Consolas\\consolab.ttf'):
        self.image = Image.open(input_image).convert('RGB')
        self.n = n
        self.a = a
        self.b = b
        self.c = b
        self.d = d
        self.font_path = font_path
        self.font_size = n + x
        self.font = ImageFont.truetype(self.font_path, self.font_size)

    def process(self, output_image='output2.png'):
        w, h = self.image.size
        output = Image.new('RGB', self.image.size, color='black')
        draw = ImageDraw.Draw(output)

        for i in range(0, w, self.n):
            for j in range(0, h, self.n):
                region = np.array(self.image.crop((i, j, i+self.n, j+self.n)))
                avg_grey = region.mean()
                yellow_pixels = region[((region[:,:,0] > 100) & (region[:,:,1] > 100) & (region[:,:,2] < 50))]
                color = (255, 255, 255)
                if len(yellow_pixels) > 0:
                    color = tuple(map(int, yellow_pixels.mean(axis=0)))

                if self.a <= avg_grey <= self.b:
                    draw.text((i, j), '1', font=self.font, fill=color)
                elif self.c <= avg_grey <= self.d:
                    draw.text((i, j), '0', font=self.font, fill=color)

        output.save(output_image)

if __name__ == '__main__':
    for b in range(80, 151, 5):
        for n in range(5, 13, 1):
            for d in range(100, 201, 5):
                for x in range(int(0.3 * n), int(0.8 * n) + 1, 1):
                    processor = ImageProcessor(n=n, b=b, d=d, x=x)
                    output_image = f'output_{n}_{b}_{d}_{x}.png'
                    processor.process(output_image)
