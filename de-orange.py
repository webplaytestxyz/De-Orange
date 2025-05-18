import os
import sys
from PIL import Image, ImageEnhance

def de_orange(image_path):
    image = Image.open(image_path)
    r, g, b = image.split()
    r = r.point(lambda i: i * 0.9)  # Reduce red/orange
    image = Image.merge('RGB', (r, g, b))
    return image

def process_folder(folder):
    output_folder = os.path.join(folder, 'output_photos')
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(folder, filename)
            output_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_fixed.jpg')
            img = de_orange(input_path)
            img.save(output_path)
    print(f"Processed images saved to: {output_folder}")

if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    process_folder(folder)
