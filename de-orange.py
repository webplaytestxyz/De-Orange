import os
import sys
from PIL import Image, ImageEnhance

def de_orange(image_path):
    image = Image.open(image_path).convert("RGB")
    r, g, b = image.split()

    # Reduce red slightly, increase blue slightly
    r = r.point(lambda i: i * 0.92)
    b = b.point(lambda i: min(i * 1.05, 255))

    image = Image.merge('RGB', (r, g, b))

    # Optional: Slight contrast boost for clarity
    image = ImageEnhance.Contrast(image).enhance(1.05)

    return image

def process_folder(folder):
    output_folder = os.path.join(folder, 'output_photos')
    os.makedirs(output_folder, exist_ok=True)

    processed_count = 0
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(folder, filename)
            output_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_fixed.jpg')

            try:
                img = de_orange(input_path)
                img.save(output_path)
                processed_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\n✅ Done! {processed_count} image(s) processed.")
    print(f"📁 Output folder: {output_folder}")

if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    print(f"📂 Processing folder: {folder}")
    process_folder(folder)
    input("\nPress Enter to exit...")
