import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller bundle """
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)

def deorange_full_image(img):
    print("✨ Doing De-Orange magic, just a sec...")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    mean_A = np.mean(A)
    mean_B = np.mean(B)
    
    # Subtract a fraction of the difference to neutralize orange tint
    B_new = cv2.subtract(B, int((mean_B - 128) * 0.3))
    A_new = cv2.subtract(A, int((mean_A - 128) * 0.1))

    B_new = np.clip(B_new, 0, 255).astype(np.uint8)
    A_new = np.clip(A_new, 0, 255).astype(np.uint8)

    lab_corrected = cv2.merge([L, A_new, B_new])
    corrected_img = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    print("✅ De-Orange - orange tint corrected!")
    return corrected_img

def plot_histograms(input_img, output_img, save_path):
    chans_in = cv2.split(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    chans_out = cv2.split(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    colors = ('r', 'g', 'b')

    plt.figure(figsize=(12,5))
    for i, color in enumerate(colors):
        hist_in = cv2.calcHist([chans_in[i]], [0], None, [256], [0,256])
        hist_out = cv2.calcHist([chans_out[i]], [0], None, [256], [0,256])

        plt.subplot(2,3,i+1)
        plt.plot(hist_in, color=color)
        plt.title(f'Input {color.upper()} Channel')
        plt.xlim([0,256])
        plt.ylim([0, max(hist_in.max(), hist_out.max())*1.1])

        plt.subplot(2,3,i+4)
        plt.plot(hist_out, color=color)
        plt.title(f'Output {color.upper()} Channel')
        plt.xlim([0,256])
        plt.ylim([0, max(hist_in.max(), hist_out.max())*1.1])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved histogram plot: {save_path}")

def process_current_folder():
    print("ℹ️ Starting batch processing...")
    current_folder = os.getcwd()
    output_folder = os.path.join(current_folder, "output")
    os.makedirs(output_folder, exist_ok=True)

    valid_exts = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(current_folder) if f.lower().endswith(valid_exts)]

    if not images:
        print("⚠️ No images found.")
        return

    for img_name in images:
        base_name, ext = os.path.splitext(img_name)
        input_path = os.path.join(current_folder, img_name)
        output_img_name = f"{base_name}_deoranged{ext}"
        output_path = os.path.join(output_folder, output_img_name)

        print(f"\n➡️ Processing: {img_name}")
        img = cv2.imread(input_path)
        if img is None:
            print(f"⚠️ Could not read {img_name}, skipping.")
            continue

        corrected_img = deorange_full_image(img.copy())
        if cv2.imwrite(output_path, corrected_img):
            print(f"🎉 Saved corrected image: {output_path}")
        else:
            print(f"❌ Failed to save {img_name}")

        hist_path = os.path.join(output_folder, f"{base_name}_deoranged_histogram.png")
        plot_histograms(img, corrected_img, hist_path)

    print("\n🚀 All done! Press Enter to exit.")
    input()

if __name__ == "__main__":
    process_current_folder()
