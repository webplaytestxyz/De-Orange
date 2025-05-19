import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller bundle """
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)

def load_dnn_face_detector():
    modelFile = resource_path("res10_300x300_ssd_iter_140000.caffemodel")
    configFile = resource_path("deploy.prototxt.txt")
    print("🧠 Loading DNN face detector model...")
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    print("✅ Model loaded successfully.")
    return net

def detect_faces_dnn(net, image, conf_threshold=0.5):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            boxes.append((startX, startY, endX - startX, endY - startY))
            print(f"😊 Face detected with confidence {confidence:.2f} at [{startX}, {startY}, {endX}, {endY}]")
    if not boxes:
        print("😕 No faces detected after thresholding.")
    return boxes

def deorange_face_adaptive(img, net):
    faces = detect_faces_dnn(net, img)
    if not faces:
        print("🚫 No faces found, skipping this image.")
        return img, False
    
    print(f"✨ {len(faces)} face(s) detected. Processing color correction...")
    
    for idx, (x, y, w, h) in enumerate(faces):
        print(f"🖼️ Processing face {idx+1}: x={x}, y={y}, w={w}, h={h}")
        face = img[y:y+h, x:x+w]
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        
        mean_A = np.mean(A)
        mean_B = np.mean(B)
        print(f"🎨 Mean A channel: {mean_A:.2f}, Mean B channel: {mean_B:.2f}")
        
        B_new = cv2.subtract(B, int((mean_B - 128) * 0.3))
        A_new = cv2.subtract(A, int((mean_A - 128) * 0.1))
        
        B_new = np.clip(B_new, 0, 255).astype(np.uint8)
        A_new = np.clip(A_new, 0, 255).astype(np.uint8)
        
        lab_corrected = cv2.merge([L, A_new, B_new])
        corrected_face = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        
        img[y:y+h, x:x+w] = corrected_face
        print(f"✅ Face {idx+1} color corrected.")
    return img, True

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

def estimate_color_temperature(rgb_avg):
    R, G, B = rgb_avg
    if R == 0:
        R = 1e-6
    ratio = B / R
    temp = 6600 / ratio  
    temp = np.clip(temp, 1000, 10000)
    return temp

def plot_color_temperature(input_img, output_img, save_path):
    print("ℹ️ Note: Color temperature values are approximate guides, not precise measurements.")
    
    input_rgb_avg = np.mean(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).reshape(-1,3), axis=0)
    output_rgb_avg = np.mean(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB).reshape(-1,3), axis=0)
    
    temp_in = estimate_color_temperature(input_rgb_avg)
    temp_out = estimate_color_temperature(output_rgb_avg)
    
    plt.figure(figsize=(6,4))
    bars = plt.bar(['Input Image', 'Output Image'], [temp_in, temp_out], color=['orange', 'skyblue'])
    plt.ylabel('Approx. Color Temperature (K)')
    plt.title('Color Temperature Comparison')
    plt.ylim(1000, 10000)
    
    for bar, temp in zip(bars, [temp_in, temp_out]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 500, f'{int(temp)}K', 
                 ha='center', color='black', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"🌡️ Saved color temperature plot: {save_path}")

def process_current_folder():
    print("ℹ️ Starting deorange batch process with color temperature approximation disclaimer.")
    current_folder = os.getcwd()
    output_folder = os.path.join(current_folder, "output")
    os.makedirs(output_folder, exist_ok=True)

    print(f"📁 Processing images in current folder: {current_folder}")
    net = load_dnn_face_detector()
    
    valid_exts = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(current_folder) if f.lower().endswith(valid_exts)]
    
    if not images:
        print("⚠️ No images found in the current folder.")
        return
    
    for img_name in images:
        base_name, ext = os.path.splitext(img_name)
        input_path = os.path.join(current_folder, img_name)
        output_img_name = f"{base_name}_deoranged{ext}"
        output_path = os.path.join(output_folder, output_img_name)

        print(f"\n➡️ Processing image: {img_name}")
        img = cv2.imread(input_path)
        if img is None:
            print(f"⚠️ Failed to read {img_name}, skipping.")
            continue

        corrected_img, changed = deorange_face_adaptive(img.copy(), net)
        if changed:
            success = cv2.imwrite(output_path, corrected_img)
            if success:
                print(f"🎉 Saved corrected image to {output_path}")
            else:
                print(f"❌ Failed to save corrected image for {img_name}")
        else:
            print(f"⚠️ No face correction done for {img_name}, saving original image.")
            corrected_img = img
            cv2.imwrite(output_path, img)

        hist_path = os.path.join(output_folder, f"{base_name}_deoranged_histogram.png")
        plot_histograms(img, corrected_img, hist_path)

        temp_path = os.path.join(output_folder, f"{base_name}_deoranged_colortemp.png")
        plot_color_temperature(img, corrected_img, temp_path)
    
    print("\n🚀 Batch processing complete! 🎊")

if __name__ == "__main__":
    process_current_folder()
    input("\n🔚 Processing done! Press Enter to exit...")
