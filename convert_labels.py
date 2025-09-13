import os
import shutil
import random
import xml.etree.ElementTree as ET

# =========================
# CONFIG
# =========================
images_dir = "Data/images"        # folder with .png files
labels_dir = "Data/annotations"   # folder with .xml files
output_dir = "dataset"           # final YOLO dataset root

train_split = 0.7
val_split = 0.2
test_split = 0.1

# Define classes in your dataset
classes = ["helmet"]  # change if you have more than one class

# =========================
# CREATE FOLDER STRUCTURE
# =========================
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# =========================
# HELPER: Convert VOC XML → YOLO TXT
# =========================
def convert_voc_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)
    
    lines = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Convert to YOLO format
        x_center = (xmin + xmax) / (2.0 * img_w)
        y_center = (ymin + ymax) / (2.0 * img_h)
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return "\n".join(lines)

# =========================
# MATCH IMAGES & SPLIT
# =========================
image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
random.shuffle(image_files)

n_total = len(image_files)
n_train = int(n_total * train_split)
n_val = int(n_total * val_split)
n_test = n_total - n_train - n_val

splits = {
    "train": image_files[:n_train],
    "val": image_files[n_train:n_train + n_val],
    "test": image_files[n_train + n_val:]
}

print(f"Total images: {n_total}")
print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

# =========================
# COPY + CONVERT FILES
# =========================
for split, files in splits.items():
    for img_file in files:
        # Copy image
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(output_dir, "images", split, img_file)
        shutil.copy2(src_img, dst_img)

        # Process matching XML
        xml_file = img_file.replace(".png", ".xml")
        src_xml = os.path.join(labels_dir, xml_file)
        if os.path.exists(src_xml):
            yolo_label = convert_voc_to_yolo(src_xml)
            dst_txt = os.path.join(output_dir, "labels", split, img_file.replace(".png", ".txt"))
            with open(dst_txt, "w") as f:
                f.write(yolo_label)
