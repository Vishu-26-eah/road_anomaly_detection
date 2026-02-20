import os
import glob
import xml.etree.ElementTree as ET

# class mapping (update if needed)
classes = ['D00', 'D01', 'D10', 'D11', 'D20', 'D40', 'D43', 'D44']

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

xml_dirs = [
    "archive/train/Czech/annotations/xmls",
    "archive/train/India/annotations/xmls",
    "archive/train/Japan/annotations/xmls"
]

output_dir = "datasets/RDD2020/labels/train"

os.makedirs(output_dir, exist_ok=True)

for xml_dir in xml_dirs:
    for xml_file in glob.glob(os.path.join(xml_dir, "*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        txt_filename = os.path.splitext(os.path.basename(xml_file))[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, "w") as out_file:
            for obj in root.iter("object"):
                cls = obj.find("name").text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find("bndbox")
                box = (
                    float(xmlbox.find("xmin").text),
                    float(xmlbox.find("xmax").text),
                    float(xmlbox.find("ymin").text),
                    float(xmlbox.find("ymax").text)
                )
                bb = convert((w, h), box)
                out_file.write(str(cls_id) + " " + " ".join(map(str, bb)) + "\n")

print("Conversion completed.")
