import zipfile
import io
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image

# Function to read images from ZIP file without extracting them
def read_images_from_zip(zip_path):
    images = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                with zip_ref.open(file) as image_file:
                    image = Image.open(io.BytesIO(image_file.read()))
                    images.append((file, image))
    return images

def dv_xml_to_csv(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    header = ('datetime', 'depth')
    table = []

    for element in root.iter("frames"):
        for frame in element:
            depth = frame.get("depth")
            time = frame.get('time')
            row = time, -float(depth)
            table.append(row)
    out_df = pd.DataFrame(table, columns=header)
    return out_df