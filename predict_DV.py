import os
import torch
from ultralytics import YOLO
from utils import read_images_from_zip, dv_xml_to_csv
from tqdm import tqdm
from csv2xml import csv2xml
import pandas as pd
import yaml


def predict_with_oom_backoff(model, images, batch, **predict_kwargs):
    """model.predict(), halving `batch` and retrying on CUDA OOM until it fits.
    Returns (results, batch) so the caller can keep using the now-known-safe batch size."""
    while True:
        try:
            return model.predict(images, batch=batch, **predict_kwargs), batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 1:
                raise
            batch = max(1, batch // 2)


def predict_zip(config, orientation, LUT):
    path_to_data = os.path.join(config["path_to_data"], orientation)
    zipfiles = sorted([i for i in os.listdir(path_to_data) if i.endswith("_active.zip")])
    model = YOLO(config['trained_model'])
    batch_size = config.get("batch", 16)
    predictions = []
    for zipfile in tqdm(zipfiles, desc=f"Predicting on images from the {orientation} camera"):
        zip_path = os.path.join(path_to_data, zipfile)
        image_data = read_images_from_zip(zip_path)
        if not image_data:
            continue
        image_names = [name for name, _ in image_data]
        images = [image for _, image in image_data]
        try:
            results, batch_size = predict_with_oom_backoff(
                model, images, batch_size, conf=config["conf"], iou=config['iou'], imgsz=config["img_size"],
                device=config.get("device"), verbose=False,
                show_labels=False, show_conf=False, show_boxes=False)
        except torch.cuda.OutOfMemoryError:
            print(f"warning: skipping {zip_path} - out of GPU memory even at batch=1")
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()  # bound memory fragmentation growth over a long run
        for image_name, result in zip(image_names, results):
            if len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for (x0, y0, x1, y1), conf, cls in zip(xyxy, confs, classes):
                label = LUT[cls]
                if config["opt_thresholds"] is not None:
                    opt_thres = config["opt_thresholds"].get(label)
                    if opt_thres is not None and conf < opt_thres:
                        continue
                predictions.append({
                    "datetime": image_name.split(".")[0],  # The image name
                    "x0": int(x0),  # The x-coordinate of the top-left corner
                    "y0": int(y0),  # The y-coordinate of the top-left corner
                    "x1": int(x1),  # The x-coordinate of the bottom-right corner
                    "y1": int(y1),  # The y-coordinate of the bottom-right corner
                    "label": label,  # The class ID
                    "score": round(float(conf), 3)  # The confidence score
                })
    df_pred = pd.DataFrame(predictions)
    return df_pred


if __name__ == '__main__':
    with open('model/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        LUT = {v: k for v, k in enumerate(sorted(config['names']))}

    df_xml = dv_xml_to_csv(config['xml_file'])
    csv_file_paths = []
    for orientation in config['orientation']:
        csvpath = config['xml_file'].split(".")[0] + "_" + orientation + ".csv"
        df_pred = predict_zip(config, orientation, LUT)
        df = pd.merge(df_xml, df_pred, left_on="datetime", right_on="datetime", how="inner")
        if not config["krill"]:
            df = df[df["label"] != "krill"]
        df.to_csv(csvpath, index=False)
        csv_file_paths.append(csvpath)
    csv2xml(config["model_name"], config["version"], config['xml_file'], csv_file_paths, config['orientation'])
