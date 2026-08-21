import gc
import os
import torch
from ultralytics import YOLO
from utils import read_images_from_zip, dv_xml_to_csv
from tqdm import tqdm
from csv2xml import csv2xml
import pandas as pd
import yaml


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

        # stream=True is required, not just an optimization: with stream=False,
        # model.predict() computes and holds a Results object - GPU tensors included -
        # for every image in `images` before returning any of them, so peak GPU memory
        # scales with how many images are left in this zip, not with `batch`. Streaming
        # lets each image's result be pulled off, converted to CPU/numpy, and dropped
        # before the next one is computed, capping peak memory to ~batch images at a time
        # regardless of zip size. On OOM, resume from the first unprocessed image instead
        # of redoing the whole zip.
        start = 0
        while start < len(images):
            try:
                results = model.predict(
                    images[start:], batch=batch_size, stream=True,
                    conf=config["conf"], iou=config['iou'], imgsz=config["img_size"],
                    device=config.get("device"), verbose=False,
                    show_labels=False, show_conf=False, show_boxes=False)
                for result in results:
                    image_name = image_names[start]
                    if len(result.boxes) != 0:
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
                    start += 1
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # torch.cuda.OutOfMemoryError only covers PyTorch's own caching
                # allocator running out; a driver-level cudaMalloc failure (seen
                # under Docker Desktop on Windows/WDDM, where the GPU is shared
                # with the desktop compositor and other apps) surfaces as a plain
                # RuntimeError instead, so re-raise anything that isn't OOM-shaped
                if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                if batch_size <= 1:
                    print(f"warning: skipping {len(images) - start} image(s) in {zip_path} - "
                          f"out of GPU memory even at batch=1")
                    break
                batch_size = max(1, batch_size // 2)
        # Results/Boxes hold back-references to each other, so refcounting alone won't
        # free the last image's GPU tensors - they only die once the cyclic collector runs
        gc.collect()
        torch.cuda.empty_cache()  # bound memory fragmentation growth over a long run
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
        if "krill" in config["names"]:
            if not config["krill"]:
                df.to_csv(csvpath.split(".")[0]+"_krill.csv", index=False)
                df = df[df["label"] != "krill"]
        df.to_csv(csvpath, index=False)
        csv_file_paths.append(csvpath)
    # MODEL_NAME is set from the VARIANT build arg in Dockerfile-YOLO_DV_predict,
    # so it always matches the variant actually baked into the image; config['model_name']
    # is only a fallback for running outside that image (e.g. locally, no Docker)
    model_name = os.environ.get("MODEL_NAME", config.get("model_name"))
    csv2xml(model_name, config["version"], config['xml_file'], csv_file_paths, config['orientation'])
