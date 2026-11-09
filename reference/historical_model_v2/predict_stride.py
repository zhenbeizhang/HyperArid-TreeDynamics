import os
import time
import numpy as np
import torch
from osgeo import gdal
from tqdm import tqdm

gdal.UseExceptions()

import src
from transforms import Compose, Normalize, ToTensor
from src import UNET


def time_synchronized():
    """Synchronize CUDA for accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def sliding_window_prediction(model, image, window_size=256, stride=256, threshold=0.5, device="cpu"):
    """
    Perform sliding window inference on large images.

    Parameters
    ----------
    model : torch.nn.Module
        Trained segmentation model.
    image : numpy.ndarray
        Input image with shape [H, W, C].
    window_size : int
        Size of the inference window.
    stride : int
        Sliding window stride.
    threshold : float
        Threshold used for binarization.
    device : str
        Device used for inference.

    Returns
    -------
    numpy.ndarray
        Binary prediction mask with shape [H, W].
    """

    h, w, c = image.shape

    result = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.int32)

    data_transform = Compose([
        ToTensor(),
        Normalize()
    ])

    with torch.no_grad():

        for y_start in range(0, h, stride):
            for x_start in range(0, w, stride):

                y_end = min(y_start + window_size, h)
                x_end = min(x_start + window_size, w)

                if y_end - y_start < window_size or x_end - x_start < window_size:

                    window = np.zeros((window_size, window_size, c), dtype=np.float32)

                    window_h = y_end - y_start
                    window_w = x_end - x_start

                    window[:window_h, :window_w] = image[y_start:y_end, x_start:x_end]

                else:
                    window = image[y_start:y_end, x_start:x_end]

                window_tensor = data_transform(window)
                window_tensor = torch.unsqueeze(window_tensor, 0).to(device)

                pred = model(window_tensor)
                pred = torch.squeeze(pred).cpu().numpy()

                if y_end - y_start < window_size or x_end - x_start < window_size:
                    pred = pred[:window_h, :window_w]

                result[y_start:y_end, x_start:x_end] += pred
                count[y_start:y_end, x_start:x_end] += 1

    result = np.divide(result, count, out=np.zeros_like(result), where=count > 0)

    result_mask = (result > threshold).astype(np.uint8) * 255

    return result_mask


def main():

    # Paths should be configured by the user
    weights_path = ""
    image_folder = ""
    output_folder = ""

    threshold = 0.5

    window_size = 256
    stride = 256

    os.makedirs(output_folder, exist_ok=True)

    band_combination = [0, 1, 2, 3, 4]
    use_ndvi = True

    num_bands = len(band_combination)

    assert os.path.exists(image_folder), f"Image folder {image_folder} does not exist."

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = src.UNET.unet(n_channels=num_bands)

    weights = torch.load(weights_path, map_location="cpu")

    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)

    model.to(device)
    model.eval()

    img_files = [f for f in os.listdir(image_folder) if f.lower().endswith(".tif")]

    print(f"{len(img_files)} images found for inference")

    for img_file in tqdm(img_files, desc="Inference", unit="image"):

        img_path = os.path.join(image_folder, img_file)

        try:

            ds = gdal.Open(img_path)
            assert ds is not None, f"Unable to read file: {img_path}"

            bands = []

            for i in range(ds.RasterCount):
                band = ds.GetRasterBand(i + 1).ReadAsArray().astype(np.float32)
                bands.append(band)

            if use_ndvi:

                red = bands[0]
                nir = bands[3]

                ndvi = (nir - red) / (nir + red + 1e-6)

                bands.append(ndvi)

            selected_bands = [bands[i] for i in band_combination]

            image = np.stack(selected_bands, axis=2)

            origin_h, origin_w = image.shape[:2]

            start_time = time_synchronized()

            pred_mask = sliding_window_prediction(
                model=model,
                image=image,
                window_size=window_size,
                stride=stride,
                threshold=threshold,
                device=device
            )

            end_time = time_synchronized()

            print(f"Inference finished in {end_time - start_time:.3f}s")

            output_name = os.path.join(
                output_folder,
                os.path.splitext(img_file)[0] + "_prediction.tif"
            )

            driver = gdal.GetDriverByName("GTiff")

            out_ds = driver.Create(output_name, origin_w, origin_h, 1, gdal.GDT_Byte)

            if ds.GetGeoTransform():
                out_ds.SetGeoTransform(ds.GetGeoTransform())

            if ds.GetProjection():
                out_ds.SetProjection(ds.GetProjection())

            out_ds.GetRasterBand(1).WriteArray(pred_mask)

            out_ds.FlushCache()

        except Exception as e:

            print(f"Error processing {img_file}: {e}")
            continue


if __name__ == "__main__":
    main()