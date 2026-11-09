class Normalize(object):
    """对每个波段进行0-1范围归一化，NDVI波段（索引4）排除异常值"""

    def __call__(self, image, target=None):
        for i in range(image.size(0)):
            band = image[i]
            if i == 4:
                mask = (band >= 0) & (band < 1)
                valid_band = band[mask]
                if valid_band.numel() == 0:
                    min_val, max_val = (0.0, 0.0)
                else:
                    min_val = valid_band.min()
                    max_val = valid_band.max()
            else:
                min_val = band.min()
                max_val = band.max()
            if max_val != min_val:
                image[i] = (band - min_val) / (max_val - min_val)
            else:
                image[i] = band - min_val
        return image if target is None else (image, target)
