class Normalize(object):
    """Apply per-channel min-max normalization with a special NDVI range rule.

    For channel 4, bounds use only values in [0, 1). All pixels are then
    transformed, so resulting NDVI values can lie outside [0, 1].
    The tensor is updated in place to preserve the checkpoint convention.
    """

    def __call__(self, image, target=None):
        for i in range(image.size(0)):
            band = image[i]
            if i == 4:
                mask = (band >= 0) & (band < 1)
                valid_band = band[mask]
                # An empty valid subset uses zero bounds; do not infer spectral metadata.
                if valid_band.numel() == 0:
                    min_val, max_val = (0.0, 0.0)
                else:
                    min_val = valid_band.min()
                    max_val = valid_band.max()
            else:
                min_val = band.min()
                max_val = band.max()
            # Constant channels are shifted without dividing by a zero range.
            if max_val != min_val:
                image[i] = (band - min_val) / (max_val - min_val)
            else:
                image[i] = band - min_val
        return image if target is None else (image, target)
