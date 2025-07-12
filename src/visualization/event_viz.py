import numba
import numpy as np

@numba.jit(nopython=True)
def draw_events_on_image(img, x, y, p, alpha=0.5):
    img_copy = img.copy()
    for i in range(len(p)):
        if y[i] < len(img):
            img[y[i], x[i], :] = alpha * img_copy[y[i], x[i], :]
            img[y[i], x[i], int(p[i])-1] += 255 * (1-alpha)
    return img

def map_polarity_to_channel_index(polarity_values):
    # numpy配列として扱うことで効率的に処理
    mapped_values = np.empty_like(polarity_values, dtype=np.int32)
    mapped_values[polarity_values == 1] = 1  # 1 (赤) -> チャンネル0 (赤)
    mapped_values[polarity_values == -1] = 3 # -1 (青) -> チャンネル2 (青)
    return mapped_values