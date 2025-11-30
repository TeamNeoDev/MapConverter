import numpy as np
from numba import jit
from PIL import Image
import csv
import time


def load_palette(csv_path):
    """
    Load color palette from CSV file.
    Excludes Base Color ID×4 + 3 (multiplier 135) colors for staircase method.
    This filters to 183 usable colors instead of 244.
    """
    colors = []
    ids = []
    blocks = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            color_id = int(row['id'])
            # Exclude colors where id % 4 == 3 (Base Color ID×4 + 3)
            if color_id % 4 != 3:
                ids.append(color_id)
                colors.append([int(row['r']), int(row['g']), int(row['b'])])
                blocks.append(row['block'])
    return np.array(ids), np.array(colors, dtype=np.uint8), blocks


def pil_to_numpy(img):
    return np.array(img, dtype=np.uint8)


def numpy_to_pil(arr):
    return Image.fromarray(arr.astype(np.uint8))


@jit(nopython=True)
def build_color_lookup_table(palette):
    """
    Build a lookup table for fast color matching.
    Uses 6-bit quantization (64 levels per channel) = 64^3 = 262,144 entries
    Each entry stores the index of the closest palette color.
    """
    # Using 6 bits = 64 levels per channel (0-63)
    levels = 64
    lut = np.empty((levels, levels, levels), dtype=np.uint8)

    for r_idx in range(levels):
        for g_idx in range(levels):
            for b_idx in range(levels):
                # Convert back to 0-255 range
                r = (r_idx * 255) // (levels - 1)
                g = (g_idx * 255) // (levels - 1)
                b = (b_idx * 255) // (levels - 1)

                # Find closest palette color
                min_dist = 999999999
                closest_idx = 0
                for i in range(len(palette)):
                    dr = r - palette[i, 0]
                    dg = g - palette[i, 1]
                    db = b - palette[i, 2]
                    dist = dr * dr + dg * dg + db * db
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i

                lut[r_idx, g_idx, b_idx] = closest_idx

    return lut


@jit(nopython=True)
def find_closest_color_lut(r, g, b, lut, palette):
    """Find the closest color index using the lookup table."""
    # Convert 0-255 to 0-63 index
    r_idx = (r * 63) // 255
    g_idx = (g * 63) // 255
    b_idx = (b * 63) // 255
    
    c = [
        [r_idx, g_idx, b_idx],
        [r_idx+1, g_idx, b_idx],
        [r_idx+1, g_idx+1, b_idx],
        [r_idx, g_idx+1, b_idx],
        [r_idx, g_idx+1, b_idx+1],
        [r_idx+1, g_idx+1, b_idx+1],
        [r_idx+1, g_idx, b_idx+1],
        [r_idx, g_idx, b_idx+1],
    ]
    
    min_dist = 10**9
    closest_idx = [0, 0, 0]
    for i in c:
        if not (0 <= i[0] < 64 and 0 <= i[1] < 64 and 0 <= i[2] < 64): continue
        dr = r - palette[lut[i[0], i[1], i[2]]][0]
        dg = g - palette[lut[i[0], i[1], i[2]]][1]
        db = b - palette[lut[i[0], i[1], i[2]]][2]
        dist = dr * dr + dg * dg + db * db
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    return lut[closest_idx[0], closest_idx[1], closest_idx[2]]


@jit(nopython=True)
def floyd_steinberg_color_optimized(image, palette, lut):
    """
    Optimized Floyd-Steinberg dithering using techniques from pythonspeed.com:
    - Integer arithmetic (0-255 range)
    - Removed conditionals by adding padding
    - Two-row buffer with swapping
    - Register-friendly local variables
    - Lookup table for O(1) color matching

    Args:
        image: np.array of shape (height, width, 3), dtype=uint8
        palette: np.array of shape (n_colors, 3), dtype=uint8
        lut: np.array of shape (64, 64, 64), dtype=uint8 - color lookup table

    Returns:
        Dithered image with colors from the palette
    """
    h, w = image.shape[0], image.shape[1]

    # Result array
    result = np.empty((h, w, 3), dtype=np.uint8)
    result_idx = np.empty((h, w), dtype=np.uint8)

    # Two-row buffers with padding to remove edge conditionals
    # Extra column on each side to avoid boundary checks
    current_row = np.zeros((w + 2, 3), dtype=np.int32)
    next_row = np.zeros((w + 2, 3), dtype=np.int32)

    # Initialize first row (offset by 1 due to padding)
    for x in range(w):
        current_row[x + 1, 0] = image[0, x, 0]
        current_row[x + 1, 1] = image[0, x, 1]
        current_row[x + 1, 2] = image[0, x, 2]

    for y in range(h):
        # Load next row if not at last row
        if y + 1 < h:
            for x in range(w):
                next_row[x + 1, 0] = image[y + 1, x, 0]
                next_row[x + 1, 1] = image[y + 1, x, 1]
                next_row[x + 1, 2] = image[y + 1, x, 2]
        else:
            # Clear next_row if we're at the last row
            for x in range(w + 2):
                next_row[x, 0] = 0
                next_row[x, 1] = 0
                next_row[x, 2] = 0

        for x in range(w):
            # Use local variables for register optimization
            idx = x + 1  # Offset for padding

            # Clamp values to 0-255 range
            old_r = max(0, min(255, current_row[idx, 0]))
            old_g = max(0, min(255, current_row[idx, 1]))
            old_b = max(0, min(255, current_row[idx, 2]))

            # Find closest palette color using lookup table (O(1) instead of O(n))
            closest_idx = find_closest_color_lut(old_r, old_g, old_b, lut, palette)
            result_idx[y, x] = closest_idx
            new_r = palette[closest_idx, 0]
            new_g = palette[closest_idx, 1]
            new_b = palette[closest_idx, 2]

            # Store result
            result[y, x, 0] = new_r
            result[y, x, 1] = new_g
            result[y, x, 2] = new_b

            # Calculate error
            err_r = old_r - new_r
            err_g = old_g - new_g
            err_b = old_b - new_b

            # Distribute error (using bit shifts for division approximation)
            # 7/16 ≈ 0.4375
            current_row[idx + 1, 0] += (err_r * 7) >> 4
            current_row[idx + 1, 1] += (err_g * 7) >> 4
            current_row[idx + 1, 2] += (err_b * 7) >> 4

            # 3/16 ≈ 0.1875
            next_row[idx - 1, 0] += (err_r * 3) >> 4
            next_row[idx - 1, 1] += (err_g * 3) >> 4
            next_row[idx - 1, 2] += (err_b * 3) >> 4

            # 5/16 ≈ 0.3125
            next_row[idx, 0] += (err_r * 5) >> 4
            next_row[idx, 1] += (err_g * 5) >> 4
            next_row[idx, 2] += (err_b * 5) >> 4

            # 1/16 ≈ 0.0625
            next_row[idx + 1, 0] += err_r >> 4
            next_row[idx + 1, 1] += err_g >> 4
            next_row[idx + 1, 2] += err_b >> 4

        # Swap rows (instead of copying)
        current_row, next_row = next_row, current_row

    return result, result_idx

def convert_mcmap(image_idx, ids, palette, block):
    h, w = image_idx.shape[0], image_idx.shape[1]
    
    _map_data = [[[0, ""] for j in range(w)] for i in range(h+1)]
    for _i in range(h, 0, -1):
        i = _i-1
        for j in range(w):
            chg = 0
            if ids[image_idx[i][j]] % 4 == 0:
                chg = 1
            elif ids[image_idx[i][j]] % 4 == 1:
                chg = 0
            elif ids[image_idx[i][j]] % 4 == 2:
                chg = -1
            _map_data[_i-1][j][0] = _map_data[_i][j][0] + chg
            _map_data[_i][j][1] = block[image_idx[i][j]]
    
    return _map_data

import sys

# Load color palette
print("Loading palette...")
start_time = time.perf_counter()
palette_ids, palette_colors, blocks = load_palette("./colors.csv")
load_time = time.perf_counter() - start_time
print(f"Loaded {len(palette_colors)} colors from palette in {load_time*1000:.2f} ms")

# Build lookup table for fast color matching
print("\nBuilding color lookup table...")
start_time = time.perf_counter()
color_lut = build_color_lookup_table(palette_colors)
lut_time = time.perf_counter() - start_time
print(f"Lookup table built in {lut_time*1000:.2f} ms")
print(f"LUT size: {color_lut.shape} = {color_lut.size:,} entries ({color_lut.nbytes / 1024:.1f} KB)")

# Load and process image
print("\nLoading image...")
start_time = time.perf_counter()
img = Image.open(sys.argv[1]).convert("RGB")
arr = pil_to_numpy(img)
img_load_time = time.perf_counter() - start_time
print(f"Image shape: {arr.shape}")
print(f"Image loaded in {img_load_time*1000:.2f} ms")

# Apply optimized Floyd-Steinberg dithering with custom palette
print("\nApplying Floyd-Steinberg dithering...")
start_time = time.perf_counter()
dithered_arr, dithered_arr_idx = floyd_steinberg_color_optimized(arr, palette_colors, color_lut)
dither_time = time.perf_counter() - start_time
print(f"Dithering completed in {dither_time*1000:.2f} ms ({dither_time:.4f} seconds)")

print("Converting image data into Minecraft Map Data...")
start_time = time.perf_counter()
map_data = convert_mcmap(dithered_arr_idx, palette_ids, palette_colors, blocks)
convert_time = time.perf_counter() - start_time
print(f"Converted data in {convert_time*1000:.2f} ms ({convert_time:.4f} seconds)")

# Save result
print("\nSaving result (image)...")
start_time = time.perf_counter()
dithered_img = numpy_to_pil(dithered_arr)
dithered_img.save("./dithered_image.png")
save_image_time = time.perf_counter() - start_time
print(f"Image saved in {save_image_time*1000:.2f} ms")

print("\nSaving result (converted data)...")
start_time = time.perf_counter()
with open("./converted_data(merged).csv", "w", encoding="utf-8") as f:
    f.write("\n".join(map(lambda a: ",".join(map(lambda b: str(b).replace(",", "\\,"), a)), map_data)))
with open("./converted_data(blocks).csv", "w", encoding="utf-8") as f:
    f.write("\n".join(map(lambda a: ",".join(map(lambda b: b[1], a)), map_data)))
with open("./converted_data(y_pos).csv", "w", encoding="utf-8") as f:
    f.write("\n".join(map(lambda a: ",".join(map(lambda b: str(b[0]), a)), map_data)))
save_data_time = time.perf_counter() - start_time
print(f"Image saved in {save_data_time*1000:.2f} ms")

# Summary
print("\n" + "="*50)
print("TIMING SUMMARY")
print("="*50)
print(f"Palette loading:  {load_time*1000:>10.2f} ms")
print(f"LUT building:     {lut_time*1000:>10.2f} ms")
print(f"Image loading:    {img_load_time*1000:>10.2f} ms")
print(f"Dithering:        {dither_time*1000:>10.2f} ms  ← Main processing")
print(f"Converting:       {convert_time*1000:>10.2f} ms")
print(f"Image saving:     {save_image_time*1000:>10.2f} ms")
print(f"Data saving:      {save_data_time*1000:>10.2f} ms")
print("-"*50)
total_time = load_time + lut_time + img_load_time + dither_time + save_image_time + save_data_time + convert_time
print(f"Total time:       {total_time*1000:>10.2f} ms ({total_time:.4f} seconds)")
print("="*50)