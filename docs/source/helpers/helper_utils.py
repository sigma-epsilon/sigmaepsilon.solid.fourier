import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

__all__ = ["save_png_images"]


def save_png_images(fig: plt.Figure, filename_base: str) -> None:
    """
    Save a matplotlib figure as PNG images for both light and dark modes.
    This function saves the provided matplotlib figure as a PNG image with a light background,
    then creates a dark mode version by inverting the colors and replacing the background with
    a dark color. Both images are saved to the ../_static/ directory with appropriate suffixes.
    
    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to be saved as PNG images.
    filename_base : str
        The base filename (without extension or mode suffix) for the output images.
    
    Returns
    -------
    None
        This function does not return anything. It saves two PNG files to disk.
    
    Notes
    -----
    - The light mode image is saved as "{filename_base}_light.png".
    - The dark mode image is saved as "{filename_base}_dark.png" with background set to [20, 24, 29].
    - The function prints the file paths of the saved images.
    """
    
    file_name_light = f"../_static/{filename_base}_light.png"
    fig.savefig(file_name_light, format="png")
    
    print(f"Saved light mode image to: {file_name_light}")
    
    # Load image
    img = Image.open(file_name_light).convert("RGB")
    arr = np.array(img)

    # PyData dark background color
    bg_color = np.array([20, 24, 29], dtype=np.uint8)

    # Threshold for detecting "white" background
    white_thresh = 245

    # Mask: background pixels (nearly white)
    bg_mask = np.all(arr >= white_thresh, axis=2)

    # Invert the whole image
    inv = 255 - arr

    # Replace background explicitly
    inv[bg_mask] = bg_color

    # Save result
    file_name_dark = f"../_static/{filename_base}_dark.png"
    Image.fromarray(inv).save(file_name_dark)
    print(f"Saved dark mode image to: {file_name_dark}")
    
    