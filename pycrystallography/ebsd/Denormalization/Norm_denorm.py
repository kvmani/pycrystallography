import os
import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt

def configure_logger():
    import logging
    logger = logging.getLogger("NormalizationScript")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def configure_path(logger):
    """
    Configures sys.path to locate the pycrystallography package if not installed globally.
    """
    try:
        from pycrystallography.core.orientation import Orientation
    except ImportError:
        logger.warning("Unable to find the pycrystallography package. Adjusting system path.")
        sys.path.insert(0, os.path.abspath('.'))
        sys.path.insert(0, os.path.dirname('..'))
        sys.path.insert(0, os.path.dirname('../../pycrystallography'))
        sys.path.insert(0, os.path.dirname('../../..'))
        for path in sys.path:
            logger.debug(f"Updated Path: {path}")
        from pycrystallography.ebsd.ebsd import Ebsd


def read_euler_angles(ebsd):
    """
    Extracts Euler angles from EBSD data (phi1, PHI, phi2 for .ang; Euler1, Euler2, Euler3 for .ctf).
    """
    if "ang" in ebsd._ebsdFormat:
        euler_angles = np.array([ebsd._data["phi1"], ebsd._data["PHI"], ebsd._data["phi2"]]).T
    elif "ctf" in ebsd._ebsdFormat:
        euler_angles = np.array([ebsd._data["Euler1"], ebsd._data["Euler2"], ebsd._data["Euler3"]]).T
    else:
        raise ValueError("Unsupported EBSD format: only .ang and .ctf are supported.")
    return euler_angles


def save_image(data, output_path):
    """
    Save an image using Pillow (PIL) instead of matplotlib.
    Assumes data is in (rows, cols, channels) format.
    """
    img = Image.fromarray(data.astype(np.uint8))
    img.save(output_path)
    print(f"Image saved to: {output_path}")


def normalize_euler_angles_to_image(ebsd, output_image_path):
    """
    Use the EBSD method to write a normalized Euler angle PNG.
    """
    ebsd.writeEulerAsPng(pathName=output_image_path, showMap=False)
    return ebsd._oriDataInt


def denormalize_euler_angles(rgb_array, lattice_limits):
    """
    Denormalize RGB image data back to Euler angles.
    """
    phi1 = np.interp(rgb_array[:, :, 0], [0, 255], [0, lattice_limits[0]])
    PHI = np.interp(rgb_array[:, :, 1], [0, 255], [0, lattice_limits[1]])
    phi2 = np.interp(rgb_array[:, :, 2], [0, 255], [0, lattice_limits[2]])
    return np.stack([phi1, PHI, phi2], axis=2).reshape(-1, 3)


def calculate_difference(original, denormalized, cols, rows):
    """
    Visualize the differences between original and denormalized Euler angles for each channel.
    """
    diff = np.abs(original - denormalized)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    return diff, max_diff, mean_diff


if __name__ == "__main__":
    logger = configure_logger()
    configure_path(logger)
    from pycrystallography.ebsd.ebsd import Ebsd

    # Define input/output paths
    input_folder = "data/sampleTestData"
    output_folder = "output/normalization"
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over files in the input folder
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if not (file_name.endswith(".ang") or file_name.endswith(".ctf")):
            continue

        logger.info(f"Processing file: {file_name}")
        ebsd = Ebsd()
        if file_name.endswith(".ang"):
            ebsd.fromAng(file_path)
        elif file_name.endswith(".ctf"):
            ebsd.fromCtf(file_path)

        # Retrieve lattice limits and Euler angles
        if ebsd._isEulerAnglesReduced:
            lattice_limits = np.array(ebsd._lattice._EulerLimits)
        else:
            lattice_limits = [np.pi * 2, np.pi, np.pi * 2]

        euler_angles = read_euler_angles(ebsd)

        # Columns and rows from ebsd
        cols = ebsd.nXPixcels
        rows = ebsd.nYPixcels

        # Save original Euler angles as an image
        original_img = euler_angles.reshape((rows, cols, 3))
        plt.figure(figsize=(10, 8))
        plt.imshow(original_img, interpolation="nearest")
        plt.title("original image")
        plt.axis('off')
        plt.colorbar(label='Euler Angles (Rad)')
        
        # original_image_path = os.path.join(output_folder, f"{file_name}_raw.png")
        # save_image(original_img, original_image_path)

        # Normalize Euler angles to image
        normalized_image_path = os.path.join(output_folder, f"{file_name}_normalized.png")
        normalized_values = normalize_euler_angles_to_image(ebsd, normalized_image_path)

        # Denormalize image back to Euler angles
        denormalized_euler_angles = denormalize_euler_angles(normalized_values, lattice_limits)

        # Save denormalized Euler angles as an image
        denorm_img = denormalized_euler_angles.reshape((rows, cols, 3))
        plt.figure(figsize=(10, 8))
        plt.imshow(denorm_img, interpolation="nearest")
        plt.title("denormalized image")
        plt.axis('off')
        plt.colorbar(label='Euler Angles (Rad)')

        # Calculate and save difference
        diff, maxDiff, meanDiff  = calculate_difference(euler_angles, denormalized_euler_angles, cols, rows)
        logger.info(f" max diff: {maxDiff}, mean diff: {meanDiff}")

        logger.info(f"Processing complete for {file_name}.")
        plt.show()
        break
