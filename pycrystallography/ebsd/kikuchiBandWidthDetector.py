import os
import json
import glob
import numpy as np
import pandas as pd
import yaml
import logging
from PIL import Image
from scipy.ndimage import gaussian_filter1d, map_coordinates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from openpyxl import Workbook


# Ensure the ./tmp directory exists
os.makedirs("./tmp", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set log level to INFO
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("./tmp/bandDetector.log"),  # Log to file
        logging.StreamHandler()  # Also log to console
    ]
)


class BandWidthDetector:
    def __init__(self, band_input_data, options):
        """
        Initialize the detector with the image, list of points, and additional properties.
        """
        self.image_base_name = None
        self.points = band_input_data.get("points")  # List of dictionaries
        self.additional_properties = {k: v for k, v in band_input_data.items() if k != "points"}
        self.smoothing_sigma = options.get('smoothing_sigma', 2)
        self.gradient_threshold = options.get('gradient_threshold', 5)
        self.line_thickness = options.get('line_thickness', 4)
        self.offset = options.get('offset', 10)
        self.band_properties_list = []  # To store properties of all bands

    def load_image(self, image_path):
        """ Load and set image for processing. """
        self.image_base_name = os.path.basename(image_path)
        try:
            image = np.array(Image.open(image_path))
            if len(image.shape) == 3 and image.shape[2] in [3, 4]:  # Check for 3 or 4 channels
                self.image = image[:, :, 0]  # Load only the first channel
                logging.info(f"Image {self.image_base_name} loaded as 3 or 4-channel, using first channel.")
            else:
                self.image = image  # Load as-is for grayscale images
                logging.info(f"Image {self.image_base_name} loaded as grayscale.")
        except Exception as e:
            logging.error(f"Failed to load image {image_path}: {e}")
            raise

    def process_all_bands(self):
        """ Process all the bands for the current image and store band properties. """
        logging.info(f"Processing image: {self.image_base_name}")
        for band_data in self.points:
            hkl = band_data.get('hkl', 'Unknown')
            logging.info(f"Processing band with hkl: {hkl}")
            P1, P2 = band_data['P1P2'][0], band_data['P1P2'][1]
            self.P1 = P1
            self.P2 = P2
            self.get_line_profile()
            self.smooth_profile()
            band_properties = self.detect_band_edges()
            band_properties.update(band_data)  # Add additional band data
            band_properties.update(self.additional_properties)
            band_properties["image_base_name"] = self.image_base_name  # Add image name for reference
            self.band_properties_list.append(band_properties)
        return self.band_properties_list

    def get_line_profile(self):
        """ Sample the line profile between points P1 and P2 using bicubic interpolation. """
        num_samples = 100
        x_vals = np.linspace(self.P1[0], self.P2[0], num_samples)
        y_vals = np.linspace(self.P1[1], self.P2[1], num_samples)
        coords = np.vstack((y_vals, x_vals))
        self.line_profile = map_coordinates(self.image, coords, order=3)
        return self.line_profile

    def smooth_profile(self):
        """ Smooth the line profile to reduce noise. """
        if self.line_profile is None:
            self.get_line_profile()
        self.smoothed_profile = gaussian_filter1d(self.line_profile, sigma=self.smoothing_sigma)
        return self.smoothed_profile

    def detect_band_edges(self):
        """ Detect the start and end of the band based on the gradient threshold. """
        gradient = np.abs(np.gradient(self.smoothed_profile))
        logging.debug(f"Gradient calculated: {gradient}")

        midpoint = len(gradient) // 2
        band_start_half = gradient[:midpoint]
        band_end_half = gradient[midpoint:]

        self.band_start = next((i for i, g in enumerate(band_start_half) if g > self.gradient_threshold), None)
        reversed_band_end_half = band_end_half[::-1]
        band_end_local_reversed = next((i for i, g in enumerate(reversed_band_end_half) if g > self.gradient_threshold), None)

        if band_end_local_reversed is not None:
            band_end_local = len(band_end_half) - band_end_local_reversed - 1
            self.band_end = band_end_local + midpoint
        else:
            self.band_end = None

        x_coords = np.linspace(self.P1[0], self.P2[0], len(self.line_profile))
        y_coords = np.linspace(self.P1[1], self.P2[1], len(self.line_profile))

        if self.band_start is None or self.band_end is None:
            logging.warning(f"Band edges not detected for band with P1: {self.P1}, P2: {self.P2}")
            self.band_start_xy = (x_coords[0], y_coords[0])
            self.band_end_xy = (x_coords[-1], y_coords[-1])
            self.band_start = 0
            self.band_end = len(x_coords)
            self.band_width = np.abs(self.band_start - self.band_end)
            self.band_detection_status = False
        else:
            self.band_start_xy = (x_coords[self.band_start], y_coords[self.band_start])
            self.band_end_xy = (x_coords[self.band_end], y_coords[self.band_end])
            self.band_width = np.sqrt(np.sum((np.array(self.band_start_xy) - np.array(self.band_end_xy)) ** 2))
            self.band_detection_status = True
            logging.info(f"Band edges detected. Band width: {self.band_width}")

        band_properties = {
            'band_width': self.band_width,
            'band_Start_End': (self.band_start, self.band_end),
            'band_start_xy_end_xy': (self.band_start_xy, self.band_end_xy),
            'bandDetectionStatus': self.band_detection_status
        }
        return band_properties

    def plot_image_with_bands(self, ax_img):
        """Plot the image with detected bands and annotations."""
        ax_img.imshow(self.image, cmap='gray')

        colormap = plt.get_cmap('tab10', len(self.points))

        for i, (band_data, band_properties) in enumerate(zip(self.points, self.band_properties_list)):
            P1, P2 = band_data['P1P2'][0], band_data['P1P2'][1]
            hkl_label = band_data['hkl']
            color = colormap(i)

            # Plot the line between P1 and P2
            ax_img.plot([P1[0], P2[0]], [P1[1], P2[1]], color='black', linewidth=self.line_thickness)

            # Annotate the hkl label and band width
            midpoint_x = (P1[0] + P2[0]) / 2 + self.offset
            midpoint_y = (P1[1] + P2[1]) / 2 + self.offset
            band_width_rounded = round(band_properties['band_width'], 2)
            ax_img.text(midpoint_x, midpoint_y, f"hkl: {hkl_label}\nw_hkl: {band_width_rounded}",
                        color='white', fontsize=14,
                        bbox=dict(facecolor='black', edgecolor='none', pad=2))

            # Plot the start and end points of the band
            ax_img.plot(band_properties['band_start_xy_end_xy'][0][0], band_properties['band_start_xy_end_xy'][0][1],
                        'go', markersize=8)
            ax_img.plot(band_properties['band_start_xy_end_xy'][1][0], band_properties['band_start_xy_end_xy'][1][1],
                        'ro', markersize=8)

        ax_img.set_title("Image with Detected Bands")

    def plot_line_profiles(self, ax_profile):
        """Plot the raw and smoothed line profiles for all bands."""
        colormap = plt.get_cmap('tab10', len(self.points))

        for i, (band_data, band_properties) in enumerate(zip(self.points, self.band_properties_list)):
            hkl_label = band_data['hkl']
            self.P1 = band_data['P1P2'][0]
            self.P2 = band_data['P1P2'][1]
            color = colormap(i)

            raw_profile = self.get_line_profile()
            smoothed_profile = self.smooth_profile()

            ax_profile.plot(raw_profile, label=f"Raw hkl {hkl_label}", linewidth=1, linestyle='-', color=color)
            ax_profile.plot(smoothed_profile, label=f"Smoothed hkl {hkl_label}", linewidth=2, linestyle='--',
                            color=color)

            ax_profile.axvline(band_properties['band_Start_End'][0], color='g', linestyle='--')
            ax_profile.axvline(band_properties['band_Start_End'][1], color='r', linestyle='--')

        ax_profile.set_title("Raw and Smoothed Line Profiles")
        ax_profile.set_xlabel("Pixel Position")
        ax_profile.set_ylabel("Intensity")
        ax_profile.legend()

    def plot_gradients(self, ax_gradient):
        """Plot the gradients of the smoothed line profiles for all bands."""
        colormap = plt.get_cmap('tab10', len(self.points))

        for i, (band_data, band_properties) in enumerate(zip(self.points, self.band_properties_list)):
            hkl_label = band_data['hkl']
            self.P1 = band_data['P1P2'][0]
            self.P2 = band_data['P1P2'][1]
            color = colormap(i)

            self.get_line_profile()
            self.smooth_profile()
            gradient = np.gradient(self.smoothed_profile)

            ax_gradient.plot(gradient, label=f"hkl {hkl_label}", color=color)

            ax_gradient.axvline(band_properties['band_Start_End'][0], color='g', linestyle='--')
            ax_gradient.axvline(band_properties['band_Start_End'][1], color='r', linestyle='--')

        ax_gradient.set_title("Gradient of Smoothed Line Profiles")
        ax_gradient.set_xlabel("Pixel Position")
        ax_gradient.set_ylabel("Gradient")
        ax_gradient.legend()

    def plot_results(self):
        """Plot the results in a 2x2 grid layout, with the image spanning both rows in the first column."""
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1])

        # Image occupies both rows in the first column
        ax_img = plt.subplot(gs[:, 0])
        self.plot_image_with_bands(ax_img)

        # Line profile occupies the first row in the second column
        ax_profile = plt.subplot(gs[0, 1])
        self.plot_line_profiles(ax_profile)

        # Gradient plot occupies the second row in the second column
        ax_gradient = plt.subplot(gs[1, 1])
        self.plot_gradients(ax_gradient)
        fig.suptitle(f"Results for Image: {self.image_base_name}", fontsize=16)
        plt.tight_layout()
        plt.show()

def load_options(yml_path):
    """ Load options from a YAML file. """
    with open(yml_path, 'r') as file:
        options = yaml.safe_load(file)
    return options


def get_output_folder(options, default_output_folder="../../tmp/"):
    """Retrieve the output folder from the options file or use the default."""
    output_folder = options.get('output_folder', default_output_folder)
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
    return output_folder


def setup_logging(output_folder):
    """Set up logging to save logs to the specified output folder."""
    logging.basicConfig(
        level=logging.INFO,  # Set log level to INFO
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_folder, "bandDetector.log")),  # Log to file in output folder
            logging.StreamHandler()  # Also log to console
        ]
    )


def get_image_folder_base_name(folder_path):
    """Extract the base name of the image folder (the last directory in the path)."""
    return os.path.basename(os.path.normpath(folder_path))


def export_hkl_band_properties(df, output_folder, base_name):
    """Export band properties for each 'hkl' into separate Excel files."""
    # Get the unique hkl values
    unique_hkl_values = df['hkl'].unique()

    for hkl_value in unique_hkl_values:
        # Filter the DataFrame for the specific 'hkl' value
        hkl_df = df[df['hkl'] == hkl_value]

        # Save to an Excel file with the name {base_name}_{hkl}_band_properties.xlsx
        file_path = os.path.join(output_folder, f"{base_name}_{hkl_value}_band_properties.xlsx")
        hkl_df.to_excel(file_path, index=False)

        logging.info(f"Saved {hkl_value} band properties to {file_path}")

def load_band_input_data(json_file_path):
    """Load band input data from a JSON file."""
    with open(json_file_path, 'r') as f:
        band_input_data = json.load(f)
    return band_input_data


def process_images_from_json(json_data, options):
    """Process each image mentioned in the JSON data."""
    all_band_properties = []
    input_image_folder = options.get('input_image_folder', '../../tmp/')  # Get input folder from options

    logging.info(f"Using input image folder: {input_image_folder}")

    for entry in json_data:
        pattern_file = entry['patternFileName']
        image_path = os.path.join(input_image_folder, pattern_file)

        logging.info(f"Processing image: {image_path} for grainId: {entry['grainId']}")

        detector = BandWidthDetector(entry, options)

        try:
            detector.load_image(image_path)  # Load the specific image mentioned in the JSON file
            band_properties_list = detector.process_all_bands()  # Process bands for this image
            all_band_properties.extend(band_properties_list)

            if options.get('plot_results', False):  # Plot results if enabled in options
                detector.plot_results()

        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")

    # Return the collected band properties
    return pd.DataFrame(all_band_properties)


if __name__ == "__main__":
    # Load the YAML configuration
    options = load_options("bandDetectorOptions.yml")

    # Load band input data from JSON file
    band_input_data = load_band_input_data(r"banInputData.json")

    # Process each image from the JSON file
    df = process_images_from_json(band_input_data, options)

    # Get output folder from options or default to ../../tmp/
    output_folder = get_output_folder(options)

    # Get the base name for output files (from the input image folder)
    base_name = get_image_folder_base_name(options['input_image_folder'])

    # Save the overall DataFrame to Excel with the base name
    overall_file_path = os.path.join(output_folder, f"{base_name}_band_properties.xlsx")
    df.to_excel(overall_file_path, index=False)
    logging.info(f"Results saved to {overall_file_path}")

    # Call the method to export properties of each hkl band to separate files
    export_hkl_band_properties(df, output_folder, base_name)