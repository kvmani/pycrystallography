import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import scipy.ndimage as ndimage
from PIL import Image
from scipy.ndimage import map_coordinates
import matplotlib.gridspec as gridspec

class BandWidthDetector:
    def __init__(self, band_input_data, smoothing_sigma=2):
        """
        Initialize the detector with the image, list of points, and additional properties.
        """
        self.image_path = band_input_data.get("imagePath")
        self.points = band_input_data.get("points")  # This is now a list of dictionaries
        self.additional_properties = {k: v for k, v in band_input_data.items() if k not in ["imagePath", "points"]}
        self.smoothing_sigma = smoothing_sigma
        self.image = np.array(Image.open(self.image_path))[:, :, 0]
        self.band_properties_list = []  # To store properties of all bands

    def process_all_bands(self, gradient_threshold=3):
        """
        Process all the bands by iterating over the list of point dictionaries (with hkl, P1P2, refWidth).
        Stores the properties of each band in band_properties_list.
        """
        for band_data in self.points:
            P1, P2 = band_data['P1P2'][0], band_data['P1P2'][1]
            self.P1 = P1
            self.P2 = P2
            self.get_line_profile()
            self.smooth_profile()
            band_properties = self.detect_band_edges(gradient_threshold=gradient_threshold)

            # Add additional attributes from band_data (hkl, refWidth, etc.)
            band_properties.update(band_data)
            band_properties.update(self.additional_properties)
            band_properties["imagePath"] = self.image_path  # Add the image path

            self.band_properties_list.append(band_properties)
        return self.band_properties_list

    def get_line_profile(self):
        """
        Sample the line profile between points P1 and P2 using bicubic interpolation.
        """
        num_samples = 100  # Number of samples along the line
        x_vals = np.linspace(self.P1[0], self.P2[0], num_samples)
        y_vals = np.linspace(self.P1[1], self.P2[1], num_samples)

        coords = np.vstack((y_vals, x_vals))
        self.line_profile = map_coordinates(self.image, coords, order=3)

        return self.line_profile

    def smooth_profile(self):
        """
        Smooth the line profile to reduce noise.
        """
        if self.line_profile is None:
            self.get_line_profile()
        self.smoothed_profile = gaussian_filter1d(self.line_profile, sigma=self.smoothing_sigma)
        return self.smoothed_profile

    def detect_band_edges(self, gradient_threshold=5):
        """
        Detect the start and end of the band based on the gradient threshold.
        """
        gradient = np.abs(np.gradient(self.smoothed_profile))

        midpoint = len(gradient) // 2
        band_start_half = gradient[:midpoint]
        band_end_half = gradient[midpoint:]

        self.band_start = next((i for i, g in enumerate(band_start_half) if g > gradient_threshold), None)
        reversed_band_end_half = band_end_half[::-1]
        band_end_local_reversed = next((i for i, g in enumerate(reversed_band_end_half) if g > gradient_threshold), None)

        if band_end_local_reversed is not None:
            band_end_local = len(band_end_half) - band_end_local_reversed - 1
            self.band_end = band_end_local + midpoint
        else:
            self.band_end = None

        x_coords = np.linspace(self.P1[0], self.P2[0], len(self.line_profile))
        y_coords = np.linspace(self.P1[1], self.P2[1], len(self.line_profile))

        if self.band_start is None or self.band_end is None:
            print("Warning: Band edges not detected based on the given threshold.")
            self.band_start_xy = (x_coords[0], y_coords[0])
            self.band_end_xy = (x_coords[-1], y_coords[-1])
            self.band_start = 0
            self.band_end = len(x_coords)
            self.band_width = np.abs(self.band_start - self.band_end)
            self.band_detection_status = False
        else:
            self.band_start_xy = (x_coords[self.band_start], y_coords[self.band_start])
            self.band_end_xy = (x_coords[self.band_end], y_coords[self.band_end])
            self.band_width = np.sum((np.array(self.band_start_xy) - np.array(self.band_end_xy)) ** 2)
            self.band_detection_status = True

        band_properties = {
            'band_width': self.band_width,
            'band_Start_End': (self.band_start, self.band_end),
            'band_start_xy_end_xy': (self.band_start_xy, self.band_end_xy),
            'bandDetectionStatus': self.band_detection_status
        }
        return band_properties

    import matplotlib.gridspec as gridspec

    import matplotlib.gridspec as gridspec

    def plot_results(self):
        """
        Plot the results for all bands in a customized layout:
        - Plot 1 (image) spans both rows of the first column.
        - Plot 2 (line profiles) and Plot 3 (gradients) span the first and second rows of the second column.
        """
        fig = plt.figure(figsize=(14, 14))

        # Create a gridspec layout with 2 rows and 2 columns
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # Plot 1: The image will span both rows in the first column
        ax_img = plt.subplot(gs[:, 0])  # Spanning all rows in the first column
        ax_img.imshow(self.image, cmap='gray')
        for band_data, band_properties in zip(self.points, self.band_properties_list):
            P1, P2 = band_data['P1P2'][0], band_data['P1P2'][1]
            hkl_label = band_data['hkl']
            ax_img.plot([P1[0], P2[0]], [P1[1], P2[1]], 'y-', linewidth=2, label=f"hkl {hkl_label}")
            ax_img.plot(band_properties['band_start_xy_end_xy'][0][0], band_properties['band_start_xy_end_xy'][0][1],
                        'go', markersize=8)
            ax_img.plot(band_properties['band_start_xy_end_xy'][1][0], band_properties['band_start_xy_end_xy'][1][1],
                        'ro', markersize=8)

        ax_img.set_title("Image with Detected Bands")
        ax_img.legend()

        # Plot 2: The smoothed line profiles in the second column, first row
        ax_profile = plt.subplot(gs[0, 1])
        for band_data, band_properties in zip(self.points, self.band_properties_list):
            hkl_label = band_data['hkl']
            self.P1 = band_data['P1P2'][0]
            self.P2 = band_data['P1P2'][1]

            # Get the line profile and smooth it
            self.get_line_profile()
            self.smooth_profile()

            # Plot the smoothed line profile
            ax_profile.plot(self.smoothed_profile, label=f"hkl {hkl_label}", linewidth=2)

            # Add dashed vertical lines for band_start and band_end (without adding them to the legend)
            ax_profile.axvline(band_properties['band_Start_End'][0], color='g', linestyle='--')  # Band start
            ax_profile.axvline(band_properties['band_Start_End'][1], color='r', linestyle='--')  # Band end

        ax_profile.set_title("Smoothed Line Profiles")
        ax_profile.set_xlabel("Pixel Position")
        ax_profile.set_ylabel("Intensity")
        ax_profile.legend()

        # Plot 3: The gradients of the profiles in the second column, second row
        ax_gradient = plt.subplot(gs[1, 1])
        for band_data, band_properties in zip(self.points, self.band_properties_list):
            hkl_label = band_data['hkl']
            self.P1 = band_data['P1P2'][0]
            self.P2 = band_data['P1P2'][1]

            # Get the line profile and smooth it
            self.get_line_profile()
            self.smooth_profile()

            # Calculate the gradient of the smoothed profile
            gradient = np.gradient(self.smoothed_profile)

            # Plot the gradient
            ax_gradient.plot(gradient, label=f"hkl {hkl_label}", color='orange')

            # Add dashed vertical lines for band_start and band_end (without adding them to the legend)
            ax_gradient.axvline(band_properties['band_Start_End'][0], color='g', linestyle='--')  # Band start
            ax_gradient.axvline(band_properties['band_Start_End'][1], color='r', linestyle='--')  # Band end

        ax_gradient.set_title("Gradient of Line Profiles")
        ax_gradient.set_xlabel("Pixel Position")
        ax_gradient.set_ylabel("Gradient")
        ax_gradient.legend()

        # Adjust layout for better usage of screen real estate
        plt.tight_layout()
        plt.show()


def create_test_image(size=200, band_width=5, noise_level=10, smoothing_sigma=2):
        """
        Create a test image with a vertical band, rotate it by 45 degrees,
        then apply Gaussian smoothing and noise.
        """
        image = np.zeros((size, size))

        # Create a vertical band of intensity 100
        center = size // 2+ 15
        image[:, center - band_width // 2:center + band_width // 2] = 100

        # Rotate the image by 45 degrees
        rotated_image = ndimage.rotate(image, 45, reshape=False)

        # Apply Gaussian smoothing
        smoothed_image = gaussian_filter1d(rotated_image, sigma=smoothing_sigma)

        # Add random noise
        noisy_image = smoothed_image + np.random.normal(0, noise_level, smoothed_image.shape)

        return np.clip(noisy_image, 0, 255)

# Example Usage
#image = create_test_image()



bandInputdata = {
    'imagePath': r"../../data/testingData/ML_kikuchi_test_1.png",
    'points': [
        {'hkl':'110', 'P1P2':[(154, 130), (163, 152)], 'refWidth':100,},
        {'hkl':'220', 'P1P2':[(81, 63), (90, 48)], 'refWidth':120,},
        {'hkl':'111', 'P1P2':[(41, 100), (61, 100)], 'refWidth':105,},
        {'hkl':'420', 'P1P2':[(107, 171), (102, 184)], 'refWidth':105,},
    ],
    'experimentID': 'EX123',
    'description': 'Detecting bands from Kikuchi pattern'
}

# Instantiate the detector with the band input data dictionary
detector = BandWidthDetector(bandInputdata)

# Process all bands
band_properties_list = detector.process_all_bands()

# Plot the results
detector.plot_results()

# Print the band properties list (including additional properties)
print(band_properties_list)

