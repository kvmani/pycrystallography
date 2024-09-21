import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import scipy.ndimage as ndimage
from PIL import Image

class BandWidthDetector:
    def __init__(self, image, points, smoothing_sigma=2):
        """
        Initialize the detector with an image and list of point pairs (P1, P2).
        """
        self.image = image
        self.points = points  # List of tuples [(P1, P2), ...]
        self.smoothing_sigma = smoothing_sigma
        self.band_properties_list = []  # To store properties of all bands

    def process_all_bands(self, gradient_threshold=5):
        """
        Process all the bands by iterating over the list of point pairs (P1, P2).
        Stores the properties of each band in band_properties_list.
        """
        for P1, P2 in self.points:
            self.P1 = P1
            self.P2 = P2
            self.get_line_profile()
            self.smooth_profile()
            band_properties = self.detect_band_edges(gradient_threshold=gradient_threshold)
            self.band_properties_list.append(band_properties)
        return self.band_properties_list

    def get_line_profile(self):
        """
        Sample the line profile between points P1 and P2.
        """
        x_vals = np.linspace(self.P1[0], self.P2[0], 100).astype(int)
        y_vals = np.linspace(self.P1[1], self.P2[1], 100).astype(int)
        self.line_profile = self.image[y_vals, x_vals]
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
        - BandStart is detected from the first half of the gradient.
        - BandEnd is detected from the last half of the gradient, but we look for the last point that exceeds the threshold.
        This is done by reversing the second half of the gradient array and detecting the first point in the reversed array.
        """
        gradient = np.abs(np.gradient(self.smoothed_profile))

        # Split the gradient array into two halves
        midpoint = len(gradient) // 2

        # Detect BandStart: Search only in the first half of the gradient
        band_start_half = gradient[:midpoint]
        self.band_start = next((i for i, g in enumerate(band_start_half) if g > gradient_threshold), None)

        # Detect BandEnd: Search only in the last half of the gradient (reversed)
        band_end_half = gradient[midpoint:]
        reversed_band_end_half = band_end_half[::-1]  # Reverse the second half to find the last occurrence

        # Detect the first point in the reversed array where the gradient exceeds the threshold
        band_end_local_reversed = next((i for i, g in enumerate(reversed_band_end_half) if g > gradient_threshold),
                                       None)

        # Adjust band_end to refer to the correct index in the full gradient array
        if band_end_local_reversed is not None:
            band_end_local = len(band_end_half) - band_end_local_reversed - 1  # Convert reversed index to original
            self.band_end = band_end_local + midpoint
        else:
            self.band_end = None

        # Calculate the corresponding image coordinates of BandStart and BandEnd
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

    def plot_results(self):
        """
        Plot the results of all bands on the same image.
        Shows the detected BandStart and BandEnd points for each band.
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        # Plot 1 (left): The image with BandStart, BandEnd points, and the lines (P1-P2)
        axs[0].imshow(self.image, cmap='gray')

        for P, band_properties in zip(self.points, self.band_properties_list):
            # Plot the line between P1 and P2
            P1,P2 = P[0],P[1]
            axs[0].plot([P1[0], P2[0]], [P1[1], P2[1]], 'y-', label="Profile Line", linewidth=2)

            # Plot BandStart and BandEnd points (green and red dots)
            axs[0].plot(band_properties['band_start_xy_end_xy'][0][0], band_properties['band_start_xy_end_xy'][0][1],
                        'go', markersize=8, label="Detected Band Start")
            axs[0].plot(band_properties['band_start_xy_end_xy'][1][0], band_properties['band_start_xy_end_xy'][1][1],
                        'ro', markersize=8, label="Detected Band End")

        axs[0].set_title("Image with Band Start, End Points, and Profile Lines")
        axs[0].legend()

        # Plot 2 (right): The smoothed line profiles for all bands
        for P, band_properties in zip(self.points, self.band_properties_list):
            P1, P2 = P[0], P[1]
            self.P1 = P1
            self.P2 = P2
            self.get_line_profile()
            self.smooth_profile()
            axs[1].plot(self.smoothed_profile, label=f"Profile Line ({P1}-{P2})", linewidth=2)
            axs[1].axvline(band_properties['band_Start_End'][0], color='g', linestyle='--', label="Band Start")
            axs[1].axvline(band_properties['band_Start_End'][1], color='r', linestyle='--', label="Band End")

        axs[1].set_title("Smoothed Line Profiles with Detected Band Edges")
        axs[1].set_xlabel("Pixel Position")
        axs[1].set_ylabel("Intensity")
        axs[1].legend()

        plt.tight_layout()
        plt.show()

def create_test_image(size=100, band_width=10, noise_level=10, smoothing_sigma=2):
        """
        Create a test image with a vertical band, rotate it by 45 degrees,
        then apply Gaussian smoothing and noise.
        """
        image = np.zeros((size, size))

        # Create a vertical band of intensity 100
        center = size // 2
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
image = np.array(Image.open(r"../../data/testingData/ML_kikuchi_test_1.png"))[:, :, 0]

points = [
    [(154, 130), (163, 152)],
    [(81, 63), (90, 48)],
    [(41, 100), (61,100)],
    [(107, 171), (102,184)],
]

# Instantiate the detector with multiple point pairs
detector = BandWidthDetector(image, points)
band_properties_list = detector.process_all_bands()  # Processes all bands

# Optionally, plot the results for each band
detector.plot_results()

# Print the band properties list
print(band_properties_list)
