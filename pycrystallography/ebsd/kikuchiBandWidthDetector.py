import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import scipy.ndimage as ndimage
from PIL import Image


class BandWidthDetector:
    def __init__(self, image, P1, P2, smoothing_sigma=2):
        """
        Initialize the detector with an image and points P1, P2.
        """
        self.image = image
        self.P1 = P1
        self.P2 = P2
        self.smoothing_sigma = smoothing_sigma
        self.line_profile = None
        self.smoothed_profile = None
        self.band_start = None
        self.band_end = None

    def get_line_profile(self):
        """
        Sample the line profile between points P1 and P2.
        """
        # Linear interpolation of points between P1 and P2
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
        The BandStart is the first point where the absolute value of the gradient
        exceeds the threshold, and BandEnd is the last such point.
        Also returns the corresponding (x, y) image coordinates of these points.
        """
        gradient = np.abs(np.gradient(self.smoothed_profile))

        # Detect BandStart: First point where gradient exceeds the threshold
        self.band_start = next((i for i, g in enumerate(gradient) if g > gradient_threshold), None)

        # Detect BandEnd: Last point where gradient exceeds the threshold
        self.band_end = next((i for i, g in reversed(list(enumerate(gradient))) if g > gradient_threshold), None)

        if self.band_start is None or self.band_end is None:
            print("Warning: Band edges not detected based on the given threshold.")
            return None, None

        # Compute the corresponding image coordinates of BandStart and BandEnd
        # Using linear interpolation between P1 and P2
        x_coords = np.linspace(self.P1[0], self.P2[0], len(self.line_profile))
        y_coords = np.linspace(self.P1[1], self.P2[1], len(self.line_profile))

        self.band_start_xy = (x_coords[self.band_start], y_coords[self.band_start])
        self.band_end_xy = (x_coords[self.band_end], y_coords[self.band_end])

        return (self.band_start, self.band_end), (self.band_start_xy, self.band_end_xy)

    def plot_results(self):
        """
        Plot the original image with detected BandStart and BandEnd,
        including the line (P1-P2) and BandStart, BandEnd as green and red points.
        Also plot the original and smoothed line profiles, and the gradient of the profile.
        """
        fig, axs = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [1, 1.5]})

        # Plot 1 (left): The image with BandStart, BandEnd points, and the line (P1-P2)
        axs[0, 0].imshow(self.image, cmap='gray')
        axs[0, 0].plot([self.P1[0], self.P2[0]], [self.P1[1], self.P2[1]], 'y-', label="Profile Line (P1-P2)",
                       linewidth=2)

        # Plot BandStart and BandEnd points (green and red dots)
        if self.band_start_xy and self.band_end_xy:
            axs[0, 0].plot(self.band_start_xy[0], self.band_start_xy[1], 'go', markersize=8,
                           label="Detected Band Start")
            axs[0, 0].plot(self.band_end_xy[0], self.band_end_xy[1], 'ro', markersize=8, label="Detected Band End")

        axs[0, 0].set_title("Image with Band Start, End Points, and Profile Line")
        axs[0, 0].legend()

        # Plot 2 (top-right): The original and smoothed line profiles
        axs[0, 1].plot(self.line_profile, label="Original Profile", alpha=0.6)
        axs[0, 1].plot(self.smoothed_profile, label="Smoothed Profile", linewidth=2)
        axs[0, 1].axvline(self.band_start, color='g', linestyle='--', label="Band Start")
        axs[0, 1].axvline(self.band_end, color='r', linestyle='--', label="Band End")
        axs[0, 1].set_title("Line Profile with Detected Band Edges")
        axs[0, 1].set_xlabel("Pixel Position")
        axs[0, 1].set_ylabel("Intensity")
        axs[0, 1].legend()

        # Plot 3 (bottom-right): The gradient of the smoothed profile
        gradient = np.gradient(self.smoothed_profile)
        axs[1, 1].plot(gradient, label="Gradient of Smoothed Profile", color='orange')
        axs[1, 1].axvline(self.band_start, color='g', linestyle='--', label="Band Start")
        axs[1, 1].axvline(self.band_end, color='r', linestyle='--', label="Band End")
        axs[1, 1].set_title("Gradient of the Line Profile")
        axs[1, 1].set_xlabel("Pixel Position")
        axs[1, 1].set_ylabel("Gradient")
        axs[1, 1].legend()

        # Hide empty bottom-left plot
        axs[1, 0].axis('off')

        plt.tight_layout()
        plt.show()


# Test with artificial image
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

image = create_test_image()
image = np.array(Image.open(r"../../data/testingData/ML_kikuchi_test_1.png"))[:,:,0]
#detector = BandWidthDetector(image, P1=(40, 60), P2=(59, 39))
points = [
    (154, 130), (163,152),
    (154, 130), (163,152),
          ]
for i in range (2):
    detector = BandWidthDetector(image, P1=(154,130), P2=(163, 152))
    detector.get_line_profile()
    detector.smooth_profile()
    detector.detect_band_edges()
    detector.plot_results()
