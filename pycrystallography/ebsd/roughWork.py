import numpy as np


def create_flip_image(m, n):
    # Create a random 1D array of size m*n*3 representing RGB image
    random_rgb_array = np.random.randint(0, 256, size=(m * n * 3), dtype=np.uint8)
    random_rgb_array[1:1500]=0

    # Reshape the random 1D array to a 3D RGB image
    rgb_image = random_rgb_array.reshape(m, n, 3)

    # Flip the image horizontally
    flipped_image = np.flip(rgb_image, axis=0)

    # Reshape the flipped 3D image back to a 1D array
    flipped_rgb_array = flipped_image.reshape(-1)

    return flipped_rgb_array


m = 100  # Height of the image
n = 50  # Width of the image

flipped_rgb_array = create_flip_image(m, n)
print(flipped_rgb_array)
