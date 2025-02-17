import glob

import PIL
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
from imgaug import parameters as iap
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from skimage.color import label2rgb
from imgaug import SegmentationMapsOnImage
from imgaug import parameters as iap
from scipy.ndimage import binary_dilation
from pycrystallography.utilities import graphicUtilities as gut
from PIL import Image
import os
import random
import h5py
import pandas as pd
from tqdm import tqdm
sometimes = lambda p, aug : iaa.Sometimes(p, aug)

listOfDataPointsData=[]

# def is_numeric(s):
#     # Check if the string contains only numeric characters (0-9)
#     return all(char.isdigit() for char in s)
#
# def is_alphanumeric(s):
#     # Check if the string contains only alphanumeric characters (letters and digits)
#     return s.isalnum()
#
# def test_string_type(s):
#     if is_numeric(s):
#         print(f'The string "{s}" contains only numeric data.')
#     elif is_alphanumeric(s):
#         print(f'The string "{s}" contains alphanumeric data.')
#     else:
#         print(f'The string "{s}" does not contain only numeric or alphanumeric data.')
#
# # Test cases
# test_string_type("12345 34.5")        # Output: The string "12345" contains only numeric data.
# test_string_type("Hello123")     # Output: The string "Hello123" contains alphanumeric data.
# test_string_type("abc")          # Output: The string "abc" contains alphanumeric data.
# test_string_type("Hello, World")


# def create_mask(mask_size, pixel_fraction):
#     """
#     :param mask_size: size of the mask
#     :param pixel_fraction: no of pixels to be turned off is decided by pixel_fraction
#     :return :erroded mask
#     """
#     mask = np.ones(mask_size, dtype=np.uint8)  # Create a mask with all ones
#
#     total_pixels = mask_size[0] * mask_size[1]
#     pixels_to_turn_off = int(pixel_fraction * total_pixels)
#     flat_mask = mask.ravel()
#     flat_mask[:pixels_to_turn_off] = 0
#     np.random.shuffle(flat_mask)
#
#     # return flat_mask.reshape(mask_size)
#     kernel = np.ones((5, 5), np.uint8)
#     erroded_mask=flat_mask.reshape(mask_size)
#     erroded_mask = cv2.erode(erroded_mask, kernel, iterations=2)
#     return erroded_mask
def unionMask(img1,img2):
    img2=img2.resize(img1.size)
    img1 = np.array(img1)
    img2 = np.array(img2)
    unionMask=np.bitwise_or(img1, img2)
    return unionMask
def addMask(img1,img2):
    aimg1 = np.array(img1)
    aimg2 = np.array(img2)
    result_array = aimg1 + aimg2
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)

    # addMaskresult = Image.fromarray(result_array)
    addMaskresult=result_array
    # Save the result
    # cv2.imwrite('result.jpg', addMaskresult)
    return addMaskresult
def subMask(img1,img2):
    aimg1 = np.array(img1)
    aimg2 = np.array(img2)
    result_array = aimg1 - aimg2
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)
    subMaskresult=result_array
    # subMaskresult = Image.fromarray(result_array)

    return subMaskresult
def visit_hdf5_group(group, indent=""):

    for name, item in group.items():
        if isinstance(item, h5py.Group):
            print(f"{indent}Group: {name}")
            visit_hdf5_group(item, indent + "  ")
        elif isinstance(item, h5py.Dataset):
            if r"DATA" in name:
                print(f"{indent}Dataset: {name}")
                # dataset = item
                # data = dataset[()]
                nPoints = item.shape[0]
                listOfDataPointsData.append({"dataset_location": item.name,
                                             "nPoints": nPoints})
    return  listOfDataPointsData


            # You can add code here to read or process the dataset if needed.


def read_ang_file(ang_file):
    NCOLS_ODD = 0
    NROWS = 0
    try:

        with open(ang_file, 'r') as file:
            for line in file:
                if "NCOLS_ODD" in line:
                    NCOLS_ODD = int(line.split(':')[1].strip())
                elif "NROWS" in line:
                    NROWS = int(line.split(':')[1].strip())

                    if NCOLS_ODD>0 and NROWS>0:
                        break

    except FileNotFoundError:
        print(f"The file '{ang_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    nPoints = NCOLS_ODD * NROWS
    return nPoints, NCOLS_ODD, NROWS



def augmentEBSD(images, augmentationParameters, nImages=10, seed=None):
    """
    wrapper method for image augmentation.
    input should be in the form of list of tuple of image and its segmentation mask (labelled matrix)
    out put will be list of tuple of images (aug_image, aug_segMask,augSegMaskPix2Pix) of length nImages
    """

    # assert len(images)==len(segMasks), f"Unequal number of images and masks are supplied: {len(images)}; {len(segmasks)}"
    if seed is not None:
        ia.seed(seed)
    seq = iaa.Sequential([
        iaa.Sequential([

            iaa.SomeOf((0, None), [
                iaa.Fliplr(augmentationParameters["FliplrProb"]),  # horizontally flip 50% of all images
                iaa.Flipud(augmentationParameters["FlipudProb"]),  # vertically flip 20% of all images
                 # sometimes(1.0, iaa.GaussianBlur(sigma=(1.0,9.0))),
                sometimes(augmentationParameters["BlurGroupProb"],
                iaa.OneOf([
                    iaa.MotionBlur(k=(15, 28), angle=[-60, 60]),
                    iaa.MedianBlur(k=(13, 17)),
                     iaa.GaussianBlur(sigma=(1.0, 9.0)),
                   iaa.WithHueAndSaturation(iaa.WithChannels(1, iaa.Add((5, 100)))),
                     iaa.Dropout((0.01, 0.1), per_channel=0.5),
                ])),
                sometimes(augmentationParameters["ContrastAndScaleGroupProb"],
                iaa.SomeOf((0, None), [
                    iaa.AllChannelsHistogramEqualization(5),  ###very good
                     iaa.Canny(alpha=(0.1, 0.5)), ## good for optical images
                    iaa.ScaleX((1.5, 3.5)),
                    iaa.ScaleY((1.5, 3.5)),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                    iaa.LinearContrast((2.4, 9.6)),
                    iaa.WithBrightnessChannels(iaa.Add((-150, 150))),
                    iaa.Sharpen(alpha=1.0),
                    iaa.HistogramEqualization(),
                    iaa.Jigsaw(nb_rows=10, nb_cols=10),  ### very good, also ver good for boundary augmentor
                    iaa.BlendAlphaCheckerboard(nb_rows=40, nb_cols=(50, 80), foreground=iaa.AddToHue((-200, 200))),

                    iaa.PerspectiveTransform(scale=(0.01, 0.1)),
                    iaa.ElasticTransformation(alpha=50.0, sigma=5.0),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    iaa.Solarize(1.0, threshold=(90, 100)),  ## good
                 ])),

            ]),
            sometimes(augmentationParameters["NoiseGroupProb"],
            iaa.SomeOf((0, None), [

                iaa.ImpulseNoise(1.0),#### this is good but we need to convert images back to gray scale

                iaa.AdditivePoissonNoise(lam=40.0),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
              iaa.BlendAlphaFrequencyNoise(
                   foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)),
                 iaa.MultiplyElementwise((1.5, 2.5)),
                 iaa.Affine(rotate=(-45, 45), scale=(0.8, 1.2)),

                iaa.SaltAndPepper((0.1, 0.8)),



             iaa.ImpulseNoise((0.1,0.5)),

            ],
            # sometimes(0.4, iaa.Affine(
            #    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            #      translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            #     rotate=(-45, 45),
            #    shear=(-16, 16), order=[0, 1],
            #    cval=(0, 255), mode=ia.ALL
            # )),

                       # do all of the above augmentations in random order
                       random_order=True

                       )),


        ],
        # do all of the above augmentations in random order
        random_order=True),
       #iaa.CropToFixedSize(height=512,width=512)
         ]
    )
    augImages = []
    count=0
    for j in tqdm(range(int(nImages/len(images))+1),desc="Augmentor"):
        for i, _ in enumerate(images):
            count+=1
            # segmap = SegmentationMapsOnImage(images[i][1], shape=images[i][0].shape)
            images_aug_i = seq(image=images[i])
            # segMapPix2Pix = (label2rgb(segmaps_aug_i.get_arr(),colors=[[0,255,0],[255,0,0]],bg_label=0)).astype(np.uint8)
            # #segMapPix2Pix = ut.segMask2RgbImage(segmaps_aug_i.get_arr())
            # segmaps_aug_i = segmaps_aug_i.get_arr()
            augImages.append(images_aug_i)
            if count >= nImages:
                break

    augImages = augImages[0:nImages]
    # assert len(augImages) == nImages, f"Problem in removing extra number of images !!number of images :{count}, desired:{nImages}"


    return augImages



if __name__ == "__main__":
    # create_mask=False
    # if create_mask:
    #  mask_size=(256,256)
    # pixel_fraction=0.005
    # mask = create_mask(mask_size,pixel_fraction)
    # print(mask)
    # plt.imshow(mask)
    # plt.show()
    # print("Completed the image splitting now exiting with exit code -100")
    # exit(-100)
    testAugmentEBSD = False
    if testAugmentEBSD:
        augmentationParameters = {
            "FliplrProb": 0.0,
            "FlipudProb": 0.0,
            "BlurGroupProb": 0.5,
            "ContrastAndScaleGroupProb": 1.0,
            "NoiseGroupProb":0.5,

        }
        baseImagePath = r"C:\Users\aa2905\OneDrive - UNT System\Amrutha\EBSD_ML\ebsd_masks\1.png"
        images = [np.stack([np.array(Image.open(baseImagePath))]*3, axis=-1), ]
        augImages = augmentEBSD(images, nImages=5, seed=10, augmentationParameters=augmentationParameters)
        for augset in augImages:
            threshold_value = 50
            #plt.imshow(augset,cmap="gray")
            binary_mask = np.where(augset[:,:,0] > threshold_value, 1, 0)
            gut.plotComparitiveImages([augset,binary_mask],titles=['gray scale image','mask'])
            #plt.show()

        print("Done")

    testAddTwoMasks=False
    if testAddTwoMasks:



        img1 = cv2.imread(r"C:\Users\aa2905\OneDrive - UNT System\Amrutha\EBSD_ML\ebsd_MaskInput\Figure_1.png")
        img2 = cv2.imread(r"C:\Users\aa2905\OneDrive - UNT System\Amrutha\EBSD_ML\ebsd_MaskInput\Figure_2.png")
        addMaskresult=addMask(img1,img2)
        threshold_value = 50
        binary_mask = np.where(addMaskresult[:, :, 0] > threshold_value, 1, 0)
        gut.plotComparitiveImages([addMaskresult, binary_mask], titles=['gray scale image', 'mask'])
        addMaskresult.save(r"C:\Users\aa2905\OneDrive - UNT System\Amrutha\EBSD_ML\ebsd_MaskResults\AFigure_1.png")
        # addMaskresult.show()


    testsubTwoMasks = False
    if testsubTwoMasks:
        img1 = cv2.imread(r"C:\Users\aa2905\OneDrive - UNT System\Amrutha\EBSD_ML\ebsd_MaskInput\Figure_1.png")
        img2 = cv2.imread(r"C:\Users\aa2905\OneDrive - UNT System\Amrutha\EBSD_ML\ebsd_MaskInput\Figure_2.png")
        subMaskresult = subMask(img1, img2)
        threshold_value = 50
        binary_mask = np.where(subMaskresult[:, :, 0] > threshold_value, 1, 0)
        gut.plotComparitiveImages([subMaskresult, binary_mask], titles=['gray scale image', 'mask'])
        subMaskresult.save(r"C:\Users\aa2905\OneDrive - UNT System\Amrutha\EBSD_ML\ebsd_MaskResults\SFigure_1.png")
        subMaskresult.show()

    testunionTwoMasks = False
    if testunionTwoMasks:
        folder_path = r"C:\Users\aa2905\OneDrive - UNT System\Amrutha\EBSD_ML\ebsd_MaskInput"
        # img1 = Image.open(r"C:\Users\aa2905\OneDrive - UNT System\Amrutha\EBSD_ML\ebsd_Addmasks\2.png")
        # img2 = Image.open(r"C:\Users\aa2905\OneDrive - UNT System\Amrutha\EBSD_ML\ebsd_Addmasks\34.png")
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        # Check if there are at least two image files in the folder
        if len(image_files) < 2:
            print("There are not enough images in the folder.")
        else:
            # Select two random images from the list
            random_images = random.sample(image_files, 2)
        image1_path = os.path.join(folder_path, random_images[0])
        image2_path = os.path.join(folder_path, random_images[1])
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)
        unionMaskresult=unionMask(img1, img2)
        gut.plotComparitiveImages([img1,img2, unionMaskresult], titles=['img1','img2', 'unionMask'],
                                  filePath=r"C:\Users\aa2905\OneDrive - UNT System\Amrutha\EBSD_ML\ebsd_MaskResults\UFigure_1.png")
        unionMaskresult = Image.fromarray(unionMaskresult)
        # unionMaskresult.save(r"C:\Users\aa2905\OneDrive - UNT System\Amrutha\EBSD_ML\ebsd_MaskResults\UFigure_1.png")

    testreadHDFfile = True
    if testreadHDFfile:
        # Define the file path and dataset location
        listOfDataPointsData=[]
        file_path =r'D:\Amrutha\ML Data\ML_EBSD\otherApeoData'
        # hdfFiles = [
        #     r'E:\Al-B4C-Composites\apreoData\Al6061-B4C.edaxh5',
        #     #r'E:\Al-B4C-Composites\apreoData\Al - b4c - 2ndProject_old.edaxh5'
        # ]
        #
        # # dataset_locations = [
        # #     r'/Al6061-B4C/A5-YZ cross section/Y-Z cross section Area 4/OIM Map 1',
        # #     r'/Al6061-B4C/A5-YZ cross section/A-X-y Cross Section/200X_OIM Map 1',
        # #                      ]
        #
        # dataset_location = r'/Al6061-B4C/A5-YZ cross section/A-X-y Cross Section/200X_OIM Map 1'
        #
        # for item in hdfFiles:
        #     # dataset_location = item
        #     # suffix =  r'/EBSD/ANG/DATA/DATA'
        #     listOfDataPointsDataFromFile=readHDFfile(item)
        #     listOfDataPointsData.extend(listOfDataPointsDataFromFile)
        #     print(listOfDataPointsData)
        # df = pd.DataFrame.from_records(listOfDataPointsData)
        # df.to_excel('ebsdNpoints.xlsx',index=False)
        # print(df)

        ang_files =  glob.glob(os.path.join(file_path, '*.ang'))
        for ang_file in ang_files:
            print(f"procesing {ang_file}")
            nPoints, NCOLS_ODD, NROWS = read_ang_file(ang_file)
            listOfDataPointsData.append({"dataset_location": ang_file,
                                 "nPoints": nPoints,
                                       "NCOLS_ODD":NCOLS_ODD, "NROWS":NROWS })
    df = pd.DataFrame.from_records(listOfDataPointsData)
    df.to_excel('ebsdNpoints.xlsx',index=False)
    df.to_json('ebsdNpoints.json')
    print(df)
