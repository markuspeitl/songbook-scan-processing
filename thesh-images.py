 
#from scipy import misc

from io import BytesIO
from PIL import Image
import numpy as np

import skimage
from skimage import data
from skimage import exposure
from skimage.color import rgb2gray, convert_colorspace, rgb2hsv
from skimage.util import img_as_uint
from scipy.ndimage import shift

from os import listdir
from os.path import isfile, join
import os
import argparse

save_processed_images_enabled = True
save_pdf_enabled = False
save_original_to_pdf_enabled = False
process_images_enabled = True
compress_images_enabled = True
rescale_images_enabled = True

src_directory = './png'
target_directory = './processed-pngs3'

out_pdf_path = './diamante_revised3.pdf'
white_threshold = 180
debug_offset = 50
debug_limit = 10
resize_factor = 0.3
jpg_quality = 70


if(not rescale_images_enabled):
    resize_factor = 1.0

binding_border_size = int(200 * resize_factor)
non_binding_border_size = int(10 * resize_factor)
vertical_border_size = int(50 * resize_factor)
placement_adjustment_shift_x = int(25 * resize_factor)
placement_adjustment_shift_y = int(5 * resize_factor)


border_color_intensity = 255
light_threshold_color_intensity = 255

text_color = (0,0,0)
annotation_color = (40,0,220)
#text_color = (255,0,0)
#annotation_color = (0,255,0)


def main():
    parser = argparse.ArgumentParser(
        description="Template for building python console packages"
    )

    parser.add_argument('source_dir', help="print optional text")
    parser.add_argument('target_dir', help="print optional text")
    parser.add_argument('-pdf','--pdf_name', help="Multiply passed numbers")
    parser.add_argument('-wt','--white_threshold', type=int, help="Multiply passed numbers", default=180)
    parser.add_argument('-bbs','--binding_border_size', type=int, help="Multiply passed numbers", default=200)
    parser.add_argument('-resize','--resize_factor', type=float, help="Multiply passed numbers", default=1.0)
    parser.add_argument('-o','--images_offset', type=int, help="Multiply passed numbers")
    parser.add_argument('-n','--images_amount', type=int, help="Multiply passed numbers")
    
    parser.add_argument('-c','--compress', action="store_true", help="Print module stuff")
    parser.add_argument('-np','--disable_processing', action="store_true", help="Print module stuff")

    parser.add_argument('-q','--jpg_quality', type=int, help="Multiply passed numbers")
    parser.add_argument('-f','--output_format', type=int, help="Multiply passed numbers", choices=['png', 'jpg'])


def adjust_contrast(image):
    p2, p98 = np.percentile(image, (1, 99))
    #image = exposure.rescale_intensity(image, in_range=(p2, p98))

def threshold_white(image, white_threshold_val):
    #pil_img = Image.fromarray(image)
    #pil_img = Image.fromarray(image)
    #img_gray_array = np.array(pil_img.convert('L'))
    img_gray_array = rgb2gray(image)
    image[img_gray_array > (white_threshold_val/255)] = light_threshold_color_intensity

def clear_margins(image, index):

    #image = np.roll(image, (100,100), axis=(1,0))
    

    if index % 2 != 0:
        image[:,image.shape[1]-binding_border_size:,:] = border_color_intensity
        image[:,0:non_binding_border_size,:] = border_color_intensity

        shifted = shift(image,[placement_adjustment_shift_y,placement_adjustment_shift_x,0], mode='constant', cval=255)
        np.copyto(image, shifted)
        
    else:
        image[:,image.shape[1]-non_binding_border_size:,:] = border_color_intensity
        image[:,0:binding_border_size,:] = border_color_intensity

    image[0:vertical_border_size,:,:] = border_color_intensity
    image[image.shape[0]-vertical_border_size:,:,:] = border_color_intensity

def detect_darken_blacks(image):
    #pil_img = Image.fromarray(image)
    #hsv_image = img_as_uint(rgb2hsv(image))
    hsv_image = rgb2hsv(image)
    #np.copyto(image, hsv_image)
    image[hsv_image[:,:,2] < 0.5] = text_color

    condition_indices = np.logical_and(np.logical_and(hsv_image[:,:,0] > (220/360), hsv_image[:,:,0] < (250/360)), hsv_image[:,:,1] > 0.3 )

    image[condition_indices] = annotation_color

    #np.where(np.logical_and(a>=6, a<=10))

    """hsv_image = np.array(pil_img.convert('HSV'))
    hsv_image[hsv_image[:,:,1] < 100] = 0
    pil_img = Image.fromarray(hsv_image)
    pil_img.convert('rgb')
    return pil_img"""

def write_processed_image(image):
    pil_img = Image.fromarray(image)
    pil_img.save('./test.png')

def process_image(img_color_array, index):
    adjust_contrast(img_color_array)
    threshold_white(img_color_array, white_threshold)
    clear_margins(img_color_array, index)
    detect_darken_blacks(img_color_array)

def compress_image(image):
    #with BytesIO() as output_buffer:
    output_buffer = BytesIO()
    #pil_img = Image.fromarray(image)
    image.save(output_buffer, format="JPEG",quality=jpg_quality, optimize=True)

    output_buffer.seek(0)
    jpg_image = Image.open(output_buffer)
    return jpg_image

def save_pdf(path, image_list):
    #with open(path, 'w+') as out_file:
    #out_file.seek(0)
    print("Saving " + str(len(image_list)) + " images to pdf " + path)
    image_list[0].save(path, "PDF", resolution=100.0, save_all=True, append_images=image_list[1:])


def process_images_of_dir_pipeline(src_directory, target_directory):
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)

    png_file_list = listdir(src_directory)
    png_file_list.sort()


    processed_image_list = []

    for index, file in enumerate(png_file_list):

        if(index >= debug_offset and ((index - debug_offset) <= debug_limit or debug_limit == -1)):
            print(file)
            print(index)

            img_path = os.path.join(src_directory, file)

            img = Image.open(img_path)

            if(rescale_images_enabled):
                img = img.resize((int(img.size[0] * resize_factor), int(img.size[1] * resize_factor)), Image.BICUBIC)
            img_color_array = np.asarray(img).copy()

            if(process_images_enabled):
                process_image(img_color_array, index)

            pil_img = Image.fromarray(img_color_array)
            
            if(compress_images_enabled):
                compressed_image = compress_image(pil_img)

            if save_processed_images_enabled:
                out_img_path = os.path.join(target_directory, file)
                compressed_image.save(out_img_path)

            if(save_original_to_pdf_enabled):
                processed_image_list.append(img)

            processed_image_list.append(compressed_image)

    if save_pdf_enabled and len(processed_image_list) > 0:
        save_pdf(out_pdf_path, processed_image_list)


process_images_of_dir_pipeline(src_directory, target_directory)

"""
exit()




#img = misc.imread('./png/diamante_001.png')

img = Image.open('./png/diamante_016.png')

img_gray_array = np.array(img.convert('L'))
img_color_array = np.asarray(img).copy()

thresh_mask = img_gray_array < 200

masked_color_image = img_color_array

#thresh_mask = np.bitwise_not(thresh_mask)

#masked_color_image = img_color_array * thresh_mask[:,:,None]

masked_color_image = img_color_array

#masked_color_image[:,:,0] = img_color_array[:,:,0] * thresh_mask
#masked_color_image[:,:,1] = img_color_array[:,:,1] * thresh_mask
#masked_color_image[:,:,2] = img_color_array[:,:,2] * thresh_mask


#masked_color_image[masked_color_image <= 0] = 255


min_intensity = np.min(img_gray_array)
max_intensity = np.max(img_gray_array)
p2, p98 = np.percentile(masked_color_image, (1, 99))
masked_color_image = exposure.rescale_intensity(masked_color_image, in_range=(p2, p98))


masked_color_image[img_gray_array > 180] = 255
#masked_color_image[img_gray_array < 50] = 0


masked_color_image[:,masked_color_image.shape[1]-200:,:] = 255
masked_color_image[:,0:200,:] = 255

masked_color_image[0:50,:,:] = 255
masked_color_image[masked_color_image.shape[0]-50:,:,:] = 255




#masked_color_image = exposure.equalize_hist(Image.fromarray(masked_color_image))




print("shape is ", masked_color_image.shape)
#print("dtype is ", img_array.dtype)
#print("ndim is ", img_array.ndim)


pil_img = Image.fromarray(masked_color_image)

pil_img.save('./test.png')
"""