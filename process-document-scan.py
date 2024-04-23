 
#from scipy import misc

from io import BytesIO
from PIL import Image
import numpy as np

import skimage
from skimage import data
from skimage import exposure
from skimage.color import rgb2gray, convert_colorspace, rgb2hsv
from skimage.util import img_as_uint
from scipy.ndimage import shift, median_filter, gaussian_filter
from skimage.filters import unsharp_mask, butterworth
from skimage.morphology import disk
from skimage.filters import rank

from os import listdir
import os
from concurrent.futures import ThreadPoolExecutor

def adjust_contrast(image):
    p2, p98 = np.percentile(image, (1, 99))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))

def threshold_white(image, white_threshold_val, target_white_intensity_value):
    print("Threshold clearing white image regions")
    #pil_img = Image.fromarray(image)
    #pil_img = Image.fromarray(image)
    #img_gray_array = np.array(pil_img.convert('L'))
    img_gray_array = rgb2gray(image)
    image[img_gray_array > (white_threshold_val/255)] = target_white_intensity_value

def clear_margins(image, options, index=-1):
    print("Clearing image margins")

    #image = np.roll(image, (100,100), axis=(1,0))

    if index % 2 != 0:
        image[:,image.shape[1]-options.binding_border_size:,:] = options.clear_color
        image[:,0:options.non_binding_border_size,:] = options.clear_color

        if(options.shift):
            #image = shift(image,[options.placement_adjustment_shift_y,options.placement_adjustment_shift_x,0], mode='constant', cval=255)
            shifted = shift(image,[options.placement_adjustment_shift_y,options.placement_adjustment_shift_x,0], mode='constant', cval=255)
            np.copyto(image, shifted)
        
    else:
        image[:,image.shape[1]-options.non_binding_border_size:,:] = options.clear_color
        image[:,0:options.binding_border_size,:] = options.clear_color

    image[0:options.vertical_border_size,:,:] = options.clear_color
    image[image.shape[0]-options.vertical_border_size:,:,:] = options.clear_color

def detect_recolor_image_layers(image, text_color=(0,0,0), annotation_color=(40,0,220), background_color=(255,255,255)):
    print("Detect and darken/recolor text and annotation layers")
    #pil_img = Image.fromarray(image)
    #hsv_image = img_as_uint(rgb2hsv(image))

    hsv_image = rgb2hsv(image)

    print("Detecting text and setting new color")

    black_text_condition= np.logical_and(hsv_image[:,:,1] < 0.6, hsv_image[:,:,2] < 0.55)

    print("Detecting annotations and setting new color")
    condition_indices = np.logical_and(np.logical_and(np.logical_and(hsv_image[:,:,0] > (210/360), hsv_image[:,:,0] < (250/360)), hsv_image[:,:,1] > 0.3 ), hsv_image[:,:,2] > 0.55)
    black_text_condition = np.logical_and(np.logical_not(condition_indices), black_text_condition)

    white_bg_condition = np.logical_and(hsv_image[:,:,1] < 0.3, hsv_image[:,:,2] > 0.74)

    image[black_text_condition] = text_color
    image[condition_indices] = annotation_color
    image[white_bg_condition] = (255,255,255)


    """hsv_image = np.array(pil_img.convert('HSV'))
    hsv_image[hsv_image[:,:,1] < 100] = 0
    pil_img = Image.fromarray(hsv_image)
    pil_img.convert('rgb')
    return pil_img"""

def filter_adjust_image(image):

    print("Filtering")
    print("Median")
    image = median_filter(image, size=4)

    #image = butterworth(image, cutoff_frequency_ratio=0.02, order=3, high_pass=False)

    print("Gaussian")
    image = gaussian_filter(image, sigma=2)

    print("Unsharp")
    image = unsharp_mask(image, radius=6, amount=2)

def write_processed_image(np_image_array, path):
    pil_img = Image.fromarray(np_image_array)
    pil_img.save(path)

def process_image(img_color_array, options, index=-1):
    print("Running image adjustments and processing for " + str(index))

    adjust_contrast(img_color_array)
    clear_margins(img_color_array, options, index)
    detect_recolor_image_layers(img_color_array, options.text_color, options.annotation_color, options.clear_color)
    filter_adjust_image(img_color_array)

def rescale_image(img, rescale_factor):
    if(rescale_factor == 1.0):
        return img
        
    return img.resize((int(img.size[0] * rescale_factor), int(img.size[1] * rescale_factor)), Image.BICUBIC)

def compress_image(image, format='png', jpg_quality=80):
    output_buffer = BytesIO()

    if('jpg' in format.lower() or 'jpeg' in format.lower()):
        format = "JPEG"
        image.save(output_buffer, format, quality=jpg_quality, optimize=True)
    else:
        format = "PNG"
        image.save(output_buffer, format)

    output_buffer.seek(0)
    compressed_pil_image = Image.open(output_buffer)
    return compressed_pil_image



def read_process_compress_image(image_path, options, index=-1):
    original_image = Image.open(image_path)
    rescaled_image = rescale_image(original_image, options.rescale_factor)
    img_color_array = np.asarray(rescaled_image)

    if(not options.disable_processing):
        process_image(img_color_array, options, index)

    processed_pil_image = Image.fromarray(img_color_array)

    if(options.compress):
        processed_pil_image = compress_image(processed_pil_image, options.output_format, options.jpg_quality)

    return rescaled_image, processed_pil_image

import re
image_file_regex = re.compile(".+\.(png|PNG|jpg|jpeg|JPG|JPEG|tiff|TIFF|gif|GIF|bmp|BMP)$")
def is_image_file(image_file_name):
    return bool(image_file_regex.match(image_file_name))

def in_processing_range(index, images_offset, images_amount):
    #print('index: ' + str(index) + " -- offset " + str(images_offset) + " -- amount " + str(images_amount))

    if(images_offset == -1 and images_amount == -1):
        return True
    
    if(images_offset == -1):
        images_offset = 0

    if(images_amount == -1):
        images_amount = index
    
    images_index_over_offset = (index - images_offset)
    
    if(index >= images_offset and images_index_over_offset < images_amount):
        return True
    
    return False


def read_process_and_write_image(img_path, options, index, images_total_length):
    print("read_process_and_write_image: " + img_path)

    rescaled_image, processed_image = read_process_compress_image(img_path, options, index)
    
    if(options.processed_dir != None and not options.save_images_after_finish_processing):
        save_image_to_dir(processed_image, get_image_id(index,images_total_length),options.processed_dir, options.output_format, options.output_image_name)
        processed_image.close()

    if(options.processed_dir != None and options.save_original):
        save_image_to_dir(rescaled_image, get_image_id(index,images_total_length) + "_original",options.processed_dir, options.output_format, options.output_image_name)
        rescaled_image.close()

def get_sorted_dir_image_paths(directory):
    image_file_list = listdir(directory)

    image_file_list = list(filter(lambda image_file_name: is_image_file(image_file_name), image_file_list))
    image_file_list.sort()

    image_paths_list = list(map(lambda image_file_name: os.path.join(directory, image_file_name), image_file_list))
    return image_paths_list

def process_images_of_dir_pipeline(src_directory, options):
    image_paths_list = get_sorted_dir_image_paths(src_directory)
    images_total_length = len(image_paths_list)

    with ThreadPoolExecutor(max_workers=options.threads) as pool:

        for index, img_path in enumerate(image_paths_list):

            if(index < images_total_length and in_processing_range(index, options.images_offset, options.images_amount)):
                print(img_path)
                print(index)

                future_worker = pool.submit(read_process_and_write_image, img_path, options, index, images_total_length)
                #rescaled_image, processed_image = read_process_compress_image(img_path, options, index)


def save_image_to_dir(image, image_unique_id, target_directory, format='png', image_name="out"):
    print("Saving image " + image_unique_id + " to directory: " + target_directory)
    
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)

    if(format.startswith('.')):
        format = format[1:]

    out_img_path = os.path.join(target_directory, image_name + '_' + image_unique_id + '.' + format)
    image.save(out_img_path)
    
def get_image_id(index, images_total_length):
    required_digits = len(str(images_total_length))
    normalized_index_string = str(index).zfill(required_digits)
    return normalized_index_string

def save_images_to_dir(image_list, target_directory, format='png', image_name="out", name_offset=0):
    print("Saving " + str(len(image_list)) + " to directory: " + target_directory)
    
    if(format.startswith('.')):
        format = format[1:]

    images_length = len(image_list)
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)

    for index, image in enumerate(image_list):

        normalized_index_string = get_image_id(index + name_offset, images_length)
        save_image_to_dir(image, normalized_index_string, target_directory, format, image_name)

def save_images_to_pdf(image_list, pdf_path, options_suffix_string="", create_pdf_dir=False, batch_size=20):
    target_directory = os.path.dirname(pdf_path)

    if(not os.path.exists(target_directory)):

        if(create_pdf_dir):
            os.mkdir(target_directory)
        else:
            raise Exception("Can not save pdf to" + pdf_path + " directory: " + target_directory + " does not exist")
    
    if(len(image_list) <= 0):
        raise Exception("image_list needs to be non empty to save pdf")

    pdf_path_parts = str(pdf_path).split('.')
    extension = pdf_path_parts.pop()

    if(len(options_suffix_string) > 0):
        options_suffix_string += '--xim-' + str(len(image_list))

    pdf_path = '.'.join(pdf_path_parts) + options_suffix_string + '.' + extension
    
    print("Saving " + str(len(image_list)) + " images to pdf " + pdf_path)
    #image_list[0].save(pdf_path, "PDF", save_all=True, append_images=image_list[1:])
    
    print('Saving image ' + str(0) + " to pdf at: " + pdf_path)
    image_list[0].save(pdf_path, "PDF")

    images_to_append_list = image_list[1:]
    images_to_append_count = len(images_to_append_list)
    nr_batches =  int(round(images_to_append_count/batch_size) + 1)

    rounded_images_to_append_count = int(nr_batches * batch_size)

    for index in range(0, rounded_images_to_append_count, batch_size):

        batch_index= int(index/batch_size)
        #print("Writing batch " + str(batch_index))

        end_batch_index = index + batch_size
        if(end_batch_index > images_to_append_count):
            end_batch_index = images_to_append_count

        images_batch = images_to_append_list[index:end_batch_index]

        images_append_batch = []
        if(len(images_batch) > 1):
            images_append_batch = images_batch[1:]

        if(len(images_batch) > 0):

            print('Saving image batch ' + str(batch_index) + ": from " +  str(index) + " to " + str(end_batch_index) + " -- to pdf at: " + pdf_path)

            images_batch[0].save(pdf_path, "PDF", save_all=True, append=True, append_images=images_append_batch)

        for image in images_batch:
            image.close()

def load_rescale_compress(image_path, options, index=-1):
    image = rescale_image(Image.open(image_path), options.rescale_factor)
    if(options.compress):
        image = compress_image(image, options.output_format, options.jpg_quality)

    print('Loaded image ' + str(index) + ": " + image_path)
    return image

def save_images_of_dir_to_pdf(src_directory, pdf_path, options):
    image_paths_list = get_sorted_dir_image_paths(src_directory)

    if(options.images_offset < 0):
        options.images_offset = 0

    if(options.images_amount > 0):
        image_paths_list = image_paths_list[options.images_offset:options.images_offset + options.images_amount]

    
    images = []

    with ThreadPoolExecutor(max_workers=options.threads) as pool:
        
        worker_futures = []
        for index, image_path in enumerate(image_paths_list):
            
            worker_future = pool.submit(load_rescale_compress, image_path, options, index)
            worker_futures.append(worker_future)

            #load_rescale_compress(image_path, images, options, index)
            #images.append(load_rescale_compress(image_path, options, index))

        for future in worker_futures:
            worker_result = future.result()
            images.append(worker_result)
            
    options_suffix_string = ""
    if(options.add_parameters):
        options_suffix_string='_' + path_string_from_options(options)

    save_images_to_pdf(images, pdf_path, options_suffix_string, create_pdf_dir=options.create_pdf_dir, batch_size=options.batch_size)

def save_images_to_pdf_options(src_directory, pdf_path, options):

    options.add_parameters = True

    pdf_option_sets = [
        {
            'rescale_factor': 1.0,
            'output_format': 'png'
        },
        {
            'rescale_factor': 0.8,
            'output_format': 'png'
        },
        {
            'rescale_factor': 0.6,
            'output_format': 'png'
        },
        {
            'rescale_factor': 0.5,
            'output_format': 'png'
        },
        {
            'rescale_factor': 0.4,
            'output_format': 'png'
        },
        {
            'rescale_factor': 0.3,
            'output_format': 'png'
        },
        {
            'rescale_factor': 1.0,
            'output_format': 'jpg',
            'jpg_quality': 70
        },
        {
            'rescale_factor': 0.7,
            'output_format': 'jpg',
            'jpg_quality': 65
        },
        {
            'rescale_factor': 0.5,
            'output_format': 'jpg',
            'jpg_quality': 65
        },
        {
            'rescale_factor': 0.6,
            'output_format': 'jpg',
            'jpg_quality': 30
        },
    ]

    with ThreadPoolExecutor(max_workers=options.threads) as pool:

        for pdf_option_set in pdf_option_sets:
            print(pdf_option_set)

            new_options = dotdict(options)
            for key in pdf_option_set:
                new_options[key] = pdf_option_set[key]

            future_worker = pool.submit(save_images_of_dir_to_pdf, src_directory, pdf_path, new_options)

def path_string_from_options(options):
    
    selected_options = [
        's-' + str(options.rescale_factor),
        'sft-' + str(int(options.shift)),
        'c-' + str(int(options.compress)),
        'f-' + options.output_format
    ]

    if(not 'png' in options.output_format.lower()):
        selected_options.append('q-' + str(options.jpg_quality))

    return '--'.join(selected_options)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def interleave_arrays(arrays):
    interleaved_array = []

    array_lengths = [ len(array) for array  in arrays ]

    max_array_length = max(array_lengths)

    for index in range(0, max_array_length):

        for array in arrays:
            if(index < len(array)):
                interleaved_array.append(array[index])

    return interleave_arrays



def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Template for building python console packages"
    )

    parser.add_argument('source_dir', help="Source directory to read the images to process and convert from")
    parser.add_argument('target', help="Where to store the result(s) to. Either a path to a .pdf file or a directory where to store the processed images to")

    parser.add_argument('-pdir','--processed_dir', help="Path to the directory where to store the processed images to")
    parser.add_argument('-pdf','--pdf_path', help="The path to the target pdf to be created")
    
    parser.add_argument('-mkdir','--create_pdf_dir', action="store_true", help="Create a new directory if the specified directory of the pdf_path does not exist", default=False)
    parser.add_argument('-imgname','--output_image_name', help="How to label the processed images when they are written to target dir", default='out')

    parser.add_argument('-sorg','--save_original', action="store_true", help="Save the original image to the target pdf for comparison", default=False)
    parser.add_argument('-ap','--add_parameters', action="store_true", help="Add parameters to pdf name", default=False)

    parser.add_argument('-wt','--white_threshold', type=int, help="Threshold pixel intensity (grayscale lightness) value at which a pixel is considered 'white'", default=180)
    parser.add_argument('-bbs','--binding_border_size', type=int, help="Size of the border to be cleared for clearing the area of the binding from the scan", default=200)
    parser.add_argument('-shift','--shift', action="store_true", help="Shift even scans to adjust difference", default=False)

    parser.add_argument('-rs','--rescale_factor', type=float, help="Rescale the image before processing and saving", default=1.0)
    
    parser.add_argument('-o','--images_offset', type=int, help="Start from the image files at the offset position", default=-1)
    parser.add_argument('-n','--images_amount', type=int, help="Amount of images to process", default=-1)
    
    parser.add_argument('-c','--compress', action="store_true", help="Compress the processed images before saving", default=False)
    parser.add_argument('-f','--output_format', help="Image Output format", choices=['png', 'jpg'], default='png')
    parser.add_argument('-q','--jpg_quality', type=int, help="Compressing quality if jpg format is selected .. extreme compression [0,100] no compression", default=80)
    parser.add_argument('-t','--threads', type=int, help="Amount of worker threads for image processing", default=1)
    parser.add_argument('-bn','--batch_size', type=int, help="Batch size for saving images to pdf file", default=20)
    
    parser.add_argument('-sv','--save_images_after_finish_processing', action="store_true", help="Save Images as soon as all of them are processed", default=False)

    parser.add_argument('-dp','--disable_processing', action="store_true", help="Disable image processing", default=False)
    parser.add_argument('-sm','--save_pdf_matrix', action="store_true", help="Save multiple pdfs with various compression options", default=False)

    parser.add_argument('-tc','--text_color', type=int, nargs='+', help="Color to set detected text pixels in the image as", default=[0,0,0])
    parser.add_argument('-ac','--annotation_color', type=int, nargs='+', help="Color to set detected text annotation pixels in the image as", default=[40,0,220])
    parser.add_argument('-cc','--clear_color', type=int, nargs='+', help="Color to set cleared areas and thresholded background to", default=[255,255,255])

    arguments = parser.parse_args()

    options = {
        'binding_border_size' : int(200 * arguments.rescale_factor),
        'non_binding_border_size' : int(10 * arguments.rescale_factor),
        'vertical_border_size' : int(50 * arguments.rescale_factor),
        'placement_adjustment_shift_x' : int(25 * arguments.rescale_factor),
        'placement_adjustment_shift_y' : int(5 * arguments.rescale_factor),
        'text_color': tuple(arguments.text_color),
        'annotation_color': tuple(arguments.annotation_color),
        'clear_color': tuple(arguments.clear_color),
    }

    if(arguments.target.endswith('.pdf')):
        print("Target is a pdf")
        options['pdf_path'] = arguments.target

    else:
        if(not os.path.isdir(arguments.target)):
           os.mkdir(arguments.target)

        print("Target is a directory")
        options['processed_dir'] = arguments.target
        
    options = {**arguments.__dict__, **options}
    #options.update(arguments.__dict__)
    options = dotdict(options)

    #By default processing and pdf creation should be either done in tandem or as 2 step process (processing+saving, loading and creating pdf)
    #Therefore processing should be disabled by default when only creating a pdf
    if(options.processed_dir == None and options.pdf_path != None):
        options.disable_processing=True

    if(options.processed_dir != None):
        process_images_of_dir_pipeline(arguments.source_dir, options)

    #print(options)
    #if(options.processed_dir != None and options.save_images_after_finish_processing):
    #    save_images_to_dir(processed_image_list, options.processed_dir, format=options.output_format, name_offset=(options.images_offset+1), image_name=options.output_image_name)

    if(options.pdf_path != None):

        source_path = None
        if(options.processed_dir != None and len(options.processed_dir) > 0 and os.path.isdir(options.processed_dir)):
            source_path = options.processed_dir
        elif(options.source_dir != None and len(options.source_dir) > 0 and os.path.isdir(options.source_dir)):
            source_path = options.source_dir

        if(not options.save_pdf_matrix):
            save_images_of_dir_to_pdf(source_path, options.pdf_path, options)
        else:
            save_images_to_pdf_options(options.source_dir, options.pdf_path, options)
main()