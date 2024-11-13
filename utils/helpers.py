from fastai.vision.all import *
import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from natsort import natsorted
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def get_im_txt_pths(dataset_dir, im_extensions=('.jpg', '.png', '.tiff')):
    """
    Retrieves image and corresponding YOLO annotation file paths.

    Parameters:
    -----------
    dataset_dir : Path
        Dataset directory where full-size PID images and their annotations in YOLO format are present.
    im_extensions : tuple, optional
        Allowed image file extensions (default: ('.jpg', '.png', '.tiff')).

    Returns:
    --------
    im_pths : list
        List of image file paths sorted naturally.
    txt_pths : list
        List of annotation file paths sorted naturally.
    """
    # Ensure directory exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")
    
    # Get image and annotation paths
    im_pths = natsorted(get_files(dataset_dir, extensions=im_extensions))
    txt_pths = natsorted(get_files(dataset_dir, extensions='.txt'))
    
    if len(im_pths) == 0:
        raise ValueError(f"No images found in {im_dir} with extensions {im_extensions}")
    
    if len(txt_pths) == 0:
        raise ValueError(f"No annotation files found in {txt_dir}")
    
    # Ensure the number of images matches the number of annotations
    if len(im_pths) != len(txt_pths):
        raise ValueError(f"Mismatch: {len(im_pths)} images and {len(txt_pths)} annotations found.")
    
    return im_pths, txt_pths

def get_im_pths(dataset_dir, im_extensions=('.jpg', '.png', '.tiff')):
    """
    Retrieves image and corresponding YOLO annotation file paths.

    Parameters:
    -----------
    dataset_dir : Path
        Dataset directory where full-size PID images and their annotations in YOLO format are present.
    im_extensions : tuple, optional
        Allowed image file extensions (default: ('.jpg', '.png', '.tiff')).

    Returns:
    --------
    im_pths : list
        List of image file paths sorted naturally.
    txt_pths : list
        List of annotation file paths sorted naturally.
    """
    # Ensure directory exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")
    
    # Get image and annotation paths
    im_pths = natsorted(get_files(dataset_dir, extensions=im_extensions))
    
    if len(im_pths) == 0:
        raise ValueError(f"No images found in {dataset_dir} with extensions {im_extensions}")
    
    return im_pths

def copy_files_to_directory(file_paths, dest_dir):
    """
    Copy a list of files to a destination directory, creating the directory if it doesn't exist.

    Parameters:
    -----------
    file_paths : list
        List of file paths to be copied.
    dest_dir : str or Path
        Destination directory to copy the files to.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    print(f"Copying {len(file_paths)} files to {dest_dir}...")
    for file_path in map(Path, file_paths):
        dest_file = dest_dir / file_path.name
        shutil.copy(file_path, dest_file)
    
    print(f"Successfully copied {len(file_paths)} files to {dest_dir}")
    

def save_pdf_page_as_image_from_dir(
    directory, 
    page_to_render=0, 
    dpi=300, 
    image_format="jpg"
):
    """
    Renders a specified page from each PDF document in the directory and saves it as an image file.
    
    Parameters:
    - directory (str): Path to the directory containing PDF documents.
    - page_to_render (int): The page number to render and save (0-indexed).
    - dpi (int): The DPI for rendering the page.
    - image_format (str): The format to save the image in, e.g., 'jpg' or 'png'.
    """
    # Get a list of all PDF files in the directory
    pdf_documents = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
    img_paths = []
    for i, pdf_document in enumerate(pdf_documents):
        try:
            # Open the PDF document
            pdf = fitz.open(pdf_document)

            # Check if the requested page number exists in the document
            if page_to_render < len(pdf):
                # Get the specified page
                page = pdf[page_to_render]

                # Render the page at the specified DPI
                matrix = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=matrix)

                # Define the path to save the image in the same directory
                image_name = f"{os.path.splitext(os.path.basename(pdf_document))[0]}_page_{page_to_render}.{image_format}"
                image_path = os.path.join(directory, image_name)
                img_paths.append(image_path)
                # Save the image with high quality
                pix.save(image_path)
                print(f"Saved {image_path}")

            else:
                print(f"Warning: Page {page_to_render} does not exist in {pdf_document}")

            # Close the PDF document
            pdf.close()
        
        except Exception as e:
            print(f"Error processing {pdf_document}: {e}")
    
    return img_paths

def draw_ocr_boxes(image_path, ocr_result):
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for line in ocr_result.text_lines:
        bbox = line.bbox
        text = line.text.strip()
        confidence = line.confidence

        # Draw rectangle for the bounding box
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=2)

        # Display the text and confidence level
        display_text = f"{text} ({confidence:.2f})"
        draw.text((bbox[0], bbox[1] - 10), display_text, fill="red")

    # Generate the new path in the parent directory's `ocr_results` folder
    base_name = os.path.basename(image_path)
    new_filename = f"{os.path.splitext(base_name)[0]}_text_ocr{os.path.splitext(base_name)[1]}"
    
    # Define the directory one level up from the current image directory
    parent_dir = os.path.dirname(os.path.dirname(image_path))
    ocr_results_dir = os.path.join(parent_dir, "ocr_results")

    # Ensure the `ocr_results` folder exists
    os.makedirs(ocr_results_dir, exist_ok=True)

    # Full path for the new image
    new_image_path = os.path.join(ocr_results_dir, new_filename)

    # Save the modified image with the same quality
    image.save(new_image_path, quality=95)
    print(f"Saved annotated image at {new_image_path}")
