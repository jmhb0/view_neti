import math
from typing import List

from PIL import Image, ImageDraw


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image


def get_image_grid_options_control_size(images: List[Image.Image], rows: int,
                                        cols) -> Image:
    num_images = len(images)
    assert num_images == rows * cols
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    ## uncomment the next stuff to transpose the grid
    # grid_image = Image.new('RGB', (rows * height, cols * width))
    # for i, img in enumerate(images):
    #     x = i % cols
    #     y = i // cols
    #     grid_image.paste(img, (y * height, x * width))
    return grid_image


def write_prompts_to_file(all_prompts: List[List[str]], fname: str):
    """ print the prompts used to generate interpolation  grids """
    with open(fname, 'w') as f:
        for i, row in enumerate(all_prompts):
            f.write(f"Row {i}\n")
            for prompt in row:
                f.write(f"\t{prompt}\n")


def downsample_image(image, scale_factor):
    """ E.g, let scale_factor=0.5 """

    # Calculate the new dimensions after downsampling
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)

    # Resize the image using Lanczos resampling for best quality
    downsized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return downsized_image


def make_square(image_path):
    """ add padding evenly to top/bottom or left/right to get this """
    # Open the image using Pillow
    img = Image.open(image_path)

    # Get the width and height of the image
    width, height = img.size

    # Find the longer side
    longer_side = max(width, height)

    # Create a new blank square image of the longer side size
    square_img = Image.new('RGB', (longer_side, longer_side), (255, 255, 255))

    # Calculate the paste coordinates to center the original image
    paste_coords = ((longer_side - width) // 2, (longer_side - height) // 2)

    # Paste the original image onto the center of the blank square image
    square_img.paste(img, paste_coords)

    return square_img


def make_square_control_padding(image_path, size, padding_position='bottom'):
    # Load the image
    image = Image.open(image_path)

    # Resize the image to the specified size
    image = image.resize((size, size))

    # Calculate the padding size required to make the image square
    if padding_position == 'bottom':
        padding = (0, size - image.size[1], 0, 0)
    elif padding_position == 'left':
        padding = (size - image.size[0], 0, 0, 0)
    else:
        raise ValueError("Invalid padding_position. Use 'bottom' or 'left'.")

    # Add padding to the image
    squared_image = ImageOps.expand(image, padding, fill='white')

    return squared_image

def add_colored_padding_top(image, padding=100, color=(0,0,0)):
    # Calculate the new height with padding
    new_height = image.height + padding

    # Create a yellow canvas with the new size
    yellow_canvas = Image.new("RGB", (image.width, new_height), color=color)

    # Paste the original image on the canvas with padding on the top
    yellow_canvas.paste(image, (0, padding))

    return yellow_canvas

def add_yellow_line(image, line_height=20, line_width=20):
    """ 
    Add yellow line to bottom of a pil image. This is so we can tag plot train 
    and test images together, but mark the train images
    `width` is the line thicknes, `line_height` is how far from the bottom the 
    line begins.
    """
    width, height = image.size
    draw = ImageDraw.Draw(image)
    line_color = (255, 255, 0) # yellow

    # Draw the yellow line
    draw.line([(0, height - line_height), (width, height - line_height)], fill=line_color, width=line_width)

    return image

def stack_images_with_line(image1, image2, line_thickness=3):
    # Get the dimensions of the two images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Calculate the height of the black line between the images
    line_height = line_thickness

    # Calculate the total height required for the stacked images
    total_height = height1 + line_height + height2

    # Create a new blank canvas with black background
    stacked_image = Image.new("RGB", (max(width1, width2), total_height), color=(0, 0, 0))

    # Paste the first image at the top of the canvas
    stacked_image.paste(image1, (0, 0))

    # Draw a black line below the first image
    draw = ImageDraw.Draw(stacked_image)
    draw.rectangle([(0, height1), (max(width1, width2), height1 + line_height)], fill=(0, 0, 0))

    # Paste the second image below the black line
    stacked_image.paste(image2, (0, height1 + line_height))

    return stacked_image
