import os
import base64
import json

def encode_images_and_prompts_to_base64(folder_path):
    """
    Encodes all image files (PNG, JPG, JPEG, GIF, BMP) and their corresponding text prompts 
    from the specified folder. Stores both the prompt and Base64 encoded image URL in JSON format.

    Args:
        folder_path (str): The path to the folder containing image and text files.

    Returns:
        str: A JSON string containing a list of dictionaries with prompts and Base64 encoded URLs.
    """
    base64_images_with_prompts = []

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check for image files based on their extensions
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Extract the base name (without extension) to match the corresponding text file
            base_name = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)

            # Construct the corresponding text file name
            text_filename = base_name + '.txt'
            text_file_path = os.path.join(folder_path, text_filename)

            # Read the image file and encode it to Base64
            with open(file_path, "rb") as image_file:
                img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

                # Determine the image format for the data URI
                file_extension = filename.split('.')[-1]
                data_uri = f"data:image/{file_extension};base64,{img_base64}"

            # Read the corresponding text prompt (if it exists)
            prompt = None
            if os.path.exists(text_file_path):
                with open(text_file_path, 'r') as text_file:
                    prompt = text_file.read().strip()
            else:
                prompt = "No prompt available"  # Default message if no text file is found

            # Append the image URL and prompt to the list in JSON format
            base64_images_with_prompts.append({
                "prompt": prompt,
                "image_url": data_uri
            })

    # Return the list of prompts and Base64 encoded image URLs as a JSON string
    return json.dumps(base64_images_with_prompts, indent=4)


folder_path = ''  # Replace with your folder path
result_json = encode_images_and_prompts_to_base64(folder_path)
print(result_json)


#### Sample output
# [
#     {
#         "prompt": "an island near the sea, with seagulls, moon shining over the sea, lighthouse, boats in the background, fish flying over the sea",
#         "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
#     },
#     {
#         "prompt": "a beautiful sunset over the mountains, with birds flying and river flowing between the mountains",
#         "image_url": "data:image/png;base64,YOUR_BASE64_STRING_2_HERE"
#     },
#     {
#         "prompt": "forest landscape with a clear blue lake, trees, and mountain range in the background",
#         "image_url": "data:image/png;base64,YOUR_BASE64_STRING_3_HERE"
#     }
# ]