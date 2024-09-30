# Install the fal-client first
!pip install fal-client

import fal
import json

# Configure FAL API credentials
fal.config({
    'credentials': 'PASTE_YOUR_FAL_KEY_HERE',  # Insert your FAL API key here
})

# Establish connection to the FAL API
connection = fal.realtime.connect("fal-ai/fast-lcm-diffusion")

# Function to handle results
def handle_result(result):
    print("Result received:", result)

# Function to handle errors
def handle_error(error):
    print("Error encountered:", error)

# Assign the result and error handlers
connection.on_result = handle_result
connection.on_error = handle_error

# Load the JSON file containing the image URLs and prompts
with open('image_prompts.json', 'r') as f:
    image_prompts = json.load(f)

# Iterate through the JSON data and send each image and prompt to FAL.ai
for idx, image_data in enumerate(image_prompts):
    print(f"Sending image {idx+1}...")
    connection.send({
        'prompt': image_data['prompt'],
        'sync_mode': True,  # Sync mode is enabled for real-time result handling
        'image_url': image_data['image_url'],  # Base64-encoded image URL
    })

print("All prompts have been sent.")


# in above we are reaing the prompt(text description of a image), base64 encoded url stored into JSON where you can read automaticcly and pass one be one
[
    {
        "prompt": "an island near the sea, with seagulls, moon shining over the sea, lighthouse, boats in the background, fish flying over the sea",
        "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
    },
    {
        "prompt": "a beautiful sunset over the mountains, with birds flying and river flowing between the mountains",
        "image_url": "data:image/png;base64,YOUR_BASE64_STRING_2_HERE"
    },
    {
        "prompt": "forest landscape with a clear blue lake, trees, and mountain range in the background",
        "image_url": "data:image/png;base64,YOUR_BASE64_STRING_3_HERE"
    }
]

