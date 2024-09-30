#pip install fal-client
export FAL_KEY="PASTE_YOUR_FAL_KEY_HERE"
import fal_client
 
handler = fal_client.submit(
    "fal-ai/lora",
    arguments={
        "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        # Replace the name of your fine-tuned model in place model name.
        # To update the model name:
        # 1. Go to Fal.ai's official website: https://fal.ai/
        # 2. Log in using your authenticated Git credentials.
        # 3. Click on the "Home" icon.
        # 4. You will see your fine-tuned model listed.
        # 5. Click on "Playground" to access the model.
        # 6. Copy the model name and update it in the script.
        "prompt": "photo of a rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography, Elke vogelsang"
    },
)
 
result = handler.get()
print(result)
