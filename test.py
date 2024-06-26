import base64
from PIL import Image
import io

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

# Encode an image to base64
image_path = "tom.jpg"
base64_string = encode_image_to_base64(image_path)
print("Base64 encoded string:")
print(base64_string[:50] + "...") # Print first 50 characters

# Decode base64 string back to image
decoded_image = decode_base64_to_image(base64_string)
print(f"Decoded image size: {decoded_image.size}")

# Optionally, save the decoded image
decoded_image.save("decoded_image.jpg")