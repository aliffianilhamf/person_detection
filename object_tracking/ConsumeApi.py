import requests
import cv2
from PIL import Image
import numpy as np
from io import BytesIO


def PostImageAndData(image, nim, name, track_id):
    # Your API endpoint
    url = 'https://api-indoor-duration.natagw.my.id/person-duration'

    # Bearer token for authorization
    bearer_token = 'gQNoGepgU1DKK3GMlnKZ6MKsXOBximpu'

    # Headers with the Authorization token
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {bearer_token}',  # Interpolating the token
    }

    # Convert numpy array to image and save it to a BytesIO object
    image_data = Image.fromarray(image)
    buffered = BytesIO()
    image_data.save(buffered, format="JPEG")
    buffered.seek(0)

    # Prepare the files and data payload
    files = {
        'file': ('image.jpg', buffered, 'image/jpeg')  # Set file name and MIME type
    }
    data = {
        'nim': nim,
        'name': name,
        'name_track_id': name + str(track_id)
    }

    # Make the POST request
    response = requests.post(url, headers=headers, files=files, data=data)

    # Check the response
    if response.status_code == 200:
        print("Request successful!")
        print("Response:", response.json())  # Adjust as needed
    else:
        print("Request failed with status code:", response.status_code)
        print("Error message:", response.text)