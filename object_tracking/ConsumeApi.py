import requests
import cv2
from PIL import Image
import numpy as np
from io import BytesIO

bearer_token = '5dddc644-2cfa-40f3-aa22-4769ddae6e90'


def PostPersonDuration(name):
    # Your API endpoint
    url = 'https://api-indoor-duration.natagw.my.id/person-durations'

    # Headers with the Authorization token
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {bearer_token}',  # Interpolating the token
    }

    # Prepare the data payload
    data = {
        'name': name
    }

    # Make the POST request
    response = requests.post(url, json=data, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Request successful!")
        print("Response:", response.json())  # Adjust as needed
    else:
        print("Request failed with status code:", response.status_code)
        print("Error message:", response.text)


def PostDetailPersonDuration(image, nim, name, track_id):
    # Your API endpoint
    url = 'https://api-indoor-duration.natagw.my.id/person-durations/detail'

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
        'image_file': ('image.jpg', buffered, 'image/jpeg')  # Set file name and MIME type
    }
    data = {
        'nim': nim,
        'name': name,
        'name_track_id': name + str(track_id)
    }

    # Make the POST request
    response = requests.post(url, headers=headers, data=data, files=files)

    # Check the response
    if response.status_code == 200:
        print("Request successful!")
        print("Response:", response.json())  # Adjust as needed
    else:
        print("Request failed with status code:", response.status_code)
        print("Error message:", response.text)


def UpdateEndTimePersonDuration(track_id, end_time):
    # Your API endpoint
    url = f"https://api-indoor-duration.natagw.my.id/person-durations/detail/{track_id}"

    # Headers with the Authorization token
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {bearer_token}',  # Interpolating the token
    }

    # Prepare the data payload
    data = {
        'end_time': end_time
    }

    # Make the Patch request
    response = requests.patch(url, json=data, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Request successful!")
        print("Response:", response.json())  # Adjust as needed
    else:
        print("Request failed with status code:", response.status_code)
        print("Error message:", response.text)