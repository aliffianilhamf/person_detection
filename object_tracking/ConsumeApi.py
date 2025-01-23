import pytz
import requests
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import datetime 

bearer_token = '5be82c8b-3c5d-4de8-ba09-22e5aa4ae97c'


def PostPersonDuration(name, track_id):
    # Your API endpoint
    # url = 'httpss://api-indoor-duration.natagw.my.id/person-durations'
    url = 'https://api-indoor-duration.natagw.my.id/person-durations'

    # Headers with the Authorization token
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {bearer_token}',  # Interpolating the token
    }

    # Prepare the data payload
    data = {
        'name': name + str(track_id)
    }
    
    print(data)

    # Make the POST request
    response = requests.post(url, json=data, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Request successful!")
        print("Response:", response.json())  # Adjust as needed
    else:
        print("Request failed with status code:", response.status_code)
        print("Error message PostPersonDuration:", response.text)


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
    print(data)

    # Make the POST request
    response = requests.post(url, headers=headers, data=data, files=files)

    # Check the response
    if response.status_code == 200:
        print("Request successful!")
        print("Response:", response.json())  # Adjust as needed
    else:
        print("Request failed with status code:", response.status_code)
        print("Error message PostDetailPersonDuration:", response.text)


def UpdateEndTimePersonDuration(track_id, end_time):
    # Your API endpoint
    url = f"https://api-indoor-duration.natagw.my.id/person-durations/detail/{track_id}"

    # Headers with the Authorization token
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {bearer_token}',  # Interpolating the token
    }

    indo_time= datetime.datetime.now(pytz.timezone('Asia/Jakarta'))
    formatted_time = indo_time.strftime('%Y-%m-%dT%H:%M:%S')
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
        print("Error message UpdateEndTimePersonDuration:", response.text)