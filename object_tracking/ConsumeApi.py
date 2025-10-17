import pytz
import requests
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import datetime 
import os
import sys
import shutil

sys.path.append(os.path.abspath('../face_recognition'))
from create_embeddings import extract_embeddings

bearer_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImFkbWluQGdtYWlsLmNvbSIsInJvbGUiOiJhZG1pbiIsImZ1bGxuYW1lIjoiYWRtaW4iLCJleHAiOjE3NTIzMDkyOTZ9.DiAsTLgOi4rRg0r-4zaVETRlEe5uarg8XXVpnCLeyoQ'


def PostPersonDuration(name, track_id):
    # Your API endpoint
    # url = 'httpss://api-indoor-duration.natagw.my.id/person-durations'
    url = 'https://v3-api-indoor-duration.natagw.my.id/api/3/person_duration/create'


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
    response = requests.post(url, data=data, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("Request PostPersonDuration successful!")
        # print("Response:", response.json())  # Adjust as needed
    else:
        print("Request failed with status code:", response.status_code)
        print("Error message PostPersonDuration:", response.text)


def PostDetailPersonDuration(image, nim, name, track_id):
    # Your API endpoint
    url = 'https://v3-api-indoor-duration.natagw.my.id/api/3/person_duration/detail/create'

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
        'image': ('image.jpg', buffered, 'image/jpeg')  # Set file name and MIME type
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
        print("Request PostDetailPersonDuration successful!")
        # print("Response:", response.json())  # Adjust as needed
    else:
        print("Request failed with status code:", response.status_code)
        print("Error message PostDetailPersonDuration:", response.text)


def UpdateEndTimePersonDuration(track_id, end_time):
    # Your API endpoint
    url = f'https://v3-api-indoor-duration.natagw.my.id/api/3/person_duration/detail/{track_id}'

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
        print("Request UpdateEndTimePersonDuration successful!")
        # print("Response:", response.json())  # Adjust as needed
    else:
        print("Request failed with status code:", response.status_code)
        print("Error message UpdateEndTimePersonDuration:", response.text)
        

def GetAllAssistant():
    assistants = []
    url = 'https://v3-api-indoor-duration.natagw.my.id/api/3/admin/assistant'
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {bearer_token}',  # Interpolating the token
    }
    response = requests.get(url, headers=headers)
    # get uuid all assistant when status_embed = false
    # cheeck if response is successful
    # for assistant in response.json()['data']:
    #     if not assistant['status_embed']:
    #         assistants.append({
    #                 'uid': assistant['uid'],
    #                 'image': assistant['image'],
    #                 'name': f"{assistant['fullname'].replace(' ', '_')}__{assistant['nim']}"
    #             })
    if response.status_code == 200:
        for assistant in response.json()['data']:
            if not assistant['status_embed']:
                assistants.append({
                    'uid': assistant['uid'],
                    'image': assistant['image'],
                    'name': f"{assistant['fullname'].replace(' ', '_')}__{assistant['nim']}"
                })
        # print("Request successful!")
        # print("Response:", response.json())  # Adjust as needed
        return assistants
    else:
        print("Request failed with status code:", response.status_code)
        print("Error message GetAllAssistant:", response.text)
        
def GetImage(image_file_name):
    url = f'https://v3-api-indoor-duration.natagw.my.id/api/3/image/{image_file_name}'
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {bearer_token}',  # Interpolating the token
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    
def save_and_extract_embeddings(recognizer=None, detector=None):
    all_assistant_embed_false = GetAllAssistant()
    if not all_assistant_embed_false:
        print("Tidak ada data asisten untuk ditraining.")
        return

    dataset_dir = 'data'
    os.makedirs(dataset_dir, exist_ok=True)

    for assistant in all_assistant_embed_false:
        # replace personname to string
        person_name = assistant['name']
        image_file_name = assistant['image']
        
        person_dir = os.path.join(dataset_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)

        img = GetImage(image_file_name)
        if img is None:
            continue

        img_format = img.format.lower()
        ext = 'jpg' if img_format == 'jpeg' else img_format
        save_path = os.path.join(person_dir, f"{person_name}.{ext}")
        try:
            img.save(save_path)
            print(f"Image saved to {save_path}")
        except Exception as e:
            print(f"Gagal menyimpan gambar {image_file_name}: {e}")
    
    # Proses embedding
    res=extract_embeddings(dataset_dir, recognizer=recognizer, detector=detector)
    
    if res:
        # remove dataset_dir after embedding
        for person_name in os.listdir(dataset_dir):
            person_path = os.path.join(dataset_dir, person_name)
            if os.path.isdir(person_path):
                shutil.rmtree(person_path, ignore_errors=True)
                
        for assistant in all_assistant_embed_false:
            uid = assistant['uid']
            # Update status_embed to true
            url = f'https://v3-api-indoor-duration.natagw.my.id/api/3/admin/assistant/{uid}'
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {bearer_token}',  # Interpolating the token
            }
            requests.patch(url, headers=headers)
            
        
    