import json

import requests

from IDLE.commons.commonUtils import download_dataset

#
url = "https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json"
#
save_path = "C:/Users/Brahmendra Bachi/PycharmProjects/tensorflow-practice/Data/test.json"

file_path = "sarcasm.json"

# Send a GET request to the URL
response = requests.get(url)


if response.status_code == 200:
    print(response.content)
    # Write the content of the response to a file
    with open(save_path, "wb") as file:
        file.write(response.content)
    print(f"File '{file_path}' downloaded successfully!")
else:
    print(f"Failed to download file from '{url}'. Status code: {response.status_code}")