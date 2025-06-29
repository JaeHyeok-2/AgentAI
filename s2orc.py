import requests

base_url = "https://api.semanticscholar.org/datasets/v1/release/"

# This endpoint requires authentication via api key
api_key = "your api key goes here"
headers = {"x-api-key": api_key}

# Set the release id
release_id = "2025-06-17"

# Define dataset name you want to download
dataset_name = 'papers'

# Send the GET request and store the response in a variable
response = requests.get(base_url + release_id + '/dataset/' + dataset_name, headers=headers)

# Process and print the response data
print(response.json())