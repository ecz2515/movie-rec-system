import requests
from dotenv import load_dotenv
import os

def load_api_key():
    load_dotenv()
    api_key = os.getenv('API_KEY')
    if api_key is not None:
        return api_key
    else:
        print("API Key not found. Make sure it's defined in the .env file.")
        return None

def make_movie_api_call(api_key, movie_name):
    url = f"https://moviesdatabase.p.rapidapi.com/titles/search/title/{movie_name}"
    querystring = {"exact": "true", "info": "base_info", "titleType": "movie", "limit": "1"}
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    return response.json()['results'][0]
