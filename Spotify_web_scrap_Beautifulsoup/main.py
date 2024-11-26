import requests
from bs4 import BeautifulSoup

# Prompt the user for the date
date_input = input("Enter the date you'd like to travel to (YYYY-MM-DD): ")

# Define the URL for the Billboard Hot 100 on the given date
url = f"https://www.billboard.com/charts/hot-100/{date_input}/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0"
}

# Make the request
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

   
    song_names_spans = soup.select("li ul li h3")
    song_names = [song.getText().strip() for song in song_names_spans]

if song_names:
        print(f"Top 100 songs on {date_input}:")
        print(song_names)

