import requests
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

# Now you can access the environment variables
Application_id = os.getenv('Application_id')
Api_key = os.getenv('API_KEY')
GENDER = "Female"
WEIGHT_KG = 55
HEIGHT_CM = 157 
AGE = 27


user_response = input("Tell me which exercises you did:")

url_endpoint = "https://trackapi.nutritionix.com/v2/natural/exercise"

headers = {
    "x-app-id" : Application_id,
    "x-app-key" : Api_key,
    "content-type" : "application/json"
}

# Set up the request body with the user input
parameters = {
    "query": user_response,
    "gender": GENDER,
    "weight_kg": WEIGHT_KG,
    "height_cm": HEIGHT_CM,
    "age": AGE

}

response = requests.post(url=url_endpoint,headers=headers,json=parameters)
 
results = response.json()

print(results)

sheety_endpoint = "https://api.sheety.co/5d102fb67fc1b5a690ec2b305a71957b/workoutTracking/sheet1"

today_date = datetime.now().strftime("%d/%m/%Y")
today_time = datetime.now().strftime("%X")

for exercise in results["exercises"]:
    sheet_inputs = {
    "sheet1" : {
        "date" : today_date,
        "time" : today_time,
        "exercise": exercise["name"].title(),
        "duration": exercise["duration_min"],
        "calories": exercise["nf_calories"] 
    }
    }
print("Payload sent to Sheety:", sheet_inputs)  # Debugging line to check the payload


sheet_response = requests.post(sheety_endpoint, json=sheet_inputs)

print(sheet_response.text)

if sheet_response.status_code == 200:
        print("Data successfully sent to Sheety:", sheet_response.text)
else:
        print(f"Failed to send data to Sheety. Status code: {sheet_response.status_code}")
        print(sheet_response.text)

