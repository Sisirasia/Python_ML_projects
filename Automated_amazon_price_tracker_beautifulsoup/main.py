from bs4 import BeautifulSoup
import requests
import smtplib
import os

# Email settings
EMAIL_ADDRESS = os.environ.get ('EMAIL_ADDRESS') # Replace with your email address
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")  # Replace with your email password
TO_EMAIL = "your_email@example.com"  # Recipient email address (can be the same as sender)
TARGET_PRICE = 100  # Set your target price

# URLs for scraping
practice_url = "https://appbrewery.github.io/instant_pot/"
# Uncomment the live_url to monitor the live product (ensure compliance with Amazon's terms of use)
# live_url = "https://www.amazon.com/dp/B075CYMYK6?psc=1&ref_=cm_sw_r_cp_ud_ct_FM9M699VKHTT47YD50Q6"

# Send a notification email
def send_email(product_title, current_price, product_url):
    with smtplib.SMTP("smtp.gmail.com", 587) as connection:
        connection.starttls()  # Secure the connection
        connection.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        subject = "Price Alert: Product Below Target Price!"
        body = (
            f"Good news!\n\n"
            f"The product '{product_title}' is now available at ${current_price}.\n"
            f"Check it out here: {product_url}"
        )
        message = f"Subject: {subject}\n\n{body}"
        connection.sendmail(EMAIL_ADDRESS, TO_EMAIL, message)
        print("Email sent successfully!")

# Scrape the page
response = requests.get(practice_url)
soup = BeautifulSoup(response.content, "html.parser")

# Extract product details
product_title = soup.find(class_="a-size-large product-title-word-break").get_text(strip=True)
price = soup.find(class_="a-offscreen").get_text()
price_without_currency = price.split("$")[1]  # Remove the dollar sign
price_as_float = float(price_without_currency)

# Check if the price is below the target and send email
if price_as_float < TARGET_PRICE:
    print(f"The price is below ${TARGET_PRICE}. Sending email...")
    send_email(product_title, price_as_float, practice_url)
else:
    print(f"The price is ${price_as_float}, which is above the target price of ${TARGET_PRICE}.")
