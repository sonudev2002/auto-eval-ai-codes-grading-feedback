import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


class Config:
    EMAIL_SENDER = os.getenv("Admin_email")
    EMAIL_PASSWORD_SENDER = os.getenv("Admin_email_password")
    FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
    MOBILE_SENDER_NUMBER = os.getenv("Admin_mobile_number")
    FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    FAST2SMS_SENDER_ID = os.getenv("FAST2SMS_SENDER_ID")
    Instagram = os.getenv("Instagram")
    Facebook = os.getenv("Facebook")
    Twitter = os.getenv("Twitter")
    Youtube = os.getenv("Youtube")
    github = os.getenv("github")
    Linkedin = os.getenv("Linkedin")
