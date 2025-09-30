# config.py — read config from environment
import os


class Config:
    EMAIL_SENDER = os.getenv("EMAIL_SENDER")
    EMAIL_PASSWORD_SENDER = os.getenv("EMAIL_PASSWORD_SENDER")
    FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
    MOBILE_SENDER_NUMBER = os.getenv("FAST2SMS_NUMBER")
    FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    FAST2SMS_SENDER_ID = os.getenv("FAST2SMS_SENDER_ID")
    Instagram = os.getenv("INSTAGRAM")
    Facebook = os.getenv("FACEBOOK")
    Twitter = os.getenv("TWITTER")
    Youtube = os.getenv("YOUTUBE")
    github = os.getenv("GITHUB")
    Linkedin = os.getenv("LINKEDIN")
