# config.py — read config from environment
import os
import cloudinary
import cloudinary.uploader


class Config:
    EMAIL_SENDER = os.getenv("EMAIL_SENDER")
    EMAIL_PASSWORD_SENDER = os.getenv("EMAIL_PASSWORD_SENDER")
    FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
    MOBILE_SENDER_NUMBER = os.getenv("Admin_mobile_number")
    FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    FAST2SMS_SENDER_ID = os.getenv("FAST2SMS_SENDER_ID")
    Instagram = os.getenv("INSTAGRAM")
    Facebook = os.getenv("FACEBOOK")
    Twitter = os.getenv("TWITTER")
    Youtube = os.getenv("YOUTUBE")
    github = os.getenv("GITHUB")
    Linkedin = os.getenv("LINKEDIN")

    # Cloudinary Credentials
    CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

    #
    CLOUDINARY_ENABLED = all(
        [CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET]
    )


if Config.CLOUDINARY_ENABLED:
    cloudinary.config(
        cloud_name=Config.CLOUDINARY_CLOUD_NAME,
        api_key=Config.CLOUDINARY_API_KEY,
        api_secret=Config.CLOUDINARY_API_SECRET,
        secure=True,
    )
