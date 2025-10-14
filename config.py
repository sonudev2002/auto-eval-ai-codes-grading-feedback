# config.py — central config loader
import os
import cloudinary
import cloudinary.uploader


class Config:
    # 🔐 Brevo SMTP
    EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp-relay.brevo.com")
    EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    EMAIL_SMTP_LOGIN = os.getenv("EMAIL_SMTP_LOGIN", "98a206004@smtp-brevo.com")
    EMAIL_PASSWORD_SENDER = os.getenv("EMAIL_PASSWORD_SENDER")
    EMAIL_SENDER = os.getenv("EMAIL_SENDER")

    # 🔑 Brevo API (used by notification_system)
    BREVO_API_KEY = os.getenv("BREVO_API_KEY")

    # 🔒 Flask + Other Secrets
    FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
    MOBILE_SENDER_NUMBER = os.getenv("Admin_mobile_number")
    FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")
    FAST2SMS_SENDER_ID = os.getenv("FAST2SMS_SENDER_ID")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # 🌐 Social links
    Instagram = os.getenv("INSTAGRAM")
    Facebook = os.getenv("FACEBOOK")
    Twitter = os.getenv("TWITTER")
    Youtube = os.getenv("YOUTUBE")
    github = os.getenv("GITHUB")
    Linkedin = os.getenv("LINKEDIN")

    # ☁️ Cloudinary
    CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

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
