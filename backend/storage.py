cat > storage.py << "EOF"
import os
import boto3
from werkzeug.utils import secure_filename

S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

if S3_BUCKET:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION,
    )
else:
    s3 = None


def make_key_for_file(file):
    import uuid

    filename = secure_filename(file.filename)
    return f"uploads/{uuid.uuid4().hex}_{filename}"


def upload_fileobj(file_obj, dest_key):
    if not s3 or not S3_BUCKET:
        raise RuntimeError("S3 not configured")
    s3.upload_fileobj(file_obj, S3_BUCKET, dest_key, ExtraArgs={"ACL": "private"})
    return f"https://{S3_BUCKET}.s3.amazonaws.com/{dest_key}"


def delete_s3_key(url_or_key):
    if not s3 or not S3_BUCKET:
        return
    # Accept either full URL or key
    if url_or_key.startswith("http"):
        from urllib.parse import urlparse

        p = urlparse(url_or_key)
        # try to extract key by removing leading path part
        key = p.path.lstrip("/")
        # if URL pattern was bucket.s3.amazonaws.com/key then key is correct
    else:
        key = url_or_key
    if key:
        s3.delete_object(Bucket=S3_BUCKET, Key=key)


EOF
