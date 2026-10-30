"""Task 4.2 - upload validation. Everything happens on bytes in memory; the
upload is never written to disk."""
import hashlib
import io
from pathlib import PurePath

from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}   # what the model was trained on
ALLOWED_FORMATS = {"JPEG", "PNG"}


class UploadError(Exception):
    def __init__(self, status, title, message):
        super().__init__(message)
        self.status, self.title, self.message = status, title, message


def validate_upload(file_storage):
    """Return (safe_filename, image_bytes, sha256) or raise UploadError."""
    if file_storage is None or not file_storage.filename:
        raise UploadError(400, "No file selected", "Choose an image to upload first.")

    # "../../etc/passwd.png" -> "etc_passwd.png"; only used for display/logging
    safe_name = secure_filename(file_storage.filename) or "upload"
    ext = PurePath(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(415, "Unsupported file type",
                          f"'{ext or 'no extension'}' is not accepted. Upload a JPG or PNG image.")

    data = file_storage.read()
    if not data:
        raise UploadError(400, "Empty file", "The uploaded file is empty.")

    # an extension proves nothing: really decode it
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise UploadError(422, "Image could not be read",
                          "The file is named like an image but its contents cannot be decoded. "
                          "It may be corrupt or not an image at all.")
    if fmt not in ALLOWED_FORMATS:
        raise UploadError(415, "Unsupported image format",
                          f"The file is actually {fmt}, not JPG or PNG.")
    return safe_name, data, hashlib.sha256(data).hexdigest()


def thumbnail_data_uri(data, size=320):
    """Small inline preview for the result page (kept in the response, not on disk)."""
    import base64
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
