from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

OUT = Path(__file__).resolve().parents[1] / "data" / "sms_spam"
OUT.mkdir(parents=True, exist_ok=True)
archive = OUT / "smsspamcollection.zip"
urlretrieve(
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    archive,
)
with ZipFile(archive) as zip_file:
    zip_file.extract("SMSSpamCollection", OUT)
archive.unlink()
print(f"SMS Spam Collection saved to {OUT}")
