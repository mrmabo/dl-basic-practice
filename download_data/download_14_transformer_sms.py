from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
OUT = Path(__file__).resolve().parents[1] / "data" / "transformer_sms_spam"
OUT.mkdir(parents=True, exist_ok=True)
archive = OUT / "smsspamcollection.zip"
urlretrieve(
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    archive,
)
with ZipFile(archive) as zip_file:
    zip_file.extract("SMSSpamCollection", OUT)
archive.unlink()
AutoTokenizer.from_pretrained(MODEL_NAME)
AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
print(f"SMS data and DistilBERT files are ready: {OUT}")
