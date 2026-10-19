# Multimodal Attendance & Engagement Agent (fixed prototype)

## Features
- OpenCV Haar-cascade face detection for uploaded images and sampled video frames.
- Optional enrolled-person matching using OpenCV LBPH (requires consent and enrollment photos).
- WAV/FLAC audio metrics and online speech-to-text using SpeechRecognition's Google recognizer.
- Session-only enrollment and downloadable attendance CSV.

## Run locally
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, and `README.md` to your GitHub repository root.
2. Open https://share.streamlit.io/ and select the repository/branch and `app.py`.
3. Deploy, then check the app logs if dependency installation fails.
4. For an existing app, push updated files to the branch connected to Streamlit; it will rebuild automatically.

## Important limitations
- Haar cascade detects faces; it does not identify a person by itself.
- LBPH matching is a basic demo method, sensitive to lighting, angle, and image quality. Enroll 2–5 clear photos per participant. "Unknown" is expected when similarity is poor. It is not secure authentication.
- Enrollment is held in Streamlit session memory and may disappear on refresh/restart.
- Speech-to-text requires internet access to Google's online service; transcription may fail if the service is unavailable or speech is unclear. WAV is recommended.
- This is not a production biometric attendance system. Get consent, protect uploaded media, and do not use approximate engagement measures for high-stakes decisions.
