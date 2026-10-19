import io
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import cv2
import soundfile as sf
import speech_recognition as sr

st.set_page_config(page_title="Multimodal Attendance & Engagement Agent", page_icon="🎓", layout="wide")

st.title("🎓 Multimodal Attendance & Engagement Agent")
st.caption("Face detection + opt-in enrolled-person matching + audio transcription")
st.warning(
    "Privacy note: Use only with participants' informed consent. Face matching is a simple classroom/demo "
    "prototype, not a secure identity verification system. Engagement indicators are approximate and must "
    "not be used alone for grading, discipline, hiring, or other high-stakes decisions."
)

@st.cache_resource
def get_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("OpenCV face detector could not be loaded.")
    return cascade

def image_to_rgb(upload):
    image = Image.open(upload)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.array(image)

def detect_faces(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = get_face_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(45, 45)
    )
    return gray, faces

def face_crop(gray, box, size=(160, 160)):
    x, y, w, h = [int(v) for v in box]
    crop = gray[y:y+h, x:x+w]
    if crop.size == 0:
        return None
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)

def lbph_available():
    return hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")

def train_recognizer(enrollment):
    if not lbph_available():
        return None, "Face matching library unavailable. Check opencv-contrib-python-headless installation."
    images, labels = [], []
    label_names = {}
    next_label = 0
    for name, samples in enrollment.items():
        if not samples:
            continue
        label_names[next_label] = name
        for sample in samples:
            images.append(sample)
            labels.append(next_label)
        next_label += 1
    if len(images) < 2:
        return None, "Upload at least two clear enrollment photos (preferably more) for a person."
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, np.asarray(labels, dtype=np.int32))
    return (model, label_names), None

def recognize_crop(model_data, crop, threshold=75.0):
    if model_data is None or crop is None:
        return "Unknown", None
    model, names = model_data
    label, distance = model.predict(crop)
    # LBPH distance is lower for a closer match. Threshold is approximate and should be tested.
    name = names.get(int(label), "Unknown") if float(distance) <= threshold else "Unknown"
    return name, float(distance)

tab1, tab2, tab3 = st.tabs(["📷 Face detection & matching", "🔊 Audio transcription", "📊 Attendance log"])

with tab1:
    st.subheader("1. Enrol participants (opt-in)")
    st.write("Har person ke liye naam aur 2–5 clear, front-facing photos upload karo. Ye enrollment is browser session mein rahega.")
    name = st.text_input("Participant name", placeholder="e.g. Rahul")
    enroll_files = st.file_uploader(
        "Enrollment photos (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="enroll_photos"
    )
    if st.button("Save enrollment photos", type="primary"):
        if not name.strip():
            st.error("Pehle participant ka naam likho.")
        elif not enroll_files:
            st.error("Kam se kam 2 photos upload karo.")
        else:
            samples = []
            for f in enroll_files:
                try:
                    rgb = image_to_rgb(f)
                    gray, faces = detect_faces(rgb)
                    if len(faces) == 0:
                        st.warning(f"{f.name}: face nahi mila; is photo ko skip kiya.")
                        continue
                    # Use the largest detected face for enrollment.
                    box = max(faces, key=lambda b: int(b[2]) * int(b[3]))
                    crop = face_crop(gray, box)
                    if crop is not None:
                        samples.append(crop)
                except Exception as e:
                    st.warning(f"{f.name}: {e}")
            if samples:
                if "enrollment" not in st.session_state:
                    st.session_state.enrollment = {}
                st.session_state.enrollment[name.strip()] = samples
                st.success(f"{name.strip()} ke {len(samples)} face sample save hue.")
            else:
                st.error("Kisi photo se face sample nahi mila. Better lighting aur front-facing photo try karo.")

    enrollment = st.session_state.get("enrollment", {})
    if enrollment:
        st.write("Enrolled participants:", {k: len(v) for k, v in enrollment.items()})
        if st.button("Clear all enrollment data"):
            st.session_state.enrollment = {}
            st.rerun()

    st.divider()
    st.subheader("2. Upload an image to detect and match faces")
    test_img = st.file_uploader("Test image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="test_img")
    threshold = st.slider("Matching strictness (LBPH distance; lower is stricter)", 35, 120, 75)
    if test_img is not None:
        try:
            rgb = image_to_rgb(test_img)
            gray, faces = detect_faces(rgb)
            model_data, train_error = train_recognizer(enrollment) if enrollment else (None, "No participants enrolled.")
            annotated = rgb.copy()
            records = []
            for i, box in enumerate(faces, start=1):
                crop = face_crop(gray, box)
                person, distance = recognize_crop(model_data, crop, threshold)
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 180, 0), 2)
                label = person if person != "Unknown" else f"Unknown face {i}"
                cv2.putText(annotated, label[:28], (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
                records.append({"Face": i, "Match": person, "LBPH distance": round(distance, 2) if distance is not None else None})
            st.image(annotated, caption=f"Detected faces: {len(faces)}", use_container_width=True)
            if train_error:
                st.info(train_error + " Detection still works; identity matching needs enrolled photos.")
            st.dataframe(pd.DataFrame(records) if records else pd.DataFrame([{"Result": "No faces detected"}]), use_container_width=True)
            if records:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_rows = [{"Timestamp": now, "Name": r["Match"], "Result": "Matched" if r["Match"] != "Unknown" else "Unidentified"} for r in records]
                if "attendance_log" not in st.session_state:
                    st.session_state.attendance_log = []
                st.session_state.attendance_log.extend(log_rows)
        except Exception as e:
            st.error(f"Image processing error: {e}")

    st.divider()
    st.subheader("3. Video face detection (sampled frames)")
    video = st.file_uploader("Video (MP4/AVI/MOV)", type=["mp4", "avi", "mov"], key="video")
    if video is not None and st.button("Analyze video"):
        tmp_path = None
        try:
            suffix = os.path.splitext(video.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(video.getbuffer())
                tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                st.error("Video open nahi hua. MP4 (H.264) format try karo.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                duration = total_frames / fps if fps else 0
                step = max(1, int(fps * 2))  # sample every ~2 seconds
                counts, sampled = [], 0
                model_data, train_error = train_recognizer(enrollment) if enrollment else (None, "No participants enrolled.")
                preview = None
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if frame_index % step != 0:
                        continue
                    sampled += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gray, faces = detect_faces(rgb)
                    counts.append(len(faces))
                    for box in faces:
                        crop = face_crop(gray, box)
                        person, _ = recognize_crop(model_data, crop, threshold)
                        x, y, w, h = [int(v) for v in box]
                        cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 180, 0), 2)
                        cv2.putText(rgb, person, (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
                    preview = rgb
                cap.release()
                if preview is not None:
                    st.image(preview, caption="Last sampled frame", use_container_width=True)
                st.metric("Video duration (approx.)", f"{duration:.1f} sec")
                st.metric("Frames sampled", sampled)
                st.metric("Average faces per sampled frame", f"{np.mean(counts):.2f}" if counts else "0")
                if train_error:
                    st.info(train_error)
                if counts:
                    st.line_chart(pd.DataFrame({"Sample": range(1, len(counts)+1), "Face count": counts}).set_index("Sample"))
        except Exception as e:
            st.error(f"Video processing error: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

with tab2:
    st.subheader("Audio: speech-to-text + signal metrics")
    st.info("Speech transcription uses Google's online speech recognition service, so internet access is required. WAV is the most reliable upload format.")
    audio_file = st.file_uploader("Upload audio (WAV/FLAC)", type=["wav", "flac"], key="audio")
    if audio_file is not None and st.button("Transcribe audio", type="primary"):
        try:
            audio_bytes = audio_file.getvalue()
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
            mono = np.mean(audio_data, axis=1)
            duration = len(mono) / sample_rate if sample_rate else 0
            rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
            peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
            st.write(f"Duration: **{duration:.2f} seconds** · Sample rate: **{sample_rate} Hz**")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMS loudness", f"{rms:.4f}")
            c2.metric("Peak amplitude", f"{peak:.4f}")
            c3.metric("Audio status", "Audio present" if rms > 0.001 else "Very quiet / silent")
            # SpeechRecognition Google API accepts PCM WAV or FLAC; write normalized mono PCM WAV.
            pcm = np.clip(mono, -1, 1)
            pcm16 = (pcm * 32767).astype(np.int16)
            import wave
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(int(sample_rate))
                wf.writeframes(pcm16.tobytes())
            wav_buffer.seek(0)
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_buffer) as source:
                audio_content = recognizer.record(source)
            try:
                transcript = recognizer.recognize_google(audio_content, language="en-IN")
                st.success("Speech transcription complete")
                st.text_area("Recognized speech", value=transcript, height=150)
                st.download_button("Download transcript (.txt)", transcript, file_name="transcript.txt", mime="text/plain")
            except sr.UnknownValueError:
                st.warning("Speech samajh nahi aayi. Clear speech, less background noise, aur WAV file try karo.")
            except sr.RequestError as e:
                st.error(f"Speech service connect nahi hui: {e}. Internet/service access check karo.")
        except Exception as e:
            st.error(f"Audio processing error: {e}")

with tab3:
    st.subheader("Session attendance log")
    rows = st.session_state.get("attendance_log", [])
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download attendance log CSV", df.to_csv(index=False), file_name="attendance_log.csv", mime="text/csv")
    else:
        st.caption("Abhi koi face matching attempt log nahi hua. Image tab mein test image upload karo.")
