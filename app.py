import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import os
from datetime import datetime
import pandas as pd
import time
import google.generativeai as genai

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Page config
st.set_page_config(page_title="Emotion Recognition Prototype", layout="wide")

# Improved modern GUI styles
st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #1a2a44, #0f1c33);}
    .stApp {background: transparent;}
    h1, h2, h3, p, div, label, span {color: #f0f4f8 !important; font-family: 'Segoe UI', sans-serif;}
    .stButton > button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 114, 255, 0.5);
    }
    .emotion-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
        transition: transform 0.2s;
    }
    .emotion-badge:hover {transform: scale(1.05);}
    .card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .fade-in {animation: fadeIn 0.8s ease-in;}
    @keyframes fadeIn {from {opacity: 0;} to {opacity: 1;}}
    .progress-bar {height: 10px; background: #333; border-radius: 5px; overflow: hidden;}
    .progress-fill {height: 100%; transition: width 0.6s ease;}
    </style>
""", unsafe_allow_html=True)

st.title("Emotion Recognition Prototype")
st.caption("Live Face, Recorded Video & Audio in Indian Languages – 4th Sem AIML Project")

# Sidebar - Alert emotions
alert_emotions = st.sidebar.multiselect(
    "Alert on these emotions (pop-up messages)",
    ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"],
    default=[]
)

tabs = st.tabs(["Live Detection", "Recorded Video", "Group Simulation", "Recorded Audio", "Journal", "About"])

# Initialize session state
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'group_history' not in st.session_state:
    st.session_state.group_history = []
if 'alert_shown' not in st.session_state:
    st.session_state.alert_shown = set()
if 'last_face_emotion' not in st.session_state:
    st.session_state.last_face_emotion = None
if 'last_face_score' not in st.session_state:
    st.session_state.last_face_score = 0.0

# Gemini API key
GEMINI_API_KEY = "type you own api key"
genai.configure(api_key=GEMINI_API_KEY)

# ────────────────────────────────────────────────
# LIVE DETECTION TAB
# ────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Live Webcam Emotion Detection")
    st.write("Click 'Start' → OpenCV window opens. Press 'q' to stop.")

    if st.button("Start Live Detection"):
        st.session_state.emotion_history = []
        st.session_state.alert_shown = set()  # Reset alerts for new session
        start_time = time.time()

        cap = cv2.VideoCapture(0)
        history = []
        annotated_path = "live_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 15
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(annotated_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time() - start_time

            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                if result and len(result) > 0:
                    dominant = result[0]['dominant_emotion']
                    score = result[0]['emotion'][dominant] * 100

                    x, y, w, h = result[0]['region']['x'], result[0]['region']['y'], result[0]['region']['w'], result[0]['region']['h']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    label = f"{dominant.upper()} {score:.0f}%"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    history.append({
                        "timestamp": f"{current_time:.1f}s",
                        "emotion": dominant,
                        "confidence": round(score, 1)
                    })

                    st.session_state.last_face_emotion = dominant
                    st.session_state.last_face_score = score

                    # Alert every time the emotion is detected (no session limit)
                    if dominant in alert_emotions:
                        st.warning(f"Alert: Detected {dominant.upper()} at {current_time:.1f}s")
            except:
                pass

            out.write(frame)

            cv2.imshow("Live Detection - Press 'q' to stop", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if history:
            df = pd.DataFrame(history)
            st.session_state.emotion_history.append({
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "Live Webcam",
                "summary": df
            })

            st.success("Detection stopped.")
            if st.button("Show emotion summary with timestamp", key="live_summary"):
                st.table(df)

            st.video(annotated_path)
            with open(annotated_path, "rb") as f:
                st.download_button(
                    label="Download Annotated Video",
                    data=f,
                    file_name="live_emotions.mp4",
                    mime="video/mp4"
                )
        else:
            st.warning("No emotions detected.")

# ────────────────────────────────────────────────
# RECORDED VIDEO TAB
# ────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Recorded Video Analysis")
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov"])

    if uploaded:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getvalue())

        st.video(temp_path)

        if st.button("Analyze Recorded Video"):
            st.session_state.alert_shown = set()  # Reset alerts
            cap = cv2.VideoCapture(temp_path)
            history = []
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            annotated_path = "recorded_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            out = cv2.VideoWriter(annotated_path, fourcc, fps, (frame_width, frame_height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                timestamp = frame_count / fps

                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                    if result and len(result) > 0:
                        dominant = result[0]['dominant_emotion']
                        score = result[0]['emotion'][dominant] * 100

                        x, y, w, h = result[0]['region']['x'], result[0]['region']['y'], result[0]['region']['w'], result[0]['region']['h']
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = f"{dominant.upper()} {score:.0f}%"
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        history.append({
                            "timestamp": f"{timestamp:.1f}s",
                            "emotion": dominant,
                            "confidence": round(score, 1)
                        })

                        st.session_state.last_face_emotion = dominant
                        st.session_state.last_face_score = score

                        if dominant in alert_emotions:
                            st.warning(f"Alert: Detected {dominant.upper()} at {timestamp:.1f}s")
                except:
                    pass

                out.write(frame)

            cap.release()
            out.release()

            if history:
                df = pd.DataFrame(history)
                st.session_state.emotion_history.append({
                    "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "Recorded Video",
                    "summary": df
                })

                st.success("Analysis Complete")
                if st.button("Show emotion summary with timestamp", key="recorded_summary"):
                    st.table(df)

                st.video(annotated_path)
                with open(annotated_path, "rb") as f:
                    st.download_button(
                        label="Download Annotated Video",
                        data=f,
                        file_name="recorded_emotions.mp4",
                        mime="video/mp4"
                    )
            else:
                st.warning("No emotions detected.")

# ────────────────────────────────────────────────
# GROUP SIMULATION TAB
# ────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Group Meet Simulation")
    st.write("Option 1: Live webcam multi-face detection")
    st.write("Option 2: Upload a recorded group video")

    mode = st.radio("Choose Mode", ["Live Webcam", "Upload Recorded Group Video"])

    if mode == "Live Webcam":
        if st.button("Start Live Group Detection"):
            st.session_state.alert_shown = set()  # Reset alerts
            cap = cv2.VideoCapture(0)
            group_history = []
            person_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_copy = frame.copy()
                try:
                    results = DeepFace.analyze(frame_copy, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                    if results:
                        for idx, result in enumerate(results):
                            dominant = result['dominant_emotion']
                            score = result['emotion'][dominant] * 100

                            x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
                            color = person_colors[idx % len(person_colors)]
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            label = f"Person {idx+1}: {dominant.upper()} {score:.0f}%"
                            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                            group_history.append({
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "person_id": idx+1,
                                "emotion": dominant,
                                "confidence": round(score, 1)
                            })

                            if dominant in alert_emotions:
                                st.warning(f"Alert: Person {idx+1} felt {dominant.upper()} at {datetime.now().strftime('%H:%M:%S')}")
                except:
                    pass

                cv2.imshow("Group Detection - Press 'q' to stop", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

            if group_history:
                df = pd.DataFrame(group_history)
                st.session_state.group_history.append({
                    "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "Live Group",
                    "summary": df
                })

                st.success("Group detection stopped.")
                if st.button("Show emotion summary with timestamp", key="group_live_summary"):
                    persons = df['person_id'].unique()
                    for person_id in sorted(persons):
                        person_df = df[df['person_id'] == person_id]
                        overall_mode = person_df['emotion'].mode()[0] if not person_df.empty else "None"
                        color_class = f"person{person_id}" if person_id <= 3 else "person3"
                        st.markdown(f"<div class='person-box {color_class}'>**Person {person_id} Summary** (Overall Mode: {overall_mode.upper()})</div>", unsafe_allow_html=True)
                        st.table(person_df)

    elif mode == "Upload Recorded Group Video":
        uploaded = st.file_uploader("Upload group video", type=["mp4", "mov"])

        if uploaded:
            temp_path = "temp_group.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded.getvalue())

            st.video(temp_path)

            if st.button("Analyze Uploaded Group Video"):
                st.session_state.alert_shown = set()  # Reset alerts
                cap = cv2.VideoCapture(temp_path)
                group_history = []
                person_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                frame_count = 0
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                annotated_path = "group_annotated.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))
                out = cv2.VideoWriter(annotated_path, fourcc, fps, (frame_width, frame_height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    timestamp = frame_count / fps

                    frame_copy = frame.copy()
                    try:
                        results = DeepFace.analyze(frame_copy, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                        if results:
                            for idx, result in enumerate(results):
                                dominant = result['dominant_emotion']
                                score = result['emotion'][dominant] * 100

                                x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
                                color = person_colors[idx % len(person_colors)]
                                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                                label = f"Person {idx+1}: {dominant.upper()} {score:.0f}%"
                                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                                group_history.append({
                                    "timestamp": f"{timestamp:.1f}s",
                                    "person_id": idx+1,
                                    "emotion": dominant,
                                    "confidence": round(score, 1)
                                })

                                if dominant in alert_emotions:
                                    st.warning(f"Alert: Person {idx+1} felt {dominant.upper()} at {timestamp:.1f}s")
                    except:
                        pass

                    out.write(frame)

                cap.release()
                out.release()

                if group_history:
                    df = pd.DataFrame(group_history)
                    st.session_state.group_history.append({
                        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "Uploaded Group Video",
                        "summary": df
                    })

                    st.success("Analysis Complete")
                    if st.button("Show emotion summary with timestamp", key="group_upload_summary"):
                        persons = df['person_id'].unique()
                        for person_id in sorted(persons):
                            person_df = df[df['person_id'] == person_id]
                            overall_mode = person_df['emotion'].mode()[0] if not person_df.empty else "None"
                            color_class = f"person{person_id}" if person_id <= 3 else "person3"
                            st.markdown(f"<div class='person-box {color_class}'>**Person {person_id} Summary** (Overall Mode: {overall_mode.upper()})</div>", unsafe_allow_html=True)
                            st.table(person_df)

                    st.video(annotated_path)
                    with open(annotated_path, "rb") as f:
                        st.download_button(
                            label="Download Annotated Group Video",
                            data=f,
                            file_name="group_emotions.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.warning("No emotions detected.")

# ────────────────────────────────────────────────
# RECORDED AUDIO TAB
# ────────────────────────────────────────────────
# ────────────────────────────────────────────────
# RECORDED AUDIO TAB (with full audio transcription + translation)
# ────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Recorded Audio Emotion Analysis")
    st.write("Upload audio in official Indian languages.")

    # Language selection for translation output
    output_lang = st.selectbox(
        "Translate transcription to:",
        ["English", "Hindi", "Tamil", "Bengali", "Telugu"]
    )

    audio_uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])

    if audio_uploaded:
        audio_path = "temp_audio" + os.path.splitext(audio_uploaded.name)[1]
        with open(audio_path, "wb") as f:
            f.write(audio_uploaded.getvalue())

        st.audio(audio_path)

        if st.button("Analyze"):
            with st.spinner("Processing audio (language detection + transcription + translation + emotion)..."):
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    audio_file = genai.upload_file(audio_path)

                    # Step 1: Detect language
                    lang_prompt = """
                    Identify the primary spoken language in this audio clip.
                    It is one of the official Indian languages (Hindi, Bengali, Tamil, Telugu, Malayalam, Kannada, Punjabi, Gujarati, Marathi, Odia, Assamese, Urdu, etc.).
                    Return only the language name, e.g., "Bengali" or "Hindi" or "Unknown".
                    """

                    lang_response = model.generate_content([lang_prompt, audio_file])
                    detected_lang = lang_response.text.strip()
                    st.info(f"**Detected Language:** {detected_lang}")

                    # Step 2: Full transcription (speech-to-text) in original language
                    transcribe_prompt = f"""
                    Transcribe the full spoken content of this audio clip accurately.
                    The language is {detected_lang}.
                    Return only the transcription text (no extra explanation).
                    """

                    transcribe_response = model.generate_content([transcribe_prompt, audio_file])
                    original_transcript = transcribe_response.text.strip()
                    st.markdown("**Original Transcription:**")
                    st.write(original_transcript)

                    # Step 3: Translate transcription to selected language
                    if output_lang != "English" or detected_lang.lower() != "english":
                        translate_prompt = f"""
                        Translate the following transcription to {output_lang}:
                        {original_transcript}
                        Return only the translated text (no extra explanation).
                        """

                        translate_response = model.generate_content(translate_prompt)
                        translated_transcript = translate_response.text.strip()
                        st.markdown(f"**Translated Transcription ({output_lang}):**")
                        st.write(translated_transcript)
                    else:
                        st.markdown("**Translated Transcription:** (Original is already in English)")
                        st.write(original_transcript)

                    # Step 4: Emotion analysis
                    emotion_prompt = f"""
                    Analyze this audio clip spoken in {detected_lang}.
                    Identify the dominant emotion: happy, sad, angry, neutral, bored, fear, surprise, disgust.
                    Return only: EMOTION: <emotion> | CONFIDENCE: <0-100>%
                    """

                    emotion_response = model.generate_content([emotion_prompt, audio_file])
                    result_text = emotion_response.text.strip()

                    emotion = "Unknown"
                    confidence = 0.0
                    for line in result_text.split('\n'):
                        if 'EMOTION:' in line:
                            emotion = line.split('EMOTION:')[1].split('|')[0].strip()
                        if 'CONFIDENCE:' in line:
                            confidence = float(line.split('CONFIDENCE:')[1].strip().replace('%', ''))

                    st.success(f"Speech Emotion: {emotion.upper()} ({confidence:.1f}%)")

                    # Fusion with last face (if available)
                    if 'last_face_emotion' in st.session_state:
                        face_emo = st.session_state.last_face_emotion
                        face_score = st.session_state.last_face_score
                        fused_emo = emotion if confidence > face_score else face_emo
                        fused_score = max(confidence, face_score)
                        st.markdown("### Fused (Face + Speech)")
                        st.success(f"{fused_emo.upper()} ({fused_score:.1f}%)")

                        st.session_state.emotion_history.append({
                            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "type": "Recorded Audio + Face Fusion",
                            "summary": pd.DataFrame([{
                                "emotion": fused_emo,
                                "confidence": round(fused_score, 1)
                            }])
                        })
                    else:
                        st.info("Run face detection first for fusion.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Tip: Try shorter audio (<30s) or different format (WAV preferred).")

        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)

# ────────────────────────────────────────────────
# JOURNAL TAB
# ────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Analysis Journal")
    if st.session_state.emotion_history or st.session_state.group_history:
        st.write("**Live, Recorded & Audio Analyses**")
        for idx, entry in enumerate(st.session_state.emotion_history):
            st.write(f"**Analysis {idx+1}** – {entry['analysis_time']} ({entry['type']})")
            if st.button("Show emotion summary with timestamp", key=f"journal_{idx}"):
                st.table(entry['summary'])
            st.markdown("---")

        st.write("**Group Analyses**")
        for idx, entry in enumerate(st.session_state.group_history):
            st.write(f"**Group Analysis {idx+1}** – {entry['analysis_time']} ({entry['type']})")
            if st.button("Show emotion summary with timestamp", key=f"group_journal_{idx}"):
                persons = entry['summary']['person_id'].unique()
                for person_id in sorted(persons):
                    person_df = entry['summary'][entry['summary']['person_id'] == person_id]
                    overall_mode = person_df['emotion'].mode()[0] if not person_df.empty else "None"
                    color_class = f"person{person_id}" if person_id <= 3 else "person3"
                    st.markdown(f"<div class='person-box {color_class}'>**Person {person_id} Summary** (Overall Mode: {overall_mode.upper()})</div>", unsafe_allow_html=True)
                    st.table(person_df)
            st.markdown("---")

        if st.button("Clear Journal"):
            st.session_state.emotion_history = []
            st.session_state.group_history = []
            st.success("Cleared!")
    else:
        st.info("No analyses yet.")

# ────────────────────────────────────────────────
# ABOUT TAB
# ────────────────────────────────────────────────
with tabs[5]:
    st.subheader("About")
    st.write("""
    This prototype captures and analyzes human emotions from live webcam, recorded videos, and audio recordings.
    It supports real-time face detection, multi-person group analysis, timestamped emotion tracking, and fusion of face and voice emotions.
    Designed especially for monitoring emotions in online education, mental health support, and communication in diverse Indian language contexts.
    Features include alerts for specific emotions, downloadable annotated videos, and a detailed analysis journal.
    """)

st.caption("4th Semester AIML Project • Tiruchirappalli")
