# 🧠 Import libraries
import whisper
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import gradio as gr
import json
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline

# --- Model Download Paths ---
# Define your custom directories for each model
WHISPER_MODEL_DIR = "./models/whisper"
EMOTION_MODEL_DIR = "./models/emotion"
SUMMARIZER_MODEL_DIR = "./models/summarizer"
TRANSLATION_EN_HI_MODEL_DIR = "./models/translation_en_hi"
TRANSLATION_EN_KN_MODEL_DIR = "./models/translation_en_kn"
STABLE_DIFFUSION_MODEL_DIR = "./models/stable_diffusion"

# Create directories if they don't exist
os.makedirs(WHISPER_MODEL_DIR, exist_ok=True)
os.makedirs(EMOTION_MODEL_DIR, exist_ok=True)
os.makedirs(SUMMARIZER_MODEL_DIR, exist_ok=True)
os.makedirs(TRANSLATION_EN_HI_MODEL_DIR, exist_ok=True)
os.makedirs(TRANSLATION_EN_KN_MODEL_DIR, exist_ok=True)
os.makedirs(STABLE_DIFFUSION_MODEL_DIR, exist_ok=True)


# 🎙 Load Whisper model
# Use download_root to specify the download location for Whisper
whisper_model = whisper.load_model("medium", download_root=WHISPER_MODEL_DIR)

# ❤ Emotion detection model
emotion_model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name, cache_dir=EMOTION_MODEL_DIR)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name, cache_dir=EMOTION_MODEL_DIR)
emotion_pipeline = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, top_k=1)

# ✍ Summarizer (T5)
summarizer_tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir=SUMMARIZER_MODEL_DIR)
summarizer_model = T5ForConditionalGeneration.from_pretrained("t5-small", cache_dir=SUMMARIZER_MODEL_DIR)

# 🌐 Translation Pipelines
translation_en_hi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi", framework="pt", cache_dir=TRANSLATION_EN_HI_MODEL_DIR)
translation_en_kn = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mul", framework="pt", cache_dir=TRANSLATION_EN_KN_MODEL_DIR)


# 🎨 Text-to-Image Generator
pipe_sd = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, cache_dir=STABLE_DIFFUSION_MODEL_DIR)
pipe_sd = pipe_sd.to("cuda" if torch.cuda.is_available() else "cpu")

# 😊 Emoji board mapping (rest of your code remains the same)
def emoji_mood_board(emotion_label):
    board = {
        "joy": ["😊", "😄", "✨", "🌞", "🎉"],
        "sadness": ["😢", "💧", "💔", "😞", "☁"],
        "anger": ["😠", "🔥", "💢", "😤", "⚡"],
        "fear": ["😨", "😱", "🫣", "😰", "🚨"],
        "confusion": ["😕", "❓", "🤯", "🌀", "🤷"],
        "surprise": ["😲", "🤯", "😯", "✨", "🎊"],
        "love": ["❤", "😍", "💖", "💕", "💘"],
        "neutral": ["😐", "💬", "📖", "📝", "🔍"]
    }
    return board.get(emotion_label.lower(), ["🤔", "🫥", "🧠", "💭", "📘"])

# 🔍 Emotion Detection
def detect_emotion(text):
    results = emotion_pipeline(text)
    top = results[0][0]
    return top['label'], round(top['score'], 2)

# ✂ Text Summarization
def summarize_text(text, max_length=60):
    input_text = "summarize: " + text
    input_ids = summarizer_tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    summary_ids = summarizer_model.generate(input_ids, max_length=max_length, min_length=15, num_beams=4, length_penalty=2.0)
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 💾 Load/Save Diary Entries
def load_entries():
    return json.load(open("diary.json", "r")) if os.path.exists("diary.json") else []

def save_entries(entries):
    json.dump(entries, open("diary.json", "w"), indent=2)

# 🖼️ Generate Image from Summary
def generate_image(summary):
    image = pipe_sd(summary).images[0]
    image_path = "summary_image.png"
    image.save(image_path)
    return image_path

# 🎛 Main Audio Processor
def process_audio(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    result = whisper.decode(whisper_model, mel)
    transcript = result.text

    emotion, confidence = detect_emotion(transcript)
    summary = summarize_text(transcript)
    emojis = " ".join(emoji_mood_board(emotion))

    translation_hi = translation_en_hi(transcript)[0]["translation_text"]
    translation_kn = translation_en_kn(transcript)[0]["translation_text"]

    image_path = generate_image(summary)

    entry = {
        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": transcript,
        "summary": summary,
        "emotion": emotion,
        "emojis": emojis,
        "kannada": translation_kn,
        "hindi": translation_hi
    }

    entries = load_entries()
    entries.append(entry)
    save_entries(entries)

    return transcript, translation_kn, translation_hi, f"{emotion} ({confidence})", summary, emojis, image_path

# 📈 Plot Mood Graph
def plot_mood_graph():
    entries = load_entries()
    if not entries:
        return None
    recent = [e for e in entries if datetime.datetime.strptime(e['datetime'], "%Y-%m-%d %H:%M:%S") >= datetime.datetime.now() - datetime.timedelta(days=7)]
    if not recent:
        return None
    mood_map = {"joy": 5, "love": 5, "surprise": 4, "neutral": 3, "confusion": 2, "fear": 2, "sadness": 1, "anger": 0}
    times = [e['datetime'] for e in recent]
    scores = [mood_map.get(e['emotion'].lower(), 3) for e in recent]
    plt.figure(figsize=(10, 5))
    plt.plot(times, scores, marker='o')
    plt.xticks(rotation=45)
    plt.yticks(list(mood_map.values()), list(mood_map.keys()))
    plt.title("Mood Trend - Last 7 Days")
    plt.tight_layout()
    plt.savefig("mood_graph.png")
    return "mood_graph.png"

# 📁 Excel Upload Plot
def upload_excel(file):
    df = pd.read_excel(file.name)
    if "datetime" not in df.columns or "emotion" not in df.columns:
        return "Invalid format. Required: datetime, emotion"
    mood_map = {"joy": 5, "love": 5, "surprise": 4, "neutral": 3, "confusion": 2, "fear": 2, "sadness": 1, "anger": 0}
    datetimes = df["datetime"].astype(str).tolist()
    scores = [mood_map.get(e.lower(), 3) for e in df["emotion"]]
    plt.figure(figsize=(10, 5))
    plt.plot(datetimes, scores, marker='o', color='orange')
    plt.xticks(rotation=45)
    plt.yticks(list(mood_map.values()), list(mood_map.keys()))
    plt.title("Mood Graph from Excel")
    plt.tight_layout()
    path = "excel_mood_graph.png"
    plt.savefig(path)
    return path

# 🎨 Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 🎙 Voice Diary + Emotion + Translation + Summary + Image")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="🎤 Upload Audio (.mp3/.wav/.mp4)")

    with gr.Row():
        transcript = gr.Textbox(label="📝 English Transcript")
        kannada = gr.Textbox(label="🇮🇳 Kannada Translation")
        hindi = gr.Textbox(label="🇮🇳 Hindi Translation")

    with gr.Row():
        emotion = gr.Textbox(label="😊 Detected Emotion")
        summary = gr.Textbox(label="✂ Summary")
        emojis = gr.Textbox(label="🎭 Emoji Mood Board")

    with gr.Row():
        image_output = gr.Image(label="🎨 AI Image from Summary", show_label=True)

    with gr.Row():
        submit = gr.Button("Process")
        submit.click(process_audio, inputs=audio_input, outputs=[transcript, kannada, hindi, emotion, summary, emojis, image_output])

    with gr.Row():
        gr.Markdown("### 📈 Mood Graph (Last 7 Days)")
        mood_graph = gr.Image()
        gr.Button("Show Mood Trend").click(plot_mood_graph, outputs=mood_graph)

    with gr.Row():
        gr.Markdown("### 📁 Upload Excel (.xlsx) with 'datetime' and 'emotion'")
        excel_file = gr.File(file_types=[".xls", ".xlsx"])
        excel_graph = gr.Image()
        excel_file.change(upload_excel, inputs=excel_file, outputs=excel_graph)

app.launch(share=True)
