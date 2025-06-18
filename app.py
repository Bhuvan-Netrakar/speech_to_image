# 🧠 Import libraries
import whisper
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import gradio as gr
import json
import subprocess
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline

print("pick your model")
print("1.tiny")
print("2.small")
print("3.medium")
print("4.large")
print("5.turbo")

whisper_model_pick = input("pick the model")

# 🎙 Load Whisper model
whisper_model = whisper.load_model(whisper_model_pick) 

# ❤ Emotion detection model
emotion_model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
emotion_pipeline = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, top_k=1)

# ✍ Summarizer (T5)
summarizer_tokenizer = T5Tokenizer.from_pretrained("t5-small")
summarizer_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# 🌐 Translation Pipelines
translation_en_hi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
translation_en_kn = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mul")

# 🎨 Text-to-Image Generator
pipe_sd = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe_sd = pipe_sd.to("cuda" if torch.cuda.is_available() else "cpu")

# 😊 Emoji board mapping
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


def generate_image(summary):
    image = pipe_sd(summary).images[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"summary_{timestamp}.png"
    image.save(image_path)

    # Track image for video later
    with open("image_list.txt", "a") as f:
        f.write(image_path + "\n")

    return image_path


def create_video_from_images():
    if not os.path.exists("image_list.txt"):
        return None

    # Create video from image list
    with open("image_list.txt", "r") as f:
        images = [line.strip() for line in f.readlines() if os.path.exists(line)]

    if not images:
        return None

    # Prepare input list for ffmpeg
    with open("ffmpeg_input.txt", "w") as f:
        for img in images:
            f.write(f"file '{img}'\n")
            f.write("duration 1\n")  # 1 sec per image

    output_video = "summary_video.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "ffmpeg_input.txt",
        "-vf", "fps=1,format=yuv420p", "-pix_fmt", "yuv420p", output_video
    ])

    return output_video if os.path.exists(output_video) else None


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
        gr.Markdown("### 🎥 Generate Diary Video from AI Images")
        video_output = gr.Video()
        gr.Button("Create Video").click(create_video_from_images, outputs=video_output)


app.launch(share=True)
