import datetime
import json
import os
import subprocess
from diffusers import StableDiffusionPipeline
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline,
)
import whisper

# 🎙️ Whisper Model Selection
print("Pick your Whisper model:")
print("1. tiny\n2. small\n3. medium\n4. large\n5. turbo")

model_mapping = {
    "1": "tiny",
    "tiny": "tiny",
    "2": "small",
    "small": "small",
    "3": "medium",
    "medium": "medium",
    "4": "large",
    "large": "large",
    "5": "turbo",
    "turbo": "turbo",
}

user_pick = input("Enter choice (1-5 or name, default 'small'): ").strip().lower()
whisper_model_name = model_mapping.get(user_pick, "small")
print(f"Loading Whisper model: {whisper_model_name}...")
whisper_model = whisper.load_model(whisper_model_name)

# ❤️ Emotion detection model
emotion_model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(
    emotion_model_name
)
emotion_pipeline = pipeline(
    "text-classification",
    model=emotion_model,
    tokenizer=emotion_tokenizer,
    top_k=1,
)

# ✍️ Summarizer (T5)
summarizer_tokenizer = T5Tokenizer.from_pretrained("t5-small")
summarizer_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# 🎨 Text-to-Image Generator (CPU / GPU fallback handling)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipe_sd = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=dtype
)
pipe_sd = pipe_sd.to(device)


# 😊 Emoji board mapping
def emoji_mood_board(emotion_label):
    board = {
        "joy": ["😊", "😄", "✨", "🌞", "🎉"],
        "sadness": ["😢", "💧", "💔", "😞", "☁"],
        "anger": ["😠", "🔥", "💢", "😤", "⚡"],
        "fear": ["😨", "😱", "🫣", "😰", "🚨"],
        "confusion": ["😕", "❓", "🤯", "🌀", "🤷"],
        "surprise": ["😲", "🤯", "😯", "✨", "🎊"],
        "love": ["❤️", "😍", "💖", "💕", "💘"],
        "neutral": ["😐", "💬", "📖", "📝", "🔍"],
    }
    return board.get(emotion_label.lower(), ["🤔", "🫥", "🧠", "💭", "📘"])


# 🖼️ Image Generation & Video Tracking
def generate_image(summary):
    image = pipe_sd(summary).images[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"summary_{timestamp}.png"
    image.save(image_path)

    # Track image path for compiling video later
    with open("image_list.txt", "a") as f:
        f.write(image_path + "\n")

    return image_path


# 🎥 Stitch Images into Video using FFmpeg
def create_video_from_images():
    if not os.path.exists("image_list.txt"):
        return None

    with open("image_list.txt", "r") as f:
        images = [line.strip() for line in f.readlines() if os.path.exists(line)]

    if not images:
        return None

    with open("ffmpeg_input.txt", "w") as f:
        for img in images:
            f.write(f"file '{img}'\n")
            f.write("duration 1\n")

    output_video = "summary_video.mp4"
    subprocess.run([
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "ffmpeg_input.txt",
        "-vf",
        "fps=1,format=yuv420p",
        "-pix_fmt",
        "yuv420p",
        output_video,
    ])

    return output_video if os.path.exists(output_video) else None


# 🔍 Emotion Detection
def detect_emotion(text):
    results = emotion_pipeline(text)
    top = results[0][0]
    return top["label"], round(top["score"], 2)


# ✂️ Text Summarization
def summarize_text(text, max_length=60):
    input_text = "summarize: " + text
    input_ids = summarizer_tokenizer.encode(
        input_text, return_tensors="pt", truncation=True
    )
    summary_ids = summarizer_model.generate(
        input_ids,
        max_length=max_length,
        min_length=15,
        num_beams=4,
        length_penalty=2.0,
    )
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# 💾 Diary JSON Storage
def load_entries():
    if os.path.exists("diary.json"):
        with open("diary.json", "r") as f:
            return json.load(f)
    return []


def save_entries(entries):
    with open("diary.json", "w") as f:
        json.dump(entries, f, indent=2)


# 🎛️ Audio Processing Pipeline
def process_audio(audio_path):
    if not audio_path:
        return (
            "No audio provided.",
            "Unknown",
            "",
            "",
            None,
        )

    # Using transcribe handles arbitrary audio duration cleanly
    result = whisper_model.transcribe(audio_path)
    transcript = result.get("text", "").strip()

    if not transcript:
        return (
            "No speech detected.",
            "None",
            "",
            "",
            None,
        )

    emotion, confidence = detect_emotion(transcript)
    summary = summarize_text(transcript)
    emojis = " ".join(emoji_mood_board(emotion))

    image_path = generate_image(summary)

    entry = {
        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": transcript,
        "summary": summary,
        "emotion": emotion,
        "confidence": confidence,
        "emojis": emojis,
        "image_path": image_path,
    }

    entries = load_entries()
    entries.append(entry)
    save_entries(entries)

    return (
        transcript,
        f"{emotion} ({confidence})",
        summary,
        emojis,
        image_path,
    )


# 📈 Plot Mood Graph (Last 7 Days)
def plot_mood_graph():
    entries = load_entries()
    if not entries:
        return None

    cutoff = datetime.datetime.now() - datetime.timedelta(days=7)
    recent = [
        e
        for e in entries
        if datetime.datetime.strptime(e["datetime"], "%Y-%m-%d %H:%M:%S")
        >= cutoff
    ]
    if not recent:
        return None

    mood_map = {
        "joy": 5,
        "love": 5,
        "surprise": 4,
        "neutral": 3,
        "confusion": 2,
        "fear": 2,
        "sadness": 1,
        "anger": 0,
    }
    times = [e["datetime"] for e in recent]
    scores = [mood_map.get(e["emotion"].lower(), 3) for e in recent]

    plt.figure(figsize=(10, 5))
    plt.plot(times, scores, marker="o", color="blue")
    plt.xticks(rotation=45)
    plt.yticks(list(mood_map.values()), list(mood_map.keys()))
    plt.title("Mood Trend - Last 7 Days")
    plt.tight_layout()

    path = "mood_graph.png"
    plt.savefig(path)
    plt.close()
    return path


# 📁 Excel Upload Plot
def upload_excel(file):
    if file is None:
        return None
    df = pd.read_excel(file.name)
    if "datetime" not in df.columns or "emotion" not in df.columns:
        return None

    mood_map = {
        "joy": 5,
        "love": 5,
        "surprise": 4,
        "neutral": 3,
        "confusion": 2,
        "fear": 2,
        "sadness": 1,
        "anger": 0,
    }
    datetimes = df["datetime"].astype(str).tolist()
    scores = [mood_map.get(str(e).lower(), 3) for e in df["emotion"]]

    plt.figure(figsize=(10, 5))
    plt.plot(datetimes, scores, marker="o", color="orange")
    plt.xticks(rotation=45)
    plt.yticks(list(mood_map.values()), list(mood_map.keys()))
    plt.title("Mood Graph from Excel")
    plt.tight_layout()

    path = "excel_mood_graph.png"
    plt.savefig(path)
    plt.close()
    return path


# 🎨 Gradio User Interface
with gr.Blocks(title="AI Voice Diary") as app:
    gr.Markdown("# 🎙️ Voice Diary + Emotion + Summary + Image")

    with gr.Row():
        audio_input = gr.Audio(
            type="filepath", label="🎤 Upload Audio (.mp3/.wav/.mp4)"
        )

    with gr.Row():
        transcript = gr.Textbox(label="📝 English Transcript", lines=3)

    with gr.Row():
        emotion = gr.Textbox(label="😊 Detected Emotion")
        summary = gr.Textbox(label="✂️ Summary")
        emojis = gr.Textbox(label="🎭 Emoji Mood Board")

    with gr.Row():
        image_output = gr.Image(label="🎨 AI Image from Summary")

    with gr.Row():
        submit = gr.Button("Process Audio", variant="primary")
        submit.click(
            process_audio,
            inputs=audio_input,
            outputs=[
                transcript,
                emotion,
                summary,
                emojis,
                image_output,
            ],
        )

    gr.Markdown("---")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📈 Mood Graph (Last 7 Days)")
            mood_graph = gr.Image()
            btn_mood = gr.Button("Show Mood Trend")
            btn_mood.click(plot_mood_graph, outputs=mood_graph)

        with gr.Column():
            gr.Markdown("### 🎥 Diary Video Montage")
            video_output = gr.Video()
            btn_video = gr.Button("Create Video from Images")
            btn_video.click(create_video_from_images, outputs=video_output)

    gr.Markdown("---")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Import Excel Log (.xlsx)")
            excel_file = gr.File(
                file_types=[".xlsx", ".xls"], label="Upload Excel Diary"
            )
            excel_graph = gr.Image(label="Excel Mood Graph")
            btn_excel = gr.Button("Plot from Excel")
            btn_excel.click(upload_excel, inputs=excel_file, outputs=excel_graph)

if __name__ == "__main__":
    app.launch(share=True)
