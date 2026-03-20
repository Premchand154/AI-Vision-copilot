# AI Vision Copilot

**Real-time Multimodal AI System for Visual Understanding & Reasoning**

AI Vision Copilot combines computer vision and large language models to **detect, describe, and reason about visual scenes** in real time.

Upload an image or use your camera → ask questions → get intelligent answers.

---

## Features

- **Object Detection** — YOLOv8 (Ultralytics)
- **Image Captioning** — BLIP (HuggingFace)
- **LLM Reasoning** — Mistral / Phi3 (Ollama)
- **Conversation Memory** — Context-aware responses
- **Live Camera + Image Upload**
- **Optimized Inference Pipeline**
- **Docker Deployment**

---

## How It Works

```

Image Input
↓
YOLOv8 → Object Detection
↓
BLIP → Caption Generation
↓
Memory → Context Tracking
↓
LLM (Ollama) → Reasoning & Answer

````

---

## Performance Optimizations

- YOLO runs every 2 frames for efficiency  
- Captioning runs every 30 frames  
- Reduced prompt size for faster LLM inference  
- Timeout handling + logging for reliability  

---

## Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| Computer Vision | OpenCV, YOLOv8 |
| Captioning | BLIP (Transformers) |
| LLM | Ollama (Mistral / Phi3) |
| UI | Streamlit |
| Deployment | Docker, Docker Compose |

---

## Getting Started

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Vision-Copilot.git
cd AI-Vision-Copilot
````

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Run LLM (Ollama)

```bash
ollama run phi3
```

> ⚡ Recommended: `phi3` for faster inference

---

### Run Application

```bash
streamlit run app.py
```

---

## Docker Setup

```bash
docker-compose up --build
```

### Pull Model

```bash
docker exec -it ollama_server ollama pull phi3
```

---

## Usage

### Upload Image Mode

* Upload image
* View detected objects & caption
* Ask questions

### Live Camera Mode

* Real-time scene understanding
* Interactive questioning

---

## Example

**Input:**
Image of a dog playing with a ball

**Output:**

* Objects: dog, sports ball
* Caption: a dog running with a ball
* Answer: The dog is playing and running with a ball

---

## Project Structure

```
AI-Vision-Copilot/
│
├── app.py
├── realtime_detection.py
│
├── captioning/
│   └── blip_caption.py
│
├── reasoning/
│   └── llm_reasoning.py
│
├── utils/
│   ├── memory.py
│   └── logger.py
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🌍 Real-World Applications

* Assistive AI for visually impaired
* Smart surveillance systems
* Robotics & autonomous agents
* AI-powered visual assistants




