import requests
from utils.memory import memory
from utils.logger import log_event

OLLAMA_URL = "http://ollama:11434/api/generate" 
MODEL = "mistral"

def ask_llm(caption, objects, question):

    objects_text = ", ".join(objects) if objects else "No objects detected"

    context = f"""
Image Caption: {caption}
Detected Objects: {objects_text}
"""

    history = memory.get_context()

    prompt = f"""
You are an AI vision assistant.

RULES:
- Use current visual context as primary source
- Use conversation history for follow-up questions
- Do NOT hallucinate unseen objects
- If unsure, say "I cannot determine from the image"
- Answer concisely (1-2 sentences)

Conversation History:
{history}

Current Context:
{context}

Question: {question}

Answer:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3
            },
            timeout=120
        )

        if response.status_code == 200:
            answer = response.json().get("response", "").strip()

            if not answer:
                answer = "I couldn't generate a reliable answer."

            memory.add(question, answer)

            log_event(f"SUCCESS | Q: {question} | A: {answer[:100]}")

            return answer

        else:
            log_event(f"FAIL | Q: {question} | Status: {response.status_code}")
            return "LLM request failed"

    except Exception as e:
        log_event(f"ERROR | Q: {question} | {str(e)}")
        print("ERROR:", str(e))
        return f"Error: {str(e)}"