import cv2
import time
from ultralytics import YOLO
from captioning.blip_caption import generate_caption
from reasoning.llm_reasoning import ask_llm

model = YOLO('yolov8n.pt')

def wrap_text(text, max_chars=40):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        if len(current) + len(word) < max_chars:
            current += word + " "
        else:
            lines.append(current)
            current = word + " "

    lines.append(current)
    return lines

def detect_objects(frame):
    results = model(frame)

    detected = set()
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detected.add(label)

    return list(detected)

def main():
    cap = cv2.VideoCapture(0)

    frame_count = 0
    caption = ""
    answer = ""
    user_text = ""
    prev_time = 0

    last_results = None
    last_objects = []

    yolo_time = 0
    caption_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------- YOLO (every 2 frames) --------
        if frame_count % 2 == 0 or last_results is None:
            start = time.time()
            last_results = model(frame)
            yolo_time = time.time() - start

            # extract objects ONLY when YOLO runs
            last_objects = detect_objects(frame)

        results = last_results
        detected_objects = last_objects

        annotated_frame = results[0].plot()

        # -------- CAPTION (every 30 frames) --------
        if frame_count % 30 == 0:
            start = time.time()
            new_caption = generate_caption(frame)
            caption_time = time.time() - start

            if new_caption:
                caption = new_caption

        if not caption:
            caption = "No caption available"

        frame_count += 1

        # -------- FPS --------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        # -------- INPUT --------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 8:
            user_text = user_text[:-1]
        elif key == 13:
            if user_text.strip():
                context = f"Objects: {', '.join(detected_objects)}. Caption: {caption}."

                answer = ask_llm(
                    caption,
                    detected_objects,
                    f"{user_text}. Context: {context}"
                )

                print("Q:", user_text)
                print("A:", answer)
                user_text = ""
        elif 32 <= key <= 126:
            user_text += chr(key)

        # -------- UI --------
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (5, 5), (640, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

        cv2.putText(annotated_frame,
                    "Objects: " + ", ".join(detected_objects),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

        cv2.putText(annotated_frame,
                    "Caption: " + caption,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,0,0),
                    2)

        cv2.putText(annotated_frame,
                    "Ask: " + user_text,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2)

        if answer:
            lines = wrap_text(answer, 45)
            y = 120
            for line in lines:
                cv2.putText(annotated_frame,
                            "AI: " + line,
                            (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0,255,255),
                            2)
                y += 25

        cv2.putText(annotated_frame,
                    f"FPS: {int(fps)}",
                    (520, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    1)

        cv2.putText(annotated_frame,
                    f"YOLO: {yolo_time:.2f}s | BLIP: {caption_time:.2f}s",
                    (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    1)

        cv2.putText(annotated_frame,
                    "Type question + ENTER | Q to quit",
                    (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200,200,200),
                    1)

        cv2.imshow("AI Vision Copilot", annotated_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()