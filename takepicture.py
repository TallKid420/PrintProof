import cv2
import os
import tkinter as tk
from datetime import datetime
from PIL import Image, ImageTk


def _save_frame(frame, save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"photo_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Picture saved to {filename}")
    return filename


def _capture_with_tk_preview(cap):
    """Fallback preview using Tkinter when cv2.imshow is unavailable."""
    captured = {"frame": None, "done": False}

    root = tk.Tk()
    root.title("Camera Preview")

    label = tk.Label(root)
    label.pack()

    info = tk.Label(root, text="Press Enter to capture, Esc/q to cancel")
    info.pack()

    def on_enter(_event=None):
        captured["done"] = True

    def on_cancel(_event=None):
        captured["frame"] = None
        captured["done"] = True

    root.bind_all("<Return>", on_enter)
    root.bind_all("<KP_Enter>", on_enter)
    root.bind_all("<Escape>", on_cancel)
    root.bind_all("q", on_cancel)
    root.protocol("WM_DELETE_WINDOW", on_cancel)

    # Ensure keyboard focus is restored for each new preview window.
    root.lift()
    root.attributes("-topmost", True)
    root.after(100, lambda: root.attributes("-topmost", False))
    root.focus_force()

    def update_frame():
        if captured["done"]:
            root.destroy()
            return

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read camera frame")
            captured["frame"] = None
            captured["done"] = True
            root.destroy()
            return

        captured["frame"] = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=image)
        label.configure(image=photo)
        label.image = photo
        root.after(15, update_frame)

    update_frame()
    root.mainloop()

    return captured["frame"]


def take_picture(save_dir="."):
    """Show camera preview, then capture on Enter and save locally."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    captured_frame = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read camera frame")
                break

            cv2.imshow("Camera Preview", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:
                captured_frame = frame
                break
            if key in (27, ord("q")):
                print("Capture cancelled")
                break

        cv2.destroyAllWindows()
    except cv2.error:
        print("OpenCV GUI preview is unavailable. Falling back to Tkinter preview.")
        captured_frame = _capture_with_tk_preview(cap)
        if captured_frame is None:
            print("Capture cancelled")

    cap.release()

    if captured_frame is not None:
        return _save_frame(captured_frame, save_dir)

    return None

if __name__ == "__main__":
    take_picture()