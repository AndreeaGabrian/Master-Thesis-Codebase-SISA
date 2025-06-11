import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn.functional as F
from architecture.model import build_model
from utils.utils import get_transform

# ----- CONFIG -----
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
NUM_SHARDS = 5
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_DESCRIPTIONS = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}


# ----- LOAD MODELS -----
def load_sisa_models():
    models = []
    for k in range(NUM_SHARDS):
        model = build_model("resnet18", NUM_CLASSES, pretrained=True).to(DEVICE)
        ckpt = f"UI_models/shard_{k}/slice_2.pt"
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        models.append(model)
    return models


# ----- INFERENCE -----
def run_inference(img_path, models, transform):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    shard_probs = []
    shard_preds = []

    for model in models:
        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
            shard_probs.append(probs)
            shard_preds.append(probs.argmax().item())

    # Aggregation
    avg_probs = torch.stack(shard_probs).mean(dim=0)
    softmax_pred = avg_probs.argmax().item()
    majority_pred = torch.mode(torch.tensor(shard_preds)).values.item()
    majority_voters = [p for p in shard_probs if p.argmax().item() == majority_pred]
    majority_probs = torch.stack(majority_voters).mean(dim=0)
    confidence_idx = torch.stack([p.max(0).values for p in shard_probs]).argmax()
    confidence_pred = shard_probs[confidence_idx].argmax().item()
    confidence_probs = shard_probs[confidence_idx]

    return shard_probs, {
        "softmax": (softmax_pred, avg_probs[softmax_pred].item()),
        "majority": (majority_pred, majority_probs[majority_pred]),
        "confidence-max": (confidence_pred, confidence_probs[confidence_pred])
    }


# ----- GUI -----
def launch_gui():
    models = load_sisa_models()
    transform = get_transform()

    root = tk.Tk()
    root.title("SISA++ Skin carcinoma classifier")
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    root.resizable(False, False)

    top_frame = tk.Frame(root, height=350)
    top_frame.pack(fill="x")

    title_label = tk.Label(top_frame, text="SISA++ Image Classification\nExample: HAM10000 Dataset",
                           font=("Helvetica", 14, "bold"), pady=10)
    title_label.pack()

    bottom_frame = tk.Frame(root)
    bottom_frame.pack(fill="both", expand=True)

    left_frame = tk.Frame(bottom_frame, width=WINDOW_WIDTH//2)
    left_frame.pack(side="left", fill="both", expand=True)

    right_frame = tk.Frame(bottom_frame, width=WINDOW_WIDTH//2)
    right_frame.pack(side="right", fill="both", expand=True)

    # ----- Image & Upload Button -----
    image_label = tk.Label(top_frame)
    image_label.pack(pady=10)

    def on_upload():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        # display image
        img = Image.open(file_path).resize((200, 200))
        img_display = ImageTk.PhotoImage(img)
        image_label.configure(image=img_display)
        image_label.image = img_display

        # SISA inference
        shard_probs, preds = run_inference(file_path, models, transform)

        # text output shard scores
        text = "Shards/models outputs\n"
        for i, probs in enumerate(shard_probs):
            prob_str = ", ".join([f"{p:.2f}" for p in probs.cpu().tolist()])
            text += f"Shard {i}: [{prob_str}]\n"

        # text output aggregated predictions
        text += "\nAggregated Predictions:\n"
        for strat, (label, score) in preds.items():
            class_name = CLASS_NAMES[label]
            text += f"{strat}: class {label} ({class_name}) — probability score: {score:.2f}\n"

        prediction_label.config(text=text)

        legend_text = "Class Legend:\n"
        for i, abbr in enumerate(CLASS_NAMES):
            legend_text += f"{abbr}: {CLASS_DESCRIPTIONS[abbr]}\n"

        if hasattr(root, "legend_label"):
            root.legend_label.config(text=legend_text)
        else:
            root.legend_label = tk.Label(left_frame, text=legend_text, justify="left", font=("Courier", 9))
            root.legend_label.pack(padx=10, pady=(0, 10), anchor="w")

        avg_probs = torch.stack(shard_probs).mean(dim=0).cpu().tolist()

        # Majority vote
        shard_preds = [p.argmax().item() for p in shard_probs]
        majority_class = torch.mode(torch.tensor(shard_preds)).values.item()
        majority_voters = [p for p in shard_probs if p.argmax().item() == majority_class]
        majority_probs = torch.stack(majority_voters).mean(dim=0).cpu().tolist()

        # Confidence-max
        confidences = torch.stack([p.max(0).values for p in shard_probs])
        best_idx = torch.argmax(confidences)
        confidence_probs = shard_probs[best_idx].cpu().tolist()

        # plot results
        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        x = list(range(NUM_CLASSES))
        bar_width = 0.25

        ax.bar([i - bar_width for i in x], avg_probs, width=bar_width, label="Softmax", color="skyblue")
        ax.bar(x, majority_probs, width=bar_width, label="Majority", color="salmon")
        ax.bar([i + bar_width for i in x], confidence_probs, width=bar_width, label="Conf-Max", color="lightgreen")

        ax.set_title("Class Probabilities by Aggregation Strategy")
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1)
        ax.legend()

        # show plot in GUI
        for widget in right_frame.winfo_children():
            widget.destroy()  # clear previous canvas

        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    upload_button = tk.Button(top_frame, text="Upload Image", command=on_upload)
    upload_button.pack()

    prediction_label = tk.Label(left_frame, text="", justify="left", anchor="nw", font=("Courier", 10))
    prediction_label.pack(padx=10, pady=10, fill="both", expand=True)

    root.mainloop()


launch_gui()


