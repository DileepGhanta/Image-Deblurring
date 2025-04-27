import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, font as tkFont
from PIL import Image, ImageTk
from scipy.signal import wiener

# --- Original Image Processing Functions (Unchanged) ---
def highBoost_filter(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (x, y) = image.shape
    K = (1 / 4.8976) * np.array([[0.3679, 0.6065, 0.3679],
                                  [0.6065, 1.0000, 0.6065],
                                  [0.3679, 0.6065, 0.3679]])
    blurred = np.zeros_like(image, dtype=np.float32)
    padim = np.pad(image, ((1, 1), (1, 1)), mode='edge')
    for row in range(x):
        for col in range(y):
            subim = padim[row:row+3, col:col+3]
            blurred[row, col] = np.sum(subim * K)
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    coin = image.astype(np.int32) - blurred.astype(np.int32)
    result = np.clip(image.astype(np.int32) + 5*(coin), 0, 255).astype(np.uint8)
    return result

def laplacian_filter(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.int32)
    (x, y) = image.shape
    laplacian = np.zeros_like(image, dtype=np.int32)
    padim = np.pad(image, ((1, 1), (1, 1)), mode='edge')
    for row in range(x):
        for col in range(y):
            subim = padim[row:row+3, col:col+3].astype(np.int32)
            laplacian[row, col] = np.sum(subim * k)
    laplacian = np.clip(laplacian, -255, 255)
    alpha = 0.3
    if image.dtype == 'uint8':
         deblurred_intermediate = image.astype(np.int32) - alpha * laplacian
    else:
         deblurred_intermediate = image - alpha * laplacian
    print(" Data type of deblurred intermediate" ,deblurred_intermediate.dtype)
    deblurred = np.clip(deblurred_intermediate, 0, 255).astype(np.uint8)
    return deblurred

def wiener_filter(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = 0
    stddev = 10
    noise = np.random.normal(mean, stddev, image.shape)
    noisy_img = image.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255)
    filtered_img = wiener(noisy_img, (7, 7))
    filtered_uint8 = np.clip(filtered_img, 0, 255).astype(np.uint8)
    return filtered_uint8

def apply_filter(image_in, filter_type):
    if image_in is None: return None
    img_copy = image_in.copy()
    if filter_type == "none":
        return img_copy
    elif filter_type == "High Boost":
        return highBoost_filter(img_copy)
    elif filter_type == "Laplacian":
        return laplacian_filter(img_copy)
    elif filter_type == "Wiener Filter":
        return wiener_filter(img_copy)
    return img_copy

# --- GUI Functions ---
def process_image():
    global image
    if image is None:
        print("Please upload an image first.")
        # Optionally show a message box: tk.messagebox.showwarning("No Image", "Please upload an image first.")
        return
    img_to_process = image.copy()
    selected_filter = filter_var.get()
    processed_result = apply_filter(img_to_process, selected_filter)
    if processed_result is not None:
        display_image(processed_result, output_canvas)

def upload_image():
    global image
    file_path = filedialog.askopenfilename(
         title="Select Image",
         filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    if file_path:
        img_read = cv2.imread(file_path)
        if img_read is None:
            print(f"Error: Could not read image file {file_path}")
            image = None
            input_canvas.delete("all")
            input_canvas.imgtk = None
            output_canvas.delete("all")
            output_canvas.create_text(150, 150, text="Could not load image", fill="red", font=default_font)
            output_canvas.imgtk = None
        else:
            image = img_read
            display_image(image, input_canvas)
            output_canvas.delete("all")
            # Added placeholder text here as well
            output_canvas.create_text(150, 150, fill=FG_COLOR, font=default_font)
            output_canvas.imgtk = None
            filter_var.set("none")

def display_image(img, canvas):
    # Original display logic with fixed 300x300 resize
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("Warning: Unexpected image shape for display.")
        return
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((300, 300), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    canvas.imgtk = imgtk
    canvas.delete("all")
    canvas.create_image(0, 0, anchor='nw', image=imgtk)

# --- GUI Setup ---

# --- New Color Palette: Dark Theme with ORANGE Accent ---
BG_COLOR = "#1A1A1A"       # Very Dark Grey Background
FG_COLOR = "#ECF0F1"       # Light Grey/Off-White Text
ACCENT_COLOR = "#FFA500"   # Bright Orange Accent
ACCENT_DARK = "#E69500"    # Darker Orange for Hover/Active
COMPONENT_BG = "#2C3E50"   # Dark Slate Blue/Grey (for dropdown)
CANVAS_BG = "#212121"      # Near-Black for canvas background

app = tk.Tk()
app.title("✨ Image Deblurring App✨")
app.geometry("900x550")
app.configure(bg=BG_COLOR)
app.resizable(False, False)

# --- Font Definitions ---
try:
    default_font = tkFont.nametofont("TkDefaultFont")
    default_font.configure(family="Segoe UI", size=10)
    title_font = tkFont.Font(family="Segoe UI", size=16, weight="bold")
    label_font = tkFont.Font(family="Segoe UI", size=11, weight="bold")
    button_font = tkFont.Font(family="Segoe UI", size=11, weight="bold")
except: # Fallback font
    default_font = tkFont.nametofont("TkDefaultFont")
    default_font.configure(family="Helvetica", size=10)
    title_font = tkFont.Font(family="Helvetica", size=16, weight="bold")
    label_font = tkFont.Font(family="Helvetica", size=11, weight="bold")
    button_font = tkFont.Font(family="Helvetica", size=11, weight="bold")
app.option_add("*Font", default_font)

# --- Style configuration for ttk widgets (Dropdown) ---
style = ttk.Style()
style.theme_use('clam')
style.configure('.', background=BG_COLOR, foreground=FG_COLOR)
style.configure('TFrame', background=BG_COLOR)
style.configure('TLabel', background=BG_COLOR, foreground=FG_COLOR, font=default_font)
style.configure('TMenubutton',
                font=default_font,
                background=COMPONENT_BG, # Keep dropdown distinct
                foreground=FG_COLOR,
                arrowcolor=FG_COLOR,
                bordercolor=ACCENT_COLOR, # Use new accent for border
                relief=tk.FLAT,
                padding=(10, 6))
style.map('TMenubutton',
          background=[('active', ACCENT_DARK)], # Use new accent hover
          foreground=[('active', 'black')]) # Black text on orange active

# --- Global Variables ---
image = None
processed_images = {}
filter_var = tk.StringVar(value="none")

# --- Main container ---
main_frame = ttk.Frame(app, padding="20 20 20 20")
main_frame.pack(fill="both", expand=True)

# --- Control panel ---
control_frame = ttk.Frame(main_frame, padding="15 15")
control_frame.pack(side="left", fill="y", padx=(0, 25))

# Title
title_label = ttk.Label(control_frame,
                        text="Image Deblurring",
                        font=title_font,
                        foreground=ACCENT_COLOR) # Orange Title
# Reduced pady below title for tighter alignment
title_label.pack(pady=(0, 15), anchor="center")

# --- Button Hover Effects ---
def on_enter(e):
    e.widget['background'] = ACCENT_DARK # Darker Orange

def on_leave(e):
    e.widget['background'] = ACCENT_COLOR # Bright Orange

# --- Buttons ---
common_button_options = {
    "bg": ACCENT_COLOR,       # Bright Orange
    "fg": "black",            # Black text for better contrast on orange
    "font": button_font,
    "relief": tk.FLAT,
    "borderwidth": 0,
    "width": 20,
    "padx": 10,
    "pady": 8,
    "activebackground": ACCENT_DARK, # Darker Orange
    "activeforeground": "black"      # Keep text black when active
}

upload_btn = tk.Button(control_frame, text="📂 Upload Image", command=upload_image, **common_button_options)
upload_btn.pack(fill="x", pady=(10, 20)) # Added some padding above
upload_btn.bind("<Enter>", on_enter)
upload_btn.bind("<Leave>", on_leave)

# Filter selection
filter_label = ttk.Label(control_frame, text="Select Filter:", font=label_font)
filter_label.pack(anchor="w", pady=(10, 5))
filter_menu = ttk.OptionMenu(control_frame, filter_var, "none", "none", "High Boost", "Laplacian", "Wiener Filter", style='TMenubutton')
filter_menu.pack(fill="x", pady=(0, 20))

process_btn = tk.Button(control_frame, text="🚀 Apply Filter", command=process_image, **common_button_options)
process_btn.pack(fill="x", pady=(15, 0))
process_btn.bind("<Enter>", on_enter)
process_btn.bind("<Leave>", on_leave)

# --- Image display frame ---
image_frame = tk.Frame(main_frame, bg=BG_COLOR)
image_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
image_frame.grid_columnconfigure(0, weight=1)
image_frame.grid_columnconfigure(1, weight=1)
image_frame.grid_rowconfigure(0, weight=0)
image_frame.grid_rowconfigure(1, weight=1)

# --- Input Image Area ---
input_label_frame = tk.Frame(image_frame, bg=BG_COLOR)
input_label_frame.grid(row=0, column=0, pady=(0, 5), sticky="ew")
input_label = tk.Label(input_label_frame, text="Original Image", font=label_font, bg=BG_COLOR, fg=FG_COLOR)
input_label.pack()

input_canvas = tk.Canvas(image_frame, width=300, height=300,
                         bg=CANVAS_BG,          # Near-black background
                         highlightthickness=1,
                         highlightbackground=ACCENT_COLOR) # Orange border
input_canvas.grid(row=1, column=0, padx=10, pady=(0, 10))

# --- Output Image Area ---
output_label_frame = tk.Frame(image_frame, bg=BG_COLOR)
output_label_frame.grid(row=0, column=1, pady=(0, 5), sticky="ew")
output_label = tk.Label(output_label_frame, text="Processed Image", font=label_font, bg=BG_COLOR, fg=FG_COLOR)
output_label.pack()

output_canvas = tk.Canvas(image_frame, width=300, height=300,
                          bg=CANVAS_BG,         # Near-black background
                          highlightthickness=1,
                          highlightbackground=ACCENT_COLOR) # Orange border
output_canvas.grid(row=1, column=1, padx=10, pady=(0, 10))

# Initial placeholder text
output_canvas.create_text(150, 150, fill=FG_COLOR, font=default_font)

app.mainloop()