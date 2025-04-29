import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from scipy.signal import wiener


def sobel_deblur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    sobel = cv2.magnitude(grad_x, grad_y)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    sobel_color = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    sharpened = cv2.addWeighted(image, 1.0, sobel_color, 0.5, 0)
    return sharpened


def prewitt_deblur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    grad_x = cv2.filter2D(gray, -1, kernel_x)
    grad_y = cv2.filter2D(gray, -1, kernel_y)

    prewitt = cv2.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32))
    prewitt = cv2.normalize(prewitt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    prewitt_color = cv2.cvtColor(prewitt, cv2.COLOR_GRAY2BGR)
    sharpened = cv2.addWeighted(image, 1.0, prewitt_color, 0.5, 0)
    return sharpened


def scharr_deblur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

    scharr = cv2.magnitude(grad_x, grad_y)
    scharr = cv2.normalize(scharr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    scharr_color = cv2.cvtColor(scharr, cv2.COLOR_GRAY2BGR)
    sharpened = cv2.addWeighted(image, 1.0, scharr_color, 0.5, 0)
    return sharpened


def highBoost_filter(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (x, y) = image.shape
    K = (1 / (2 * np.pi)) * np.array([[0.1353, 0.2707, 0.1353],
                                     [0.2707, 0.5413, 0.2707],
                                     [0.1353, 0.2707, 0.1353]])
    K = K / K.sum()
    blurred = np.zeros_like(image, dtype=np.float32)
    padim = np.pad(image, ((1, 1), (1, 1)), mode='edge')
    for row in range(x):
        for col in range(y):
            subim = padim[row:row+3, col:col+3]
            blurred[row, col] = np.sum(subim * K)
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    mask = image.astype(np.int32) - blurred.astype(np.int32)
    k_boost = 1.2
    result = image.astype(np.int32) + k_boost * mask
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def laplacian_filter(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.int32)
    (x, y) = image.shape
    laplacian = np.zeros_like(image, dtype=np.int32)
    padim = np.pad(image, ((1, 1), (1, 1)), mode='edge')
    for row in range(x):
        for col in range(y):
            subim = padim[row:row+3, col:col+3]
            laplacian[row, col] = np.sum(subim * k)
    alpha = 0.5
    enhanced = image.astype(np.float32) - alpha * laplacian.astype(np.float32)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced

def wiener_filter(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_img = wiener(image.astype(np.float32), (5, 5))
    filtered_uint8 = np.clip(filtered_img, 0, 255).astype(np.uint8)
    return filtered_uint8


def apply_filter(image, filter_type):
    if image is None:
        print("No image loaded.")
        return None
    img_copy = image.copy()
    if filter_type == "none":
        return img_copy
    elif filter_type == "High Boost":
        return highBoost_filter(img_copy)
    elif filter_type == "Laplacian":
        return laplacian_filter(img_copy)
    elif filter_type == "Wiener Filter":
        return wiener_filter(img_copy)
    elif filter_type == "Sobel Filter":
        return sobel_deblur(img_copy)
    elif filter_type == "Prewitt Filter":
        return prewitt_deblur(img_copy)
    elif filter_type == "Scharr Filter":
        return scharr_deblur(img_copy)
    return img_copy

def display_image(img, canvas):
    if img is None:
        canvas.delete("all")
        if 'imgtk' in canvas.__dict__: del canvas.imgtk
        return

    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        messagebox.showerror("Display Error", f"Unexpected image shape: {img.shape}")
        return

    canvas_width = 300
    canvas_height = 300
    img_pil = Image.fromarray(img_rgb)
    img_ratio = img_pil.width / img_pil.height
    canvas_ratio = canvas_width / canvas_height

    if img_ratio > canvas_ratio:
        new_width = canvas_width
        new_height = int(new_width / img_ratio)
    else:
        new_height = canvas_height
        new_width = int(new_height * img_ratio)

    new_width = max(1, new_width)
    new_height = max(1, new_height)

    try:
        img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except ValueError as e:
         messagebox.showerror("Resize Error", f"Error resizing image: {e}")
         return

    imgtk = ImageTk.PhotoImage(image=img_pil)
    canvas.delete("all")
    pad_x = (canvas_width - new_width) // 2
    pad_y = (canvas_height - new_height) // 2
    canvas.create_image(pad_x, pad_y, anchor='nw', image=imgtk)
    canvas.imgtk = imgtk

def display_processed_image_final(processed_img):
    """Displays the processed image on the output canvas."""
    display_image(processed_img, output_canvas)
    download_button.pack(pady=(15, 0), anchor='center')

def process_image():
    """Applies the selected filter, shows processing text, then displays result."""
    global original_image, processed_images
    if original_image is None:
        messagebox.showwarning("No Image", "Please upload an image first.")
        return

    selected_filter = filter_var.get()

    output_canvas.delete("all")
    if 'imgtk' in output_canvas.__dict__: del output_canvas.imgtk
    download_button.pack_forget() 
    output_canvas.create_text(150, 150, text="Processing...",
                              font=('Segoe UI', 14, 'italic'), fill=FG_COLOR, anchor='center')
    app.update_idletasks()

    processed_img = None 
    if selected_filter == "none":
        processed_img = original_image.copy()
        processed_images["output"] = processed_img
        app.after(300, lambda: display_processed_image_final(processed_img))
    else:
        processed_img = apply_filter(original_image, selected_filter)
        if processed_img is not None:
            processed_images["output"] = processed_img
            app.after(500, lambda: display_processed_image_final(processed_img))
        else:
            output_canvas.delete("all")
            output_canvas.create_text(150, 150, text="Filter Failed",
                                      font=('Segoe UI', 14, 'bold'), fill='red', anchor='center')
            processed_images["output"] = None 


def upload_image():
    """Handles image upload and resets state."""
    global original_image, processed_images
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif")]
    )
    if file_path:
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Failed to load image file.")
            return
        original_image = img
        processed_images["input"] = original_image.copy()
        display_image(original_image, input_canvas)

        output_canvas.delete("all")
        if 'imgtk' in output_canvas.__dict__: del output_canvas.imgtk
        processed_images["output"] = None
        filter_var.set("none")
        download_button.pack_forget()

def download_image():
    """Opens a save dialog and saves the processed image."""
    if processed_images.get("output") is None:
        messagebox.showwarning("No Image", "No processed image available to download.")
        return

    filter_name = filter_var.get().replace(" ", "_") if filter_var.get() != "none" else "filtered"
    initial_filename = f"processed_{filter_name}.png"

    file_path = filedialog.asksaveasfilename(
        title="Save Processed Image As...",
        initialfile=initial_filename,
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"),
                   ("JPEG files", "*.jpg"),
                   ("BMP files", "*.bmp"),
                   ("TIFF files", "*.tif"),
                   ("All files", "*.*")]
    )

    if file_path:
        try:
            success = cv2.imwrite(file_path, processed_images["output"])
            if success:
                messagebox.showinfo("Success", f"Image saved successfully to:\n{file_path}")
            else:
                messagebox.showerror("Save Error", "Failed to save image. Check file path and extension.")
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving:\n{e}")


app = tk.Tk()
app.title("Image Enhancement App")
app.geometry("950x600")

BG_COLOR = "#1e1e1e"
FG_COLOR = "#eeeeee"
ACCENT_ORANGE = "#FF7700"
ACCENT_ORANGE_ACTIVE = "#FF8C00"
ACCENT_ORANGE_PRESSED = "#E66A00"
BTN_FG = "#FFFFFF"
FRAME_BG = "#2d2d2d"
CANVAS_BG = "#3c3c3c"
SELECT_BG = "#3c3c3c"
SELECT_FG = "#eeeeee"
CANVAS_BORDER = "#4f4f4f"

style = ttk.Style(app)
style.theme_use('clam')
app.configure(bg=BG_COLOR)

style.configure('TFrame', background=FRAME_BG)
style.configure('TLabel', background=FRAME_BG, foreground=FG_COLOR, font=('Segoe UI', 10))
style.configure('Title.TLabel', background=BG_COLOR, foreground=ACCENT_ORANGE, font=('Segoe UI', 18, 'bold'))
style.configure('Header.TLabel', background=FRAME_BG, foreground=ACCENT_ORANGE, font=('Segoe UI', 12, 'bold'))
style.configure('TButton',
                font=('Segoe UI', 11, 'bold'), padding=(15, 10), relief='flat',
                background=ACCENT_ORANGE, foreground=BTN_FG)
style.map('TButton',
          background=[('active', ACCENT_ORANGE_ACTIVE), ('pressed', ACCENT_ORANGE_PRESSED)],
          foreground=[('active', BTN_FG)])
style.configure('TMenubutton',
                font=('Segoe UI', 10), background=SELECT_BG, foreground=SELECT_FG,
                padding=(10, 5), arrowcolor=FG_COLOR, relief='flat')
style.map('TMenubutton',
          background=[('active', '#4f4f4f')],
          arrowcolor=[('active', ACCENT_ORANGE)])
style.configure('Download.TButton',
                font=('Segoe UI', 10, 'bold'), padding=(12, 8), relief='flat',
                background=FG_COLOR, foreground=BG_COLOR)
style.map('Download.TButton',
          background=[('active', '#cccccc'), ('pressed', '#bbbbbb')],
          foreground=[('active', BG_COLOR)])


original_image = None
processed_images = {"input": None, "output": None}

filter_var = tk.StringVar(value="none")

main_app_frame = ttk.Frame(app, padding="20 20 20 20", style='BGColor.TFrame')
style.configure('BGColor.TFrame', background=BG_COLOR)
main_app_frame.pack(fill="both", expand=True)

top_frame = ttk.Frame(main_app_frame, style='BGColor.TFrame')
top_frame.pack(fill="both", expand=True)

control_frame = ttk.Frame(top_frame, padding="15 15", style='TFrame')
control_frame.pack(side="left", fill="y", padx=(0, 20))

image_display_frame = ttk.Frame(top_frame, style='TFrame')
image_display_frame.pack(side="right", fill="both", expand=True)

title_label = ttk.Label(main_app_frame,
                        text="Image Enhancement",
                        style='Title.TLabel',
                        anchor="w")
title_label.pack(pady=(0, 10), padx=(0,0), anchor='nw', before=top_frame)

upload_btn = ttk.Button(control_frame, text="📂 Upload Image", command=upload_image, style='TButton')
upload_btn.pack(fill="x", pady=(10, 30))

filter_label = ttk.Label(control_frame, text="Select Filter:", style='Header.TLabel')
filter_label.pack(anchor="w", pady=(0, 5))

filter_menu = ttk.OptionMenu(control_frame, filter_var, "none", "none", "High Boost", "Laplacian", "Wiener Filter", "Sobel Filter","Prewitt Filter","Scharr Filter", style='TMenubutton')
filter_menu["menu"].config(bg=SELECT_BG, fg=SELECT_FG,
                           activebackground=ACCENT_ORANGE, activeforeground=BTN_FG,
                           font=('Segoe UI', 10), relief='flat')
filter_menu.pack(fill="x", pady=(0, 20))

process_btn = ttk.Button(control_frame, text="🚀 Apply Filter", command=process_image, style='TButton')
process_btn.pack(fill="x", pady=(20, 0))

input_frame = ttk.Frame(image_display_frame, padding="10 10", style='TFrame')
input_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

output_frame = ttk.Frame(image_display_frame, padding="10 10", style='TFrame')
output_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

input_label = ttk.Label(input_frame, text="Original Image", style='Header.TLabel', anchor="center")
input_label.pack(pady=(0, 10), fill='x')
input_canvas = tk.Canvas(input_frame, width=300, height=300, bg=CANVAS_BG, highlightthickness=1, highlightbackground=CANVAS_BORDER)
input_canvas.pack(pady=5)

output_label = ttk.Label(output_frame, text="Processed Image", style='Header.TLabel', anchor="center")
output_label.pack(pady=(0, 10), fill='x')
output_canvas = tk.Canvas(output_frame, width=300, height=300, bg=CANVAS_BG, highlightthickness=1, highlightbackground=CANVAS_BORDER)
output_canvas.pack(pady=5)

download_frame = ttk.Frame(main_app_frame, style='BGColor.TFrame')
download_frame.pack(fill="x", pady=(10, 0))

download_button = ttk.Button(download_frame,
                             text="⬇️ Download Image",
                             command=download_image,
                             style='TButton')


app.update_idletasks()

app.mainloop()