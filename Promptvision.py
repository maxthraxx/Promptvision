import customtkinter as ctk
import os
import pandas as pd
import json
from PIL import ImageTk, Image
from typing import List, Dict
import shutil
from tkinter import filedialog
import tkinter.messagebox as messagebox
import queue
import sys
import logging
from functools import lru_cache
from views.gallery_view import GalleryView
from concurrent.futures import ThreadPoolExecutor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from functools import wraps
import time
from views.metadata_view import MetadataView

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s'
)
def debounce(wait):
    def decorator(fn):
        last_called = None
        @wraps(fn)
        def debounced(*args, **kwargs):
            nonlocal last_called
            if last_called is None or time.time() - last_called >= wait:
                last_called = time.time()
                return fn(*args, **kwargs)
        return debounced
    return decorator

class ImageChangeHandler(FileSystemEventHandler):
    def __init__(self, callback, image_files):
        self.callback = callback
        self.known_files = set(os.path.normpath(f.lower()) for f in image_files)
        self.last_modified = {}
        self.tmp_files = set()
        logging.debug(f"ImageChangeHandler initialized with {len(self.known_files)} known files")

    def on_created(self, event):
        self.handle_event(event)

    def on_modified(self, event):
        self.handle_event(event)

    def handle_event(self, event):
        logging.debug(f"Handling event: {event.event_type} - {event.src_path}")
        if not event.is_directory:
            file_path = os.path.normpath(event.src_path.lower())
            
            if file_path.endswith(".tmp"):
                self.tmp_files.add(file_path)
                threading.Timer(1.0, self.check_for_image, args=(file_path,)).start()
            elif file_path.endswith((".jpg", ".jpeg", ".png", ".gif")):
                self.process_image_file(file_path)

    def check_for_image(self, tmp_path):
        base_path, _ = os.path.splitext(tmp_path)
        for ext in [".jpg", ".jpeg", ".png", ".gif"]:
            image_path = base_path + ext
            if os.path.exists(image_path):
                self.process_image_file(image_path)
                break
        self.tmp_files.discard(tmp_path)

    def process_image_file(self, file_path):
        logging.debug(f"Processing image file: {file_path}")
        if file_path not in self.known_files:
            logging.info(f"New image detected: {file_path}")
            self.known_files.add(file_path)
            self.callback(file_path)
        else:
            logging.info(f"Image already known: {file_path}")


class ImageViewer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.thumbnail_size = (100, 100)
        self.thumbnail_queue = queue.Queue()
        self.current_page = 0
        self.total_pages = 0
        self.thumbnails_per_page = 0
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.initialize_ui()
        self.initialize_data()
        self.create_widgets()
        self.bind_events()
        self.load_settings()
        self.load_images()
        self.update_image()
        self.image_display_frame.bind("<Configure>", self.on_frame_resize)
        self.after(100, self.process_thumbnail_queue)
        self.setup_file_watcher()
        self.protocol("WM_DELETE_WINDOW", self.cleanup)  # Bind the cleanup function


    def initialize_ui(self):
        self.title("Promptvision")
        self.geometry("1280x1024")
        self.set_icon()

    def set_icon(self):
        if getattr(sys, "frozen", False):
            # Running as compiled executable
            base_path = sys._MEIPASS
        else:
            # Running in a normal Python environment
            base_path = os.path.dirname(os.path.abspath(__file__))

        logo_path = os.path.join(base_path, "assets", "pvlogo.png")
        if os.path.exists(logo_path):
            logo_image = Image.open(logo_path)
            self.iconpath = ImageTk.PhotoImage(logo_image)
            self.wm_iconbitmap()
            self.iconphoto(False, self.iconpath)
        else:
            logging.debug(f"Logo not found at {logo_path}")

    def setup_file_watcher(self):
        if self.directory:
            logging.info(f"Setting up file watcher for directory: {self.directory}")
            self.watch_event.clear()
            event_handler = ImageChangeHandler(self.on_new_image, self.image_files)
            self.observer = Observer()
            self.observer.schedule(event_handler, self.directory, recursive=False)
            self.observer.start()
            self.watch_thread = threading.Thread(target=self.watch_directory)
            self.watch_thread.start()
        else:
            logging.warning("No directory set for file watcher")

    def watch_directory(self):
        logging.info("Starting directory watch")
        event_handler = ImageChangeHandler(self.on_new_image, self.image_files)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.directory, recursive=False)
        self.observer.start()
        try:
            while not self.watch_event.is_set():
                time.sleep(1)
        except Exception as e:
            logging.error(f"Error in watch_directory: {e}")
        finally:
            self.observer.stop()
            self.observer.join()
            logging.info("Stopped directory watch")
            
    def initialize_data(self):
        self.image_files: List[str] = []
        self.current_index: int = 0
        self.directory: str = ""
        self.current_image_data: bytes = b""
        self.current_image_metadata: Dict[str, str] = {}
        self.settings_file = "viewer_settings.json"
        self.favorites = set()
        self.df = pd.DataFrame()
        self.gallery_images = []
        self.thumbnail_cache = {}
        self.total_images: int = 0
        self.show_favorites_var = ctk.BooleanVar(value=False)
        self.auto_update_var = ctk.BooleanVar(value=False)
        self.watch_thread = None
        self.watch_event = threading.Event()
        self.observer = None
        self.metadata_view = MetadataView(self)
        self.processed_images = set()

    def create_widgets(self):
        self.create_tabs()
        self.create_image_viewer()
        self.create_settings()
        self.create_gallery_view()
        self.create_progress_bar()
        self.create_navigation_frame()

    def create_progress_bar(self):
        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.set(0)

    def show_progress_bar(self):
        self.progress_bar.pack(fill="x", padx=10, pady=10)

    def hide_progress_bar(self):
        self.progress_bar.pack_forget()

    def go_to_image(self):
        try:
            index = int(self.go_to_entry.get()) - 1  # Subtract 1 to convert to zero-based index
            if 0 <= index < len(self.image_files):
                self.current_index = index
                self.update_image()
                self.go_to_entry.selection_clear()
                self.go_to_entry.icursor(ctk.END)
                self.focus()
            else:
                messagebox.showerror("Invalid Index", "Please enter a valid image number.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")

    def create_settings(self):
        settings_frame = self.settings_tab.tab("Settings")
        settings_frame.grid_columnconfigure(1, weight=1)

        row = 0
        ctk.CTkLabel(settings_frame, text="Image Directory:").grid(
            row=row, column=0, sticky="e", padx=(10, 5), pady=5
        )
        self.folder_entry = ctk.CTkEntry(settings_frame)
        self.folder_entry.grid(row=row, column=1, sticky="ew", padx=(5, 5), pady=5)
        folder_button = ctk.CTkButton(
            settings_frame, text="Browse", command=self.browse_folder
        )
        folder_button.grid(row=row, column=2, sticky="w", padx=(5, 10), pady=5)

        row += 1
        apply_button = ctk.CTkButton(
            settings_frame, text="Apply", command=self.apply_folder
        )
        apply_button.grid(row=row, column=1, sticky="ew", padx=(5, 5), pady=5)

        row += 1
        self.auto_update_var = ctk.BooleanVar(value=False)
        auto_update_checkbox = ctk.CTkCheckBox(
            settings_frame,
            text="Auto-update images",
            variable=self.auto_update_var,
            command=self.toggle_auto_update,
        )
        auto_update_checkbox.grid(row=row, column=1, sticky="w", padx=(5, 5), pady=5)

        row += 1
        ctk.CTkLabel(settings_frame, text="Export Favorites:").grid(
            row=row, column=0, sticky="e", padx=(10, 5), pady=5
        )
        self.export_entry = ctk.CTkEntry(settings_frame)
        self.export_entry.grid(row=row, column=1, sticky="ew", padx=(5, 5), pady=5)
        export_browse_button = ctk.CTkButton(
            settings_frame, text="Browse", command=self.browse_export_folder
        )
        export_browse_button.grid(row=row, column=2, sticky="w", padx=(5, 10), pady=5)

        row += 1
        export_button = ctk.CTkButton(
            settings_frame, text="Export Favorites", command=self.export_favorites
        )
        export_button.grid(row=row, column=1, sticky="ew", padx=(5, 5), pady=5)

    def create_tabs(self):
        self.settings_tab = ctk.CTkTabview(self, command=self.on_tab_changed)
        self.settings_tab.pack(fill="both", expand=True, padx=10, pady=10)
        self.settings_tab.add("Image Viewer")
        self.settings_tab.add("Settings")
        self.settings_tab.add("Gallery View")

    def on_tab_changed(self):
        selected_tab = self.settings_tab.get()
        logging.debug(f"Tab changed to: {selected_tab}")
        if selected_tab == "Gallery View":
            self.load_gallery_images()

    def load_gallery_images(self):
        self.gallery_view.image_files = self.image_files
        self.gallery_view.update_gallery()

    def create_image_viewer(self):
        viewer_frame = self.settings_tab.tab("Image Viewer")
        content_frame = ctk.CTkFrame(viewer_frame)
        content_frame.pack(expand=True, fill="both", padx=5, pady=5)

        # Create a horizontal frame to hold the image and metadata frames side by side
        self.horizontal_frame = ctk.CTkFrame(content_frame)
        self.horizontal_frame.pack(expand=True, fill="both")

        self.create_image_frame()
        self.create_metadata_view()

    def create_image_frame(self):
        self.image_frame = ctk.CTkFrame(self.horizontal_frame)
        self.image_frame.pack(side="left", expand=True, fill="both", padx=(0, 5))

        # Create a fixed-size frame for the image
        self.image_display_frame = ctk.CTkFrame(self.image_frame, width=800, height=600)
        self.image_display_frame.pack(expand=True, fill="both")
        self.image_display_frame.pack_propagate(False)

        self.image_label = ctk.CTkLabel(self.image_display_frame, text="")
        self.image_label.pack(expand=True, fill="both")

    def create_metadata_view(self):
        self.metadata_view = MetadataView(self.horizontal_frame)
        self.metadata_view.pack(side="right", expand=True, fill="both", padx=(5, 0))
        self.metadata_text = self.metadata_view.metadata_text

    def create_navigation_frame(self):
        nav_frame = ctk.CTkFrame(self.image_frame)
        nav_frame.pack(fill="x", pady=10)

        self.prev_button = ctk.CTkButton(
            nav_frame, text="Previous", command=self.previous_image
        )
        self.prev_button.pack(side="left", padx=5)

        self.image_counter = ctk.CTkLabel(nav_frame, text="")
        self.image_counter.pack(side="left", expand=True)

        self.next_button = ctk.CTkButton(
            nav_frame, text="Next", command=self.next_image
        )
        self.next_button.pack(side="left", padx=5)

        self.favorite_button = ctk.CTkButton(
            nav_frame, text="Favorite", command=self.toggle_favorite
        )
        self.favorite_button.pack(side="left", padx=5)

        # Add the go to image input and button to the navigation frame
        ctk.CTkLabel(nav_frame, text="Go to image:").pack(side="left", padx=5)
        self.go_to_entry = ctk.CTkEntry(nav_frame, width=50)
        self.go_to_entry.pack(side="left", padx=5)

        go_button = ctk.CTkButton(nav_frame, text="Go", command=self.go_to_image)
        go_button.pack(side="left", padx=5)

    def bind_events(self):
        self.bind("<Left>", lambda event: self.handle_left_key())
        self.bind("<Right>", lambda event: self.handle_right_key())
        self.bind("f", lambda event: self.toggle_favorite())
        self.bind("F", lambda event: self.toggle_favorite())
        self.bind("a", lambda event: self.handle_a_key())
        self.bind("A", lambda event: self.handle_a_key())
        self.bind("d", lambda event: self.handle_d_key())
        self.bind("D", lambda event: self.handle_d_key())
        self.go_to_entry.bind("<Return>", lambda event: self.go_to_image())

    def handle_left_key(self):
        if self.settings_tab.get() == "Image Viewer":
            self.previous_image()
        elif self.settings_tab.get() == "Gallery View":
            self.gallery_view.prev_page()

    def handle_right_key(self):
        if self.settings_tab.get() == "Image Viewer":
            self.next_image()
        elif self.settings_tab.get() == "Gallery View":
            self.gallery_view.next_page()

    def handle_a_key(self):
        if self.settings_tab.get() == "Image Viewer":
            self.previous_image()
        elif self.settings_tab.get() == "Gallery View":
            self.gallery_view.prev_page()

    def handle_d_key(self):
        if self.settings_tab.get() == "Image Viewer":
            self.next_image()
        elif self.settings_tab.get() == "Gallery View":
            self.gallery_view.next_page()

    def on_window_resize(self, event):
        # Check if the resize event is for the main window
        if event.widget == self:
            self.update_image_size()

    def on_frame_resize(self, event):
        self.update_image_size()

    def load_settings(self):
        logging.debug(f"Loading settings. Current directory: {self.directory}")
        if os.path.exists(self.settings_file):
            with open(self.settings_file, "r") as f:
                settings = json.load(f)
                self.directory = settings.get("last_directory", "")
                self.current_index = settings.get("last_position", 0)
                self.export_directory = settings.get("export_directory", "")
                self.auto_update_var.set(settings.get("auto_update", False))

            if hasattr(self, "export_entry"):
                self.export_entry.delete(0, ctk.END)
                self.export_entry.insert(0, self.export_directory)

            if hasattr(self, "sampler_filter"):
                self.toggle_auto_update()

            if self.directory:
                self.load_images()
                self.load_favorites()
                self.update_image()
            else:
                self.update_ui_for_empty_state()
        else:
            self.directory = ""
            self.current_index = 0
            self.favorites = set()
            self.export_directory = ""
            self.auto_update_var.set(False)
            # Update the UI to reflect the current state
            self.update_ui_for_empty_state()
        logging.debug(f"After loading settings. Directory: {self.directory}")

    def update_ui_for_empty_state(self):
        logging.debug("Updating UI for empty state")
        
        # Clear existing content
        for widget in self.image_display_frame.winfo_children():
            widget.destroy()

        # Create a frame to hold the logo and text
        welcome_frame = ctk.CTkFrame(self.image_display_frame)
        welcome_frame.pack(expand=True, fill="both")

        # Load and display the logo
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "pvlogo.png")
        if os.path.exists(logo_path):
            logo_image = Image.open(logo_path)
            ctk_image = ctk.CTkImage(light_image=logo_image, dark_image=logo_image, size=(200, 200))
            logo_label = ctk.CTkLabel(welcome_frame, image=ctk_image, text="")
            logo_label.pack(pady=(20, 10))

        # Add welcome text below the logo
        welcome_text = "Welcome to Promptvision!\nPlease select a directory to start."
        text_label = ctk.CTkLabel(welcome_frame, text=welcome_text, font=("Arial", 16))
        text_label.pack(pady=(0, 20))

        # Update other UI elements
        self.metadata_view.update_metadata("")
        self.image_counter.configure(text="No images loaded")
        self.prev_button.configure(state="disabled")
        self.next_button.configure(state="disabled")
        self.favorite_button.configure(state="disabled")

    def save_settings(self):
        settings = {
            "last_directory": self.directory,
            "last_position": self.current_index,
            "export_directory": self.export_entry.get(),
            "auto_update": self.auto_update_var.get(),
        }
        logging.debug(f"Saving directory: {self.directory}")  # Debug print
        with open(self.settings_file, "w") as f:
            json.dump(settings, f)
        self.save_favorites()  # Save favorites to the current directory

    def save_favorites(self):
        if self.directory:
            favorites_file = os.path.join(self.directory, "favorites.json")
            with open(favorites_file, "w") as f:
                json.dump(list(self.favorites), f)

    def load_favorites(self):
        if self.directory:
            favorites_file = os.path.join(self.directory, "favorites.json")
            if os.path.exists(favorites_file):
                with open(favorites_file, "r") as f:
                    self.favorites = set(json.load(f))

    def apply_folder(self):
        new_directory = self.folder_entry.get()
        if os.path.isdir(new_directory):
            self.save_favorites_and_position()  # Save before changing folders
            self.directory = new_directory
            logging.debug(f"Applying new directory: {self.directory}")
            if hasattr(self, 'observer') and self.observer:
                self.observer.stop()
                self.observer.join()
            self.load_images()
            self.load_favorites()
            self.update_image()
            self.settings_tab.set("Image Viewer")
            self.setup_file_watcher()  # Restart the file watcher
            self.folder_entry.selection_clear()
            self.folder_entry.icursor(ctk.END)
            self.focus()
            self.update_image_counter()
        else:
            ctk.messagebox.showerror(
                "Invalid Directory", "Please enter a valid directory path."
            )
    def load_images(self):
        logging.debug(f"Loading images from directory: {self.directory}")
        if not self.directory or not os.path.isdir(self.directory):
            self.handle_invalid_directory()
            return

        self.show_progress_bar()

        try:
            with os.scandir(self.directory) as entries:
                self.image_files = [
                    os.path.abspath(entry.path)
                    for entry in entries
                    if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))
                ]

            if not self.image_files:
                self.display_no_images_warning()
            else:
                self.handle_successful_load()

        except OSError as e:
            self.handle_load_error(e)

        finally:
            self.hide_progress_bar()
            self.total_images = len(self.image_files)
            self.update_image_counter()
            logging.debug(f"Total images loaded: {self.total_images}")

    def display_no_images_warning(self):
        if not self.directory:
            self.update_ui_for_empty_state()
            return

        warning_message = f"No supported image files found in the selected directory:\n{self.directory}"

        # Clear the image display
        self.image_label.configure(image="")
        self.image_label.image = None

        # Display the warning message in the image frame
        self.image_label.configure(text=warning_message)

        # Clear the metadata display
        self.metadata_text.configure(text="")

        # Update the image counter
        self.image_counter.configure(text="No images found")

        # Disable navigation buttons
        self.prev_button.configure(state="disabled")
        self.next_button.configure(state="disabled")
        self.favorite_button.configure(state="disabled")

        # Clear any existing image data
        self.current_image_data = b""
        self.current_image_metadata = {}
        
        # Reset the current index
        self.current_index = 0

    def clear_no_images_warning(self):
        welcome_frame = next((widget for widget in self.image_display_frame.winfo_children() if isinstance(widget, ctk.CTkFrame)), None)
        if welcome_frame:
            welcome_frame.destroy()
        
        if hasattr(self, 'image_label') and self.image_label.winfo_exists():
            self.image_label.configure(text="", image=None)
        else:
            self.image_label = ctk.CTkLabel(self.image_display_frame, text="")
            self.image_label.pack(expand=True, fill="both")

    def update_favorite_status(self):
        if self.image_files:
            current_image = self.image_files[self.current_index]
            if current_image in self.favorites:
                self.favorite_button.configure(text="Unfavorite", fg_color="#FF3B30")
            else:
                self.favorite_button.configure(text="Favorite", fg_color="#FF9500")
    
    @debounce(0.1)
    def update_image(self):
        logging.debug("Updating image in UI")
        if self.image_files:
            new_image_path = self.image_files[self.current_index]
            if not hasattr(self, 'current_image_path') or self.current_image_path != new_image_path:
                self.current_image_path = new_image_path
                try:
                    self.pil_image = Image.open(new_image_path)
                    self.ctk_image = ctk.CTkImage(light_image=self.pil_image, dark_image=self.pil_image, size=(self.image_display_frame.winfo_width(), self.image_display_frame.winfo_height()))
                    self.image_label.configure(image=self.ctk_image)
                    self.update_image_size()
                    
                    # Extract metadata using the existing function
                    image_data, self.current_image_metadata = self.metadata_view.get_image_data_and_metadata(new_image_path)
                    
                    # Update metadata display
                    self.metadata_view.update_metadata(new_image_path)
                    
                    # Update favorite status
                    self.update_favorite_status()

                    self.update_image_counter()
                except Exception as e:
                    logging.error(f"Error updating image: {e}")
            else:
                logging.debug("Image unchanged, skipping update")
        else:
            self.display_no_images_warning()

    def save_favorites_and_position(self):
        settings = {
            "last_directory": self.directory,
            "last_position": self.current_index,
            "export_directory": self.export_entry.get(),
            "auto_update": self.auto_update_var.get(),
        }
        with open(self.settings_file, "w") as f:
            json.dump(settings, f)
        self.save_favorites()

    def update_image_size(self):
        logging.debug("Updating image size in UI")
        if not hasattr(self, "pil_image"):
            logging.debug("No image loaded, skipping size update")
            return

        try:
            frame_width = self.image_display_frame.winfo_width()
            frame_height = self.image_display_frame.winfo_height()

            if frame_width <= 0 or frame_height <= 0:
                logging.debug("Invalid frame dimensions, retrying later")
                self.after(100, self.update_image_size)
                return

            img_width, img_height = self.pil_image.size
            aspect_ratio = img_width / img_height

            if frame_width / frame_height > aspect_ratio:
                new_height = frame_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = frame_width
                new_height = int(new_width / aspect_ratio)

            self.ctk_image.configure(size=(new_width, new_height))
            self.image_label.configure(image=self.ctk_image)

        except Exception as e:
            logging.error(f"Error updating image size: {e}")

    def load_and_resize_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                self.pil_image = img.copy()
            self.update_image_size()
        except Exception as e:
            logging.debug(f"Error loading image: {e}")
            self.image_label.configure(image="", text="Error loading image")

    def next_image(self):
        if self.image_files:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.update_image()

    def previous_image(self):
        if self.image_files:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.update_image()

    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_entry.delete(0, ctk.END)
            self.folder_entry.insert(0, folder_path)

    def browse_export_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.export_entry.delete(0, ctk.END)
            self.export_entry.insert(0, folder_path)

    def export_favorites(self):
        export_directory = self.export_entry.get()
        if not os.path.isdir(export_directory):
            messagebox.showerror(
                "Invalid Directory", "Please enter a valid export directory path."
            )
            return

        exported_count = 0
        for favorite in self.favorites:
            if os.path.isfile(favorite):
                filename = os.path.basename(favorite)
                destination = os.path.join(export_directory, filename)
                shutil.copy2(favorite, destination)
                exported_count += 1

        messagebox.showinfo(
            "Export Complete",
            f"Exported {exported_count} favorite images to {export_directory}",
        )

    def toggle_favorite(self):
        if self.image_files:
            current_image = self.image_files[self.current_index]
            if current_image in self.favorites:
                self.favorites.remove(current_image)
                self.favorite_button.configure(text="Favorite", fg_color="#FF9500")
            else:
                self.favorites.add(current_image)
                self.favorite_button.configure(text="Unfavorite", fg_color="#FF3B30")
            self.save_settings()

    def on_new_image(self, image_path):
        if image_path not in self.processed_images:
            self.processed_images.add(image_path)
            if image_path not in self.image_files:
                self.image_files.append(image_path)
                self.total_images = len(self.image_files)
                self.after_idle(self.update_image_counter)
                self.after_idle(self.update_image)
                self.after_idle(self.update_gallery_if_visible)
            logging.info(f"Added new image: {image_path}. Total images: {self.total_images}")
        else:
            logging.info(f"Image already processed: {image_path}")

    def update_image_counter(self):
        self.image_counter.configure(
            text=f"Image {self.current_index + 1} of {self.total_images}"
        )
        logging.debug(f"Updated image counter to {self.current_index + 1} of {self.total_images}")
    def cleanup(self):
        logging.info("Cleaning up before exit")
        self.save_favorites_and_position()
        if self.watch_thread:
            self.watch_event.set()
            self.watch_thread.join()
        self.executor.shutdown(wait=False)
        self.quit()
        self.destroy()

    def toggle_auto_update(self):
        if self.auto_update_var.get():
            logging.info("Enabling auto-update")
            self.setup_file_watcher()
        else:
            logging.info("Disabling auto-update")
            if self.watch_thread:
                self.watch_event.set()
                self.watch_thread.join()
                self.watch_thread = None


    def check_for_new_images(self):
        if self.auto_update_var.get():
            try:
                current_files = set([
                    os.path.abspath(os.path.join(self.directory, f))
                    for f in os.listdir(self.directory)
                    if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.gif']
                ])

                if current_files != set(self.image_files):
                    new_files = current_files - set(self.image_files)
                    for new_file in new_files:
                        self.on_new_image(new_file)

                self._job = self.after(5000, self.check_for_new_images)
            except FileNotFoundError as e:
                messagebox.showerror("Directory Not Found", f"Error checking for new images: {e}")
                self.auto_update_var.set(False)
                self.save_settings()

    def update_gallery_if_visible(self):
        if self.settings_tab.get() == "Gallery View":
            self.gallery_view.update_gallery()
    def preload_visible_thumbnails(self):
        visible_images = self.gallery_images[: self.thumbnails_per_page]
        for image_path in visible_images:
            if image_path not in self.thumbnail_cache:
                self.executor.submit(self.load_thumbnail, image_path)

    @lru_cache(maxsize=128)
    def load_thumbnail(self, image_path):
        try:
            logging.debug(f"Attempting to load image: {image_path}")  # Debug print
            if self.thumbnail_size[0] == 0 or self.thumbnail_size[1] == 0:
                logging.debug(
                    f"Invalid thumbnail size: {self.thumbnail_size}"
                )  # Debug print
                return

            with Image.open(image_path) as image:
                image.thumbnail(self.thumbnail_size)
                ctk_image = ctk.CTkImage(
                    light_image=image, dark_image=image, size=self.thumbnail_size
                )
                self.thumbnail_cache[image_path] = ctk_image
                self.thumbnail_queue.put((image_path, ctk_image))
        except Exception as e:
            logging.debug(
                f"Error loading thumbnail for {image_path}: {e}"
            )  # Debug print
    def process_thumbnail_queue(self):
        if self.settings_tab.get() == "Gallery View":
            logging.debug("Processing thumbnail queue for Gallery View")
            if hasattr(self, "gallery_view"):
                self.gallery_view.check_thumbnail_queue()
        else:
            logging.debug(f"Skipping thumbnail queue processing. Current tab: {self.settings_tab.get()}")
    def select_image_from_gallery(self, image_path):
        try:
            self.current_index = self.image_files.index(image_path)
            self.update_image()
            self.settings_tab.set("Image Viewer")  # Change to the Image Viewer tab
        except ValueError:
            logging.debug(f"Image path {image_path} not found in image_files")
        except Exception as e:
            logging.debug(f"Error selecting image from gallery: {e}")

    def create_gallery_view(self):
        gallery_frame = self.settings_tab.tab("Gallery View")
        self.gallery_view = GalleryView(
            gallery_frame,
            self.image_files,
            self.thumbnail_size,
            self.select_image_from_gallery,
        )
        self.gallery_view.pack(expand=True, fill="both")
        self.gallery_view.current_page = 0  # Start on page 1

    def handle_invalid_directory(self):
        self.directory = ""
        self.image_files = []
        self.current_index = 0
        self.update_ui_for_empty_state()
        self.save_settings()

    def handle_successful_load(self):
        self.clear_no_images_warning()
        self.current_index = min(self.current_index, len(self.image_files) - 1)
        self.update_image()
        self.save_settings()
        self.gallery_images = self.image_files[:100]
        self.update_gallery_if_visible()
        self.preload_visible_thumbnails()

    def handle_load_error(self, error):
        self.directory = ""
        self.image_files = []
        self.current_index = 0
        self.display_no_images_warning()
        messagebox.showerror("Invalid Directory", f"Error loading directory: {error}")
        self.save_settings()

if __name__ == "__main__":
    app = ImageViewer()
    app.mainloop()
