import customtkinter as ctk
from PIL import Image
import threading
import logging
from tkinter import messagebox
import queue
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import mmap
import io
from threading import Lock
import time

class GalleryView(ctk.CTkFrame):
    def __init__(self, parent, image_files, thumbnail_size=(100, 100), select_callback=None):
        super().__init__(parent)
        self.image_files = image_files
        self.thumbnail_size = thumbnail_size
        self.thumbnail_cache = {}
        self.thumbnail_lock = Lock()
        self.current_page = 0
        self.total_pages = 0
        self.thumbnails_per_page = 0
        self.thumbnail_queue = queue.Queue(maxsize=100)
        self.load_thread = None
        self.create_gallery_view()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.select_callback = select_callback

    def shutdown(self):
        self.executor.shutdown(wait=True)
        # Clear any remaining items in the queue
        while not self.thumbnail_queue.empty():
            try:
                self.thumbnail_queue.get_nowait()
            except queue.Empty:
                break

    @lru_cache(maxsize=1000)
    def load_thumbnail(self, image_path):
        with self.thumbnail_lock:
            if image_path not in self.thumbnail_cache:
                try:
                    with Image.open(image_path) as img:
                        img.thumbnail(self.thumbnail_size)
                        thumbnail = ctk.CTkImage(light_image=img, dark_image=img, size=self.thumbnail_size)
                        self.thumbnail_cache[image_path] = thumbnail
                except Exception as e:
                    logging.error(f"Error loading thumbnail for {image_path}: {e}")
                    return None
            return self.thumbnail_cache[image_path]

    def load_thumbnail_mmap(self, image_path):
        with open(image_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            with Image.open(io.BytesIO(mm.read())) as img:
                img.thumbnail(self.thumbnail_size)
                return ctk.CTkImage(light_image=img, dark_image=img, size=self.thumbnail_size)
    def create_gallery_view(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        bg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTk"]["fg_color"])
        self.gallery_canvas = ctk.CTkCanvas(self, bg=bg_color, highlightthickness=0)
        self.gallery_canvas.grid(row=0, column=0, sticky="nsew")

        self.gallery_scrollbar = ctk.CTkScrollbar(self, command=self.gallery_canvas.yview)
        self.gallery_scrollbar.grid(row=0, column=1, sticky="ns")

        self.gallery_canvas.configure(yscrollcommand=self.gallery_scrollbar.set)
        self.gallery_canvas.bind('<Configure>', self.on_gallery_canvas_configure)

        self.gallery_inner_frame = ctk.CTkFrame(self.gallery_canvas, fg_color=bg_color)
        self.gallery_canvas.create_window((0, 0), window=self.gallery_inner_frame, anchor="nw", tags="inner_frame")

        self.pagination_frame = ctk.CTkFrame(self)
        self.pagination_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.prev_page_button = ctk.CTkButton(self.pagination_frame, text="Previous", command=self.prev_page)
        self.prev_page_button.pack(side="left", padx=5)

        self.page_label = ctk.CTkLabel(self.pagination_frame, text="Page 1")
        self.page_label.pack(side="left", padx=5)

        self.page_entry = ctk.CTkEntry(self.pagination_frame, width=50)
        self.page_entry.pack(side="left", padx=5)

        self.go_to_page_button = ctk.CTkButton(self.pagination_frame, text="Go", command=self.go_to_page)
        self.go_to_page_button.pack(side="left", padx=5)

        self.next_page_button = ctk.CTkButton(self.pagination_frame, text="Next", command=self.next_page)
        self.next_page_button.pack(side="left", padx=5)

    def update_gallery(self):
        start_time = time.time()  # Start the timer

        self.calculate_gallery_layout()
        start = self.current_page * self.thumbnails_per_page
        end = start + self.thumbnails_per_page
        self.visible_images = self.image_files[start:end]
        
        self.clear_gallery()
        self.render_thumbnails()
        self.update_pagination_controls()

        # Start background loading of full-resolution thumbnails
        self.load_thread = threading.Thread(target=self.load_thumbnails_thread)
        self.load_thread.start()

        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time
        logging.info(f"Gallery rendered in {elapsed_time:.2f} seconds")

    def load_thumbnails_thread(self):
        for image_path in self.visible_images:
            thumbnail = self.load_thumbnail(image_path)
            if thumbnail:
                self.thumbnail_queue.put((image_path, thumbnail))
        
        self.after(50, self.check_thumbnail_queue)

    def clear_gallery(self):
        for widget in self.gallery_inner_frame.winfo_children():
            widget.destroy()

    def calculate_gallery_layout(self):
        canvas_width = self.gallery_canvas.winfo_width()
        canvas_height = self.gallery_canvas.winfo_height()
        self.gallery_inner_frame.configure(width=canvas_width)

        self.thumbnails_per_row = max(1, canvas_width // (self.thumbnail_size[0] + 10))
        rows_per_page = max(1, canvas_height // (self.thumbnail_size[1] + 10))
        self.thumbnails_per_page = self.thumbnails_per_row * rows_per_page

        self.total_pages = (len(self.image_files) + self.thumbnails_per_page - 1) // self.thumbnails_per_page
        self.current_page = min(self.current_page, self.total_pages - 1)


    def load_visible_images(self):
        start_index = self.current_page * self.thumbnails_per_page
        end_index = min(start_index + self.thumbnails_per_page, len(self.image_files))
        self.visible_images = self.image_files[start_index:end_index]

        threading.Thread(target=self.cache_thumbnails, args=(self.visible_images,), daemon=True).start()

    def check_thumbnail_queue(self):
        updated_thumbnails = []
        processed_paths = set()
        try:
            for _ in range(self.thumbnails_per_page):
                image_path, thumbnail = self.thumbnail_queue.get_nowait()
                if image_path not in processed_paths and image_path not in self.thumbnail_cache:
                    self.thumbnail_cache[image_path] = thumbnail
                    updated_thumbnails.append((image_path, thumbnail))
                    processed_paths.add(image_path)
                self.thumbnail_queue.task_done()
        except queue.Empty:
            pass

        if updated_thumbnails:
            logging.debug(f"Processing {len(updated_thumbnails)} new thumbnails")
            self.batch_render_thumbnails(updated_thumbnails)

        self.after(100, self.check_thumbnail_queue)

    def batch_render_thumbnails(self, updated_thumbnails):
        for image_path, thumbnail in updated_thumbnails:
            self.render_thumbnail(image_path, thumbnail)
        self.gallery_canvas.update_idletasks()

    def render_thumbnails(self):
        for idx, image_path in enumerate(self.visible_images):
            row, col = divmod(idx, self.thumbnails_per_row)
            frame = ctk.CTkFrame(self.gallery_inner_frame, width=self.thumbnail_size[0], height=self.thumbnail_size[1])
            frame.grid(row=row, column=col, padx=5, pady=5)
            frame.grid_propagate(False)
            frame.image_path = image_path

            thumbnail = self.load_thumbnail(image_path)
            if thumbnail:
                label = ctk.CTkLabel(frame, image=thumbnail, text="")
                label.pack(fill="both", expand=True)
                if self.select_callback:
                    label.bind("<Button-1>", lambda e, path=image_path: self.select_callback(path))
            else:
                placeholder = ctk.CTkLabel(frame, text="Loading...")
                placeholder.pack(fill="both", expand=True)

        self.gallery_canvas.update_idletasks()
    def render_thumbnail(self, image_path, thumbnail):
        for widget in self.gallery_inner_frame.winfo_children():
            if hasattr(widget, 'image_path') and widget.image_path == image_path:
                for child in widget.winfo_children():
                    child.destroy()
                label = ctk.CTkLabel(widget, image=thumbnail, text="")
                label.pack(fill="both", expand=True)
                if self.select_callback:
                    label.bind("<Button-1>", lambda e, path=image_path: self.select_callback(path))
                break

    def cache_single_thumbnail(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.thumbnail(self.thumbnail_size)
                ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=self.thumbnail_size)
                self.thumbnail_cache[image_path] = ctk_image
                self.update_gallery()  # Refresh the gallery to show the newly cached thumbnail
        except Exception as e:
            logging.error(f"Error caching thumbnail for {image_path}: {e}")

    def cache_thumbnails(self, image_paths):
        for image_path in image_paths:
            if image_path not in self.thumbnail_cache:
                try:
                    with Image.open(image_path) as img:
                        img.thumbnail(self.thumbnail_size)
                        ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=self.thumbnail_size)
                        self.thumbnail_cache[image_path] = ctk_image
                except Exception as e:
                    logging.error(f"Error caching thumbnail for {image_path}: {e}")

    def update_pagination_controls(self):
        self.total_pages = (len(self.image_files) + self.thumbnails_per_page - 1) // self.thumbnails_per_page
        self.page_label.configure(text=f"Page {self.current_page + 1} of {self.total_pages}")
        self.prev_page_button.configure(state="normal")
        self.next_page_button.configure(state="normal")

    def prev_page(self):
        self.current_page = (self.current_page - 1) % self.total_pages
        self.update_gallery()

    def next_page(self):
        self.current_page = (self.current_page + 1) % self.total_pages
        self.update_gallery()

    def go_to_page(self):
        try:
            page_number = int(self.page_entry.get()) - 1
            if 0 <= page_number < self.total_pages:
                self.current_page = page_number
                self.update_gallery()
                self.page_entry.delete(0, 'end')
            else:
                messagebox.showerror("Invalid Page", "Please enter a valid page number.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid page number.")

    def on_gallery_canvas_configure(self, event):
        canvas_width = event.width
        self.gallery_inner_frame.configure(width=canvas_width)
        self.update_gallery()