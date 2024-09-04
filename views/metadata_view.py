import customtkinter as ctk
import os
import png
from typing import Dict, Tuple
import pyperclip

class MetadataView(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.current_image_metadata = {}
        self.create_metadata_view()

    def create_metadata_view(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.metadata_label = ctk.CTkLabel(
            self, text="Image Metadata:", font=("", 16, "bold")
        )
        self.metadata_label.grid(row=0, column=0, pady=(5, 2), sticky="w")

        self.filename_label = ctk.CTkLabel(
            self, text="", font=("", 12, "italic")
        )
        self.filename_label.grid(row=1, column=0, pady=(0, 5), sticky="w")

        self.metadata_scrollable_frame = ctk.CTkScrollableFrame(self)
        self.metadata_scrollable_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 5))

        self.metadata_text = ctk.CTkLabel(
            self.metadata_scrollable_frame, text="", justify="left", anchor="nw"
        )
        self.metadata_text.pack(expand=True, fill="both")

        self.key_style = ("", 12, "bold")
        self.value_style = ("", 11)

        self.metadata_scrollable_frame.bind("<Configure>", self.adjust_wraplength)

        # Create the copy prompt button (it will be placed dynamically)
        self.copy_prompt_button = ctk.CTkButton(
            self.metadata_scrollable_frame, text="Copy Prompt", command=self.copy_prompt_to_clipboard
        )

    def adjust_wraplength(self, event):
        new_wraplength = self.metadata_scrollable_frame.winfo_width()
        if self.metadata_text.winfo_exists():
                self.metadata_text.configure(wraplength=new_wraplength)
    def update_metadata(self, image_path):
        if image_path:
            filename = os.path.basename(image_path)
            self.filename_label.configure(text=f"File: {filename}")

            _, metadata = self.get_image_data_and_metadata(image_path)
            self.current_image_metadata = metadata

            # Clear previous content
            for widget in self.metadata_scrollable_frame.winfo_children():
                widget.destroy()

            # Calculate the width for wrapping
            wrap_width = self.metadata_scrollable_frame.winfo_width() - 20

            metadata_text = ""
            for key, value in self.current_image_metadata.items():
                if key == "Prompt":
                    metadata_text += f"{key} (click to copy):\n{value}\n\n"
                else:
                    metadata_text += f"{key}:\n{value}\n\n"

            self.metadata_text = ctk.CTkLabel(
                self.metadata_scrollable_frame,
                text=metadata_text,
                justify="left",
                anchor="nw",
                wraplength=wrap_width
            )
            self.metadata_text.pack(fill="both", expand=True)
            
            # Bind the click event to the entire label
            self.metadata_text.bind("<Button-1>", self.copy_prompt_to_clipboard)

        else:
            self.filename_label.configure(text="")
            for widget in self.metadata_scrollable_frame.winfo_children():
                widget.destroy()

    def copy_prompt_to_clipboard(self, event=None):
        if "Prompt" in self.current_image_metadata:
            prompt = self.current_image_metadata["Prompt"]
            pyperclip.copy(prompt)
            print("Prompt copied to clipboard!")

    def get_image_data_and_metadata(self, image_path: str) -> Tuple[bytes, Dict[str, str]]:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_file.seek(0)

            reader = png.Reader(file=image_file)
            for chunk_type, chunk_data in reader.chunks():
                if chunk_type == b"tEXt" and chunk_data.startswith(b"parameters"):
                    try:
                        parameters = chunk_data.split(b"\0", 1)[1].decode("utf-8")
                    except UnicodeDecodeError:
                        parameters = chunk_data.split(b"\0", 1)[1].decode(
                            "latin-1", errors="replace"
                        )
                    parsed_metadata = self.parse_metadata(parameters)
                    return image_data, parsed_metadata

        return image_data, {"Error": "No metadata found"}

    def parse_metadata(self, metadata_string: str) -> Dict[str, str]:
        parsed_data = {}
        parts = metadata_string.split("Steps:")
        if len(parts) == 2:
            prompt = parts[0].strip()
            params = "Steps:" + parts[1]

            parsed_data["Prompt"] = prompt

            param_pairs = params.split(",")
            for pair in param_pairs:
                key_value = pair.split(":")
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    parsed_data[key] = value

        return parsed_data