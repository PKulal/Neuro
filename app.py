"""
app.py
------
STEP 12 of NeuroConnect AI v2 -- the GUI.

The user selects ONE whole brain MRI (.nii / .nii.gz) and gets ONE
patient-level result. Slice extraction, per-slice scoring and aggregation all
happen internally; individual slice results are never shown.

Run:
    python app.py
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# Colours
BG = "#10141c"
PANEL = "#1a2130"
FG = "#e8edf5"
MUTED = "#8fa0bb"
ACCENT = "#4a9eff"
ALERT = "#ff6b6b"
OK = "#3ddc97"

# The window opens maximised, but the controls stay inside a centred column.
# The exact width is computed from the screen at startup.
CONTENT_WIDTH_MAX = 900
CONTENT_WIDTH_MIN = 520


def enable_dpi_awareness() -> None:
    """Tell Windows this process handles high-DPI scaling itself.

    Without this a Tk window on a scaled display (e.g. 1920x1200 at 150%) is
    bitmap-stretched by the OS: text turns blurry and the real window size
    stops matching the numbers Tk reports.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()       # Windows 7/8
    except Exception:
        pass  # cosmetic only -- never block startup over this

HELP_TEXT = (
    "How to read this result\n\n"
    "• 'Autism Probability' and 'Healthy Probability' are the model's "
    "estimated scores for its two classes, produced by combining the scores "
    "of the brain slices extracted from the MRI you selected. The two add up "
    "to 100%.\n\n"
    "• They are NOT a measure of how much autism a person biologically has, "
    "and they are NOT a clinical diagnosis.\n\n"
    "• The model's decision threshold is not exactly 50%, so a HEALTHY "
    "result can occasionally show a Healthy score slightly below 50%. That "
    "is the model's real score, shown unaltered, and it will always be "
    "flagged as Low confidence.\n\n"
    "• 'Confidence' is the share of the MRI's brain slices whose own scores "
    "landed on the same side of the decision line as the final answer. It "
    "measures how CONSISTENT the evidence was — which the probability above "
    "does not tell you.\n\n"
    "• The High / Moderate / Low label is downgraded if either the slices "
    "disagreed OR the overall score only barely cleared the decision line, "
    "so a verdict is only ever as strong as its weaker signal.\n\n"
    "• Confidence is about the model's internal agreement, NOT medical "
    "certainty. 90% confidence does not mean the answer is 90% likely to be "
    "clinically correct — for that, see the measured test accuracy in "
    "Reports/classification_report.txt.\n\n"
    "• NeuroConnect AI is a research prototype trained on the ABIDE-II "
    "dataset. It has not been clinically validated and must not be used to "
    "make medical decisions."
)


class NeuroConnectApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NeuroConnect AI - MRI Autism Research Tool")
        self.configure(bg=BG)
        self.minsize(560, 600)

        # Open maximised. F11 switches to true fullscreen (no title bar),
        # Escape leaves it again.
        self._fullscreen = False
        try:
            self.state("zoomed")
        except tk.TclError:                      # non-Windows fallback
            self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._exit_fullscreen)

        # Half the screen width reads comfortably on both a laptop panel and
        # a wide external monitor, clamped so it never gets silly either way.
        self.content_width = max(
            CONTENT_WIDTH_MIN,
            min(CONTENT_WIDTH_MAX, int(self.winfo_screenwidth() * 0.5)),
        )

        self.mri_path: str | None = None
        self.predictor = None
        self._queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._load_model_async()
        self.after(100, self._drain_queue)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Content sits in a fixed-width column centred on screen. Stretching
        # the controls across a 1920px monitor would leave the text stranded
        # in the middle of very wide buttons.
        centre = tk.Frame(self, bg=BG)
        centre.place(relx=0.5, y=0, anchor="n", width=self.content_width,
                     relheight=1.0)

        header = tk.Frame(centre, bg=BG)
        header.pack(fill="x", pady=(48, 6))
        tk.Label(header, text="NeuroConnect AI", bg=BG, fg=FG,
                 font=("Segoe UI", 34, "bold")).pack()
        tk.Label(header, text="MRI Autism Research Tool", bg=BG, fg=MUTED,
                 font=("Segoe UI", 14)).pack(pady=(4, 0))

        tk.Frame(centre, bg="#2a3348", height=1).pack(fill="x", pady=26)

        # --- file selection --------------------------------------------
        body = tk.Frame(centre, bg=BG)
        body.pack(fill="x")

        self.select_btn = tk.Button(
            body, text="Select Brain MRI", command=self._select_file,
            bg=PANEL, fg=FG, activebackground="#243049", activeforeground=FG,
            font=("Segoe UI", 14), relief="flat", bd=0, cursor="hand2",
            padx=18, pady=14,
        )
        self.select_btn.pack(fill="x")

        self.file_label = tk.Label(
            body, text="No file selected", bg=BG, fg=MUTED,
            font=("Segoe UI", 11), wraplength=self.content_width - 40,
            justify="center",
        )
        self.file_label.pack(pady=(12, 0))

        self.predict_btn = tk.Button(
            body, text="PREDICT", command=self._predict,
            bg=ACCENT, fg="#08111f", activebackground="#66b0ff",
            activeforeground="#08111f", font=("Segoe UI", 16, "bold"),
            relief="flat", bd=0, cursor="hand2", padx=18, pady=16,
            state="disabled", disabledforeground="#4d6e88",
        )
        self.predict_btn.pack(fill="x", pady=(18, 0))

        self.progress = ttk.Progressbar(body, mode="indeterminate")

        # --- result panel ----------------------------------------------
        self.result_frame = tk.Frame(centre, bg=PANEL)
        self.result_frame.pack(fill="both", expand=True, pady=(26, 12))

        # Spacers above and below keep the verdict vertically centred in the
        # panel however tall the window is
        tk.Frame(self.result_frame, bg=PANEL).pack(expand=True)

        self.result_title = tk.Label(
            self.result_frame, text="Result", bg=PANEL, fg=MUTED,
            font=("Segoe UI", 11),
        )
        self.result_title.pack(pady=(0, 8))

        self.result_label = tk.Label(
            self.result_frame, text="—", bg=PANEL, fg=FG,
            font=("Segoe UI", 34, "bold"), wraplength=self.content_width - 60,
            justify="center",
        )
        self.result_label.pack(pady=(0, 10))

        self.detail_label = tk.Label(
            self.result_frame, text="", bg=PANEL, fg=MUTED,
            font=("Segoe UI", 16), justify="center",
            wraplength=self.content_width - 70,
        )
        self.detail_label.pack(pady=(0, 6))

        tk.Frame(self.result_frame, bg=PANEL).pack(expand=True)

        # --- footer -----------------------------------------------------
        footer = tk.Frame(centre, bg=BG)
        footer.pack(fill="x", pady=(0, 22))

        tk.Button(footer, text="What does this mean?", command=self._show_help,
                  bg=BG, fg=ACCENT, activebackground=BG, activeforeground=FG,
                  font=("Segoe UI", 11, "underline"), relief="flat", bd=0,
                  cursor="hand2").pack()

        self.status = tk.Label(
            footer,
            text="Research prototype - not a clinically validated diagnostic system.",
            bg=BG, fg=MUTED, font=("Segoe UI", 9),
            wraplength=self.content_width - 40,
        )
        self.status.pack(pady=(8, 0))

        tk.Label(footer, text="F11 fullscreen  ·  Esc exit fullscreen",
                 bg=BG, fg="#5b6b85", font=("Segoe UI", 8)).pack(pady=(4, 0))

    # ------------------------------------------------------------------
    # Model loading (background, so the window appears immediately)
    # ------------------------------------------------------------------
    def _load_model_async(self) -> None:
        self.status.config(text="Loading model, please wait...")

        def work():
            try:
                from predictor import NeuroPredictor
                predictor = NeuroPredictor()
                self._queue.put(("model_ok", predictor))
            except ModuleNotFoundError as exc:
                # Almost always means the app was launched with the system
                # Python instead of the project's virtual environment.
                self._queue.put(("model_err", (
                    "WRONG PYTHON",
                    f"'{exc.name}' is not installed in the Python running this "
                    f"app.\n\nRun it from the project virtual environment "
                    f"instead:\n\n"
                    f"    cd \"{PROJECT_DIR}\"\n"
                    f"    .venv\\Scripts\\python.exe app.py\n\n"
                    f"(currently running: {sys.executable})"
                )))
            except FileNotFoundError as exc:
                self._queue.put(("model_err", (
                    "NO TRAINED MODEL",
                    f"{exc}\n\nRun train_model.py first.",
                )))
            except Exception as exc:
                self._queue.put(("model_err",
                                 ("MODEL NOT LOADED", f"{type(exc).__name__}: {exc}")))

        threading.Thread(target=work, daemon=True).start()

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "model_ok":
                    self.predictor = payload
                    self.status.config(
                        text="Research prototype - not a clinically "
                             "validated diagnostic system.")
                    self._refresh_predict_state()
                elif kind == "model_err":
                    title, detail = payload
                    self._set_result(title, ALERT, detail)
                    self.detail_label.config(font=("Consolas", 9), justify="left")
                    self.status.config(
                        text="The app cannot make predictions until this is "
                             "resolved.")
                elif kind == "result":
                    self._finish_prediction(payload)
                elif kind == "error":
                    self.progress.pack_forget()
                    self.progress.stop()
                    self._set_result("ERROR", ALERT, payload)
                    self._refresh_predict_state()
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _select_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a whole-brain MRI",
            filetypes=[("NIfTI MRI", "*.nii *.nii.gz"), ("All files", "*.*")],
        )
        if not path:
            return
        self.mri_path = path
        self.file_label.config(text=os.path.basename(path), fg=FG)
        self._set_result("—", FG, "")
        self._refresh_predict_state()

    def _refresh_predict_state(self) -> None:
        ready = self.predictor is not None and self.mri_path is not None
        self.predict_btn.config(state="normal" if ready else "disabled")

    def _predict(self) -> None:
        if self.predictor is None or not self.mri_path:
            return

        self.predict_btn.config(state="disabled")
        self.select_btn.config(state="disabled")
        self._set_result("Analysing MRI...", MUTED, "")
        self.progress.pack(fill="x", pady=(12, 0))
        self.progress.start(12)

        path = self.mri_path

        def work():
            try:
                self._queue.put(("result", self.predictor.predict_mri(path)))
            except Exception as exc:
                self._queue.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def _finish_prediction(self, result: dict) -> None:
        self.progress.stop()
        self.progress.pack_forget()
        self.select_btn.config(state="normal")
        self._refresh_predict_state()

        if not result["valid"]:
            self._set_result(
                "Invalid MRI file.", ALERT,
                "Please select a valid whole-brain MRI (.nii or .nii.gz).\n"
                f"({result['reason']})",
            )
            return

        conf = (f"Confidence: {result['confidence_percent']:.1f}% "
                f"({result['confidence']})")

        if result["label"] == "AUTISM":
            self._set_result(
                "AUTISM DETECTED", ALERT,
                f"Autism Probability: {result['autism_probability'] * 100:.1f}%\n"
                f"{conf}",
            )
        else:
            self._set_result(
                "HEALTHY", OK,
                f"Healthy Probability: {result['healthy_probability'] * 100:.1f}%\n"
                f"{conf}",
            )

    def _toggle_fullscreen(self, event=None) -> str:
        self._fullscreen = not self._fullscreen
        self.attributes("-fullscreen", self._fullscreen)
        return "break"

    def _exit_fullscreen(self, event=None) -> str:
        if self._fullscreen:
            self._fullscreen = False
            self.attributes("-fullscreen", False)
        return "break"

    def _set_result(self, text: str, colour: str, detail: str) -> None:
        self.result_label.config(text=text, fg=colour)
        # Reset styling each time; the error path overrides it afterwards
        self.detail_label.config(text=detail, font=("Segoe UI", 16),
                                 justify="center")

    def _show_help(self) -> None:
        win = tk.Toplevel(self)
        win.title("How to read this result")
        win.configure(bg=BG)
        win.geometry("640x580")
        win.transient(self)
        tk.Label(win, text=HELP_TEXT, bg=BG, fg=FG, font=("Segoe UI", 11),
                 wraplength=570, justify="left").pack(padx=32, pady=28)
        tk.Button(win, text="Close", command=win.destroy, bg=PANEL, fg=FG,
                  relief="flat", bd=0, cursor="hand2", padx=20, pady=8,
                  font=("Segoe UI", 10)).pack(pady=(0, 20))


if __name__ == "__main__":
    enable_dpi_awareness()
    NeuroConnectApp().mainloop()
