"""InfraExplorer desktop UI. Dark instrument chrome, local files only."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import numpy as np

from .engine import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_FMAX,
    DEFAULT_FMIN,
    DEFAULT_OUTPUT_RATE,
    DEFAULT_SPEED,
    DSP_VERSION,
    GAIN_CAPS,
    HEADROOM_DB,
    INGEST_RATE_CAP,
    PRESET_LABELS,
    PROJECT_SCHEMA,
    SYNTHETICS,
    Band,
    ProcessConfig,
    ProcessResult,
    SourceInfo,
    analysis_csv,
    auto_working_rate,
    decode_audio,
    encode_wav32f,
    format_band,
    format_bytes,
    format_db,
    format_duration,
    format_hz,
    generate_synthetic,
    master_trim,
    mix_bands,
    preset_bands,
    process_channels,
    processing_log_text,
    run_self_test,
    safe_filename,
    source_from_file,
    source_from_synthetic,
    spectrogram,
    speed_safety,
    time_compress,
)

BG = "#0b0c0e"
FG = "#e7e4dc"
CARD = "#13151a"
MUTED = "#8d908c"
ACCENT = "#9eb8b0"
BORDER = "#2a2d32"
INPUT = "#1a1d24"
PRIMARY = "#c5cec8"
WARN = "#c4a574"
OK = "#7d9a86"
DANGER = "#c47a6a"
MEASURE = "#d4cfc4"

CLASS_LABEL = {
    "very-weak": "Very weak",
    "weak": "Weak",
    "moderate": "Moderate",
    "strong": "Strong",
    "very-strong": "Very strong",
}


def _font(size: int = 10, mono: bool = False, medium: bool = False) -> tuple:
    family = "Consolas" if mono else "Segoe UI"
    if sys.platform == "darwin":
        family = "Menlo" if mono else "Helvetica Neue"
    elif sys.platform.startswith("linux"):
        family = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    return (family, size, "bold" if medium else "normal")


def _play_wav(path: str) -> None:
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        return
    import subprocess

    for cmd in (("afplay", path), ("paplay", path), ("aplay", path)):
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue
    try:
        os.startfile(path)  # type: ignore[attr-defined]
    except Exception:
        pass


def _stop_wav() -> None:
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(None, winsound.SND_PURGE)


class SpectrumCanvas(tk.Canvas):
    def __init__(self, master: tk.Widget, **kw) -> None:
        super().__init__(
            master,
            bg=CARD,
            highlightthickness=0,
            height=kw.pop("height", 140),
            **kw,
        )
        self.bind("<Configure>", lambda _e: self._redraw())
        self._freqs: np.ndarray | None = None
        self._db: np.ndarray | None = None
        self._fmin = DEFAULT_FMIN
        self._fmax = DEFAULT_FMAX
        self._bands: list[Band] = []

    def set_data(
        self,
        freqs: np.ndarray | None,
        db: np.ndarray | None,
        fmin: float,
        fmax: float,
        bands: list[Band],
    ) -> None:
        self._freqs = freqs
        self._db = db
        self._fmin = fmin
        self._fmax = fmax
        self._bands = bands
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        w = max(2, self.winfo_width())
        h = max(2, self.winfo_height())
        if self._freqs is None or self._db is None or self._freqs.size == 0:
            self.create_text(w // 2, h // 2, text="No spectrum yet", fill=MUTED, font=_font(9))
            return
        lo = max(self._fmin, 1e-4)
        hi = max(self._fmax, lo * 1.01)
        log_lo, log_hi = np.log(lo), np.log(hi)
        db_lo, db_hi = -100.0, 0.0

        def x_of(f: float) -> float:
            f = min(max(f, lo), hi)
            return (np.log(f) - log_lo) / (log_hi - log_lo) * (w - 8) + 4

        def y_of(db: float) -> float:
            t = (db - db_lo) / (db_hi - db_lo)
            t = min(1.0, max(0.0, t))
            return h - 6 - t * (h - 12)

        for i, b in enumerate(self._bands):
            x0, x1 = x_of(b.lo), x_of(b.hi)
            color = "#1c2422" if i % 2 == 0 else "#181c20"
            self.create_rectangle(x0, 0, x1, h, fill=color, outline="")
        pts: list[float] = [4, h - 4]
        for f, d in zip(self._freqs, self._db):
            if f < lo or f > hi:
                continue
            pts += [x_of(float(f)), y_of(float(d))]
        pts += [w - 4, h - 4]
        if len(pts) >= 8:
            self.create_polygon(pts, fill="#2c3834", outline=ACCENT, width=1)
        self.create_text(8, 10, anchor="w", text="MEAN SPECTRUM", fill=MUTED, font=_font(8, mono=True))


class SpecgramCanvas(tk.Canvas):
    def __init__(self, master: tk.Widget, **kw) -> None:
        super().__init__(master, bg=CARD, highlightthickness=0, height=kw.pop("height", 140), **kw)
        self._img = None
        self.bind("<Configure>", lambda _e: self._redraw())
        self._signal: np.ndarray | None = None
        self._rate = 200.0
        self._fmin = DEFAULT_FMIN
        self._fmax = DEFAULT_FMAX

    def set_data(self, signal: np.ndarray | None, rate: float, fmin: float, fmax: float) -> None:
        self._signal = signal
        self._rate = rate
        self._fmin = fmin
        self._fmax = fmax
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        w = max(8, self.winfo_width())
        h = max(8, self.winfo_height())
        self.create_text(8, 10, anchor="w", text="SPECTROGRAM", fill=MUTED, font=_font(8, mono=True))
        if self._signal is None or self._signal.size < 64:
            return
        cols = min(220, max(40, w // 3))
        rows = min(80, max(24, h // 3))
        img = spectrogram(self._signal, self._rate, self._fmin, self._fmax, cols, rows)
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            return
        lo, hi = np.percentile(finite, 8), np.percentile(finite, 98)
        if hi - lo < 1:
            hi = lo + 1
        norm = np.clip((img - lo) / (hi - lo), 0, 1)
        photo = tk.PhotoImage(width=cols, height=rows)
        rows_hex = []
        for r in range(rows):
            parts = []
            for c in range(cols):
                t = float(norm[r, c])
                rr = int(18 + t * 150)
                gg = int(22 + t * 170)
                bb = int(24 + t * 140)
                parts.append(f"#{rr:02x}{gg:02x}{bb:02x}")
            rows_hex.append("{" + " ".join(parts) + "}")
        photo.put(" ".join(rows_hex))
        photo = photo.zoom(max(1, w // cols), max(1, h // rows))
        self._img = photo
        self.create_image(0, 0, anchor="nw", image=photo)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_NAME}  {APP_VERSION}")
        self.configure(bg=BG)
        self.geometry("1280x820")
        self.minsize(960, 640)
        try:
            self.tk.call("tk", "scaling", 1.15)
        except tk.TclError:
            pass
        self._style()

        self.source: SourceInfo | None = None
        self.raw: list[np.ndarray] | None = None
        self.raw_rate = 0
        self.channel_names = ["Channel 1", "Channel 2"]
        self.preset = "coarse"
        self.bands = preset_bands("coarse")
        self.f_min = DEFAULT_FMIN
        self.f_max = DEFAULT_FMAX
        self.working_auto = True
        self.working_rate = auto_working_rate(self.f_max)
        self.speed = DEFAULT_SPEED
        self.gain_mode = "medium"
        self.test_duration: float | None = None
        self.hardware = {k: "" for k in ("sensor", "preamp", "interface", "mounting", "location", "notes")}
        self.result: ProcessResult | None = None
        self.gains: list[list[float]] = []
        self.mute: list[list[bool]] = []
        self.solo: list[list[bool]] = []
        self.preview_mode = "balanced"
        self.layout_mode = "ab"
        self.master_trim_db = 0.0
        self.processing = False
        self.progress: dict | None = None
        self.notice = "No source loaded. Original files are never overwritten."
        self.error: str | None = None
        self.log: list[str] = []
        self.selftest: list[dict] | None = None
        self.playing = False
        self.play_started = 0.0
        self.play_offset = 0.0
        self.loop = False
        self.play_path: str | None = None
        self.mix_cache: list[np.ndarray] | None = None
        self.mix_duration = 0.0
        self.tab = "load"

        self._build()
        self._show("load")
        self.after(80, self._tick)

    def _style(self) -> None:
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", background=BG, foreground=FG, fieldbackground=INPUT, bordercolor=BORDER)
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background=CARD)
        s.configure("TLabel", background=BG, foreground=FG, font=_font(10))
        s.configure("Muted.TLabel", background=BG, foreground=MUTED, font=_font(9))
        s.configure("Mono.TLabel", background=BG, foreground=MUTED, font=_font(9, mono=True))
        s.configure("Head.TLabel", background=BG, foreground=FG, font=_font(22, medium=True))
        s.configure("H2.TLabel", background=BG, foreground=FG, font=_font(12, medium=True))
        s.configure("Card.TLabel", background=CARD, foreground=FG)
        s.configure("CardMuted.TLabel", background=CARD, foreground=MUTED, font=_font(9))
        s.configure("TButton", background=INPUT, foreground=FG, padding=(12, 7), font=_font(10))
        s.map("TButton", background=[("active", "#242830"), ("disabled", "#14161a")])
        s.configure("Accent.TButton", background=PRIMARY, foreground=BG, padding=(14, 8), font=_font(10, medium=True))
        s.map("Accent.TButton", background=[("active", ACCENT)])
        s.configure("TEntry", fieldbackground=INPUT, foreground=FG, insertcolor=FG)
        s.configure("TCombobox", fieldbackground=INPUT, foreground=FG, background=INPUT)
        s.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=INPUT, bordercolor=BORDER)
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=INPUT, foreground=MUTED, padding=(14, 8), font=_font(10))
        s.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", FG)])
        s.configure("TCheckbutton", background=CARD, foreground=FG)
        s.configure("TRadiobutton", background=BG, foreground=FG)

    def _build(self) -> None:
        header = tk.Frame(self, bg=BG, highlightthickness=1, highlightbackground=BORDER)
        header.pack(fill="x")
        tk.Label(header, text=APP_NAME, bg=BG, fg=FG, font=_font(12, medium=True)).pack(
            side="left", padx=(16, 8), pady=10
        )
        tk.Label(
            header,
            text=f"{APP_VERSION}  ·  DSP {DSP_VERSION}",
            bg=BG,
            fg=MUTED,
            font=_font(9, mono=True),
        ).pack(side="left")
        self.nav = tk.Frame(header, bg=BG)
        self.nav.pack(side="right", padx=8)
        self.nav_btns: dict[str, tk.Button] = {}
        for key, label in (("setup", "Setup"), ("mixer", "Mixer"), ("report", "Report")):
            b = tk.Button(
                self.nav,
                text=label,
                bg=BG,
                fg=MUTED,
                relief="flat",
                font=_font(10),
                padx=12,
                pady=8,
                command=lambda k=key: self._show(k),
            )
            b.pack(side="left")
            self.nav_btns[key] = b

        self.banner = tk.Label(
            self, text=self.notice, bg=INPUT, fg=MUTED, anchor="w", font=_font(9), padx=16, pady=8
        )
        self.banner.pack(fill="x")

        self.prog_wrap = tk.Frame(self, bg=CARD)
        self.prog_label = tk.Label(self.prog_wrap, text="", bg=CARD, fg=MUTED, font=_font(9, mono=True), anchor="w")
        self.prog_label.pack(fill="x", padx=16, pady=(8, 4))
        self.prog_bar = ttk.Progressbar(self.prog_wrap, mode="determinate")
        self.prog_bar.pack(fill="x", padx=16, pady=(0, 10))

        self.body = tk.Frame(self, bg=BG)
        self.body.pack(fill="both", expand=True)

        self.screens: dict[str, tk.Frame] = {}
        self.screens["load"] = self._build_load()
        self.screens["setup"] = self._build_setup()
        self.screens["mixer"] = self._build_mixer()
        self.screens["report"] = self._build_report()
        for fr in self.screens.values():
            fr.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.transport = tk.Frame(self, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        self.transport.pack(fill="x", side="bottom")
        self.play_btn = tk.Button(
            self.transport,
            text="Play",
            bg=PRIMARY,
            fg=BG,
            relief="flat",
            font=_font(10, medium=True),
            padx=16,
            pady=8,
            command=self.toggle_play,
        )
        self.play_btn.pack(side="left", padx=12, pady=10)
        self.loop_btn = tk.Button(
            self.transport, text="Loop", bg=INPUT, fg=MUTED, relief="flat", padx=12, pady=8, command=self.toggle_loop
        )
        self.loop_btn.pack(side="left")
        self.time_label = tk.Label(self.transport, text="—", bg=CARD, fg=MUTED, font=_font(9, mono=True))
        self.time_label.pack(side="right", padx=16)
        self.status = tk.Label(
            self,
            text="No source loaded · WAV / FLAC / AIFF",
            bg=BG,
            fg=MUTED,
            font=_font(8, mono=True),
            anchor="w",
            padx=16,
            pady=6,
        )
        self.status.pack(fill="x", side="bottom")

    def _card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)

    def _build_load(self) -> tk.Frame:
        wrap = tk.Frame(self.body, bg=BG)
        inner = tk.Frame(wrap, bg=BG)
        inner.pack(fill="both", expand=True, padx=40, pady=32)
        tk.Label(
            inner,
            text="ANALYSIS  ·  RENDERING  ·  NOT A DESTRUCTIVE EDITOR",
            bg=BG,
            fg=MUTED,
            font=_font(8, mono=True),
        ).pack(anchor="w")
        tk.Label(inner, text="InfraExplorer", bg=BG, fg=FG, font=_font(28, medium=True)).pack(anchor="w", pady=(8, 0))
        tk.Label(
            inner,
            text="Take long piezo recordings, keep every channel honest, and translate selected\ninfrasound into something you can actually hear — without pretending a +36 dB\nlistening boost was in the room.",
            bg=BG,
            fg=MUTED,
            font=_font(11),
            justify="left",
        ).pack(anchor="w", pady=(12, 24))

        drop = self._card(inner)
        drop.pack(fill="x", pady=(0, 24))
        row = tk.Frame(drop, bg=CARD)
        row.pack(fill="x", padx=20, pady=20)
        left = tk.Frame(row, bg=CARD)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text="Open a recording", bg=CARD, fg=FG, font=_font(11, medium=True)).pack(anchor="w")
        tk.Label(
            left,
            text="WAV, FLAC, RF64, or AIFF. The original is never written.",
            bg=CARD,
            fg=MUTED,
            font=_font(10),
        ).pack(anchor="w", pady=(4, 0))
        tk.Button(
            row,
            text="Browse…",
            bg=PRIMARY,
            fg=BG,
            relief="flat",
            font=_font(10, medium=True),
            padx=16,
            pady=8,
            command=self.open_file,
        ).pack(side="right")

        tk.Label(inner, text="Or load a known session", bg=BG, fg=FG, font=_font(11, medium=True)).pack(
            anchor="w", pady=(0, 10)
        )
        grid = tk.Frame(inner, bg=BG)
        grid.pack(fill="x")
        for i, spec in enumerate(SYNTHETICS):
            card = self._card(grid)
            card.grid(row=i // 2, column=i % 2, sticky="nsew", padx=(0, 12) if i % 2 == 0 else (0, 0), pady=6)
            grid.columnconfigure(i % 2, weight=1)
            tk.Label(card, text=spec.title, bg=CARD, fg=FG, font=_font(10, medium=True)).pack(
                anchor="w", padx=16, pady=(12, 0)
            )
            tk.Label(
                card,
                text=f"{spec.channels}ch · {spec.sample_rate} Hz · {int(spec.duration // 60)} min",
                bg=CARD,
                fg=MUTED,
                font=_font(8, mono=True),
            ).pack(anchor="w", padx=16)
            tk.Label(card, text=spec.blurb, bg=CARD, fg=MUTED, font=_font(9), wraplength=420, justify="left").pack(
                anchor="w", padx=16, pady=(6, 8)
            )
            tk.Button(
                card,
                text="Load",
                bg=INPUT,
                fg=FG,
                relief="flat",
                command=lambda k=spec.id: self.load_synthetic(k),
            ).pack(anchor="e", padx=16, pady=(0, 12))

        tk.Label(
            inner,
            text="Preserve first. Measure second. Translate third. Enhance only for listening.",
            bg=BG,
            fg=MUTED,
            font=_font(8, mono=True),
        ).pack(anchor="w", pady=16)
        return wrap

    def _build_setup(self) -> tk.Frame:
        wrap = tk.Frame(self.body, bg=BG)
        canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        scroll = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.setup_inner = inner
        self.setup_meta = tk.Label(inner, text="", bg=BG, fg=MUTED, font=_font(9, mono=True), justify="left")
        self.setup_meta.pack(anchor="w", padx=24, pady=(18, 8))

        rec = self._card(inner)
        rec.pack(fill="x", padx=24, pady=8)
        top = tk.Frame(rec, bg=CARD)
        top.pack(fill="x", padx=16, pady=12)
        tk.Label(top, text="RECORDING", bg=CARD, fg=MUTED, font=_font(8, mono=True)).pack(anchor="w")
        self.setup_name = tk.Label(top, text="", bg=CARD, fg=FG, font=_font(12, medium=True))
        self.setup_name.pack(anchor="w")
        tk.Button(top, text="Unload", bg=INPUT, fg=FG, relief="flat", command=self.unload).pack(side="right")

        names = tk.Frame(rec, bg=CARD)
        names.pack(fill="x", padx=16, pady=(0, 12))
        tk.Label(names, text="Channel names", bg=CARD, fg=MUTED, font=_font(9)).pack(anchor="w")
        self.ch_vars = [tk.StringVar(value="Channel 1"), tk.StringVar(value="Channel 2")]
        for i in range(2):
            e = tk.Entry(names, textvariable=self.ch_vars[i], bg=INPUT, fg=FG, insertbackground=FG, relief="flat")
            e.pack(fill="x", pady=3)
            e.bind("<KeyRelease>", lambda _e: self._read_channel_names())

        rng = self._card(inner)
        rng.pack(fill="x", padx=24, pady=8)
        tk.Label(rng, text="ANALYSIS RANGE", bg=CARD, fg=MUTED, font=_font(8, mono=True)).pack(
            anchor="w", padx=16, pady=(12, 4)
        )
        grid = tk.Frame(rng, bg=CARD)
        grid.pack(fill="x", padx=16, pady=(0, 12))
        self.var_fmin = tk.StringVar(value=str(DEFAULT_FMIN))
        self.var_fmax = tk.StringVar(value=str(DEFAULT_FMAX))
        self.var_speed = tk.StringVar(value=str(int(DEFAULT_SPEED)))
        self.var_preset = tk.StringVar(value="coarse")
        self.var_gain = tk.StringVar(value="medium")
        self.var_test = tk.StringVar(value="full")
        self.var_work = tk.StringVar(value="auto")
        for i, (lab, var) in enumerate((("F min Hz", self.var_fmin), ("F max Hz", self.var_fmax), ("Speed ×", self.var_speed))):
            tk.Label(grid, text=lab, bg=CARD, fg=MUTED, font=_font(9)).grid(row=0, column=i, sticky="w", padx=(0, 16))
            tk.Entry(grid, textvariable=var, bg=INPUT, fg=FG, insertbackground=FG, relief="flat", width=12).grid(
                row=1, column=i, sticky="w", padx=(0, 16), pady=(0, 8)
            )
        tk.Label(grid, text="Band preset", bg=CARD, fg=MUTED, font=_font(9)).grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            grid,
            textvariable=self.var_preset,
            values=list(PRESET_LABELS.keys()),
            state="readonly",
            width=16,
        ).grid(row=3, column=0, sticky="w", padx=(0, 16))
        tk.Label(grid, text="Gain mode", bg=CARD, fg=MUTED, font=_font(9)).grid(row=2, column=1, sticky="w")
        ttk.Combobox(
            grid,
            textvariable=self.var_gain,
            values=["preserve", "conservative", "medium", "aggressive"],
            state="readonly",
            width=16,
        ).grid(row=3, column=1, sticky="w", padx=(0, 16))
        tk.Label(grid, text="Duration", bg=CARD, fg=MUTED, font=_font(9)).grid(row=2, column=2, sticky="w")
        ttk.Combobox(
            grid,
            textvariable=self.var_test,
            values=["full", "30s test", "2 min test", "10 min test"],
            state="readonly",
            width=16,
        ).grid(row=3, column=2, sticky="w")

        self.warn_label = tk.Label(rng, text="", bg=CARD, fg=WARN, font=_font(9), wraplength=800, justify="left")
        self.warn_label.pack(anchor="w", padx=16, pady=(0, 12))

        hw = self._card(inner)
        hw.pack(fill="x", padx=24, pady=8)
        tk.Label(hw, text="HARDWARE NOTES  (optional, written into the report)", bg=CARD, fg=MUTED, font=_font(8, mono=True)).pack(
            anchor="w", padx=16, pady=(12, 6)
        )
        self.hw_vars = {k: tk.StringVar() for k in self.hardware}
        g = tk.Frame(hw, bg=CARD)
        g.pack(fill="x", padx=16, pady=(0, 12))
        for i, key in enumerate(("sensor", "preamp", "interface", "mounting", "location", "notes")):
            tk.Label(g, text=key.capitalize(), bg=CARD, fg=MUTED, font=_font(9)).grid(
                row=i // 2 * 2, column=i % 2, sticky="w", padx=(0, 12)
            )
            tk.Entry(g, textvariable=self.hw_vars[key], bg=INPUT, fg=FG, insertbackground=FG, relief="flat").grid(
                row=i // 2 * 2 + 1, column=i % 2, sticky="ew", padx=(0, 12), pady=(0, 6)
            )
            g.columnconfigure(i % 2, weight=1)

        tk.Button(
            inner,
            text="Process recording",
            bg=PRIMARY,
            fg=BG,
            relief="flat",
            font=_font(11, medium=True),
            padx=20,
            pady=10,
            command=self.process,
        ).pack(anchor="w", padx=24, pady=16)
        return wrap

    def _build_mixer(self) -> tk.Frame:
        wrap = tk.Frame(self.body, bg=BG)
        plots = tk.Frame(wrap, bg=BORDER)
        plots.pack(fill="x")
        self.spec_plot = SpectrumCanvas(plots, height=150)
        self.spec_plot.pack(side="left", fill="both", expand=True, padx=(0, 1))
        self.gram_plot = SpecgramCanvas(plots, height=150)
        self.gram_plot.pack(side="left", fill="both", expand=True)

        bar = tk.Frame(wrap, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        bar.pack(fill="x")
        self.mode_label = tk.Label(bar, text="Measurement / Listening", bg=CARD, fg=MUTED, font=_font(9, mono=True))
        self.mode_label.pack(side="left", padx=16, pady=8)
        for text, cmd in (
            ("Honest", lambda: self.set_preview("honest")),
            ("Balanced", lambda: self.set_preview("balanced")),
            ("Mic1 L / Mic2 R", lambda: self.set_layout("ab")),
            ("Sum", lambda: self.set_layout("all")),
            ("Reset auto", lambda: self.reset_gains("auto")),
            ("Reset original", lambda: self.reset_gains("original")),
        ):
            tk.Button(bar, text=text, bg=INPUT, fg=FG, relief="flat", padx=10, pady=6, command=cmd).pack(
                side="left", padx=3, pady=6
            )
        tk.Button(bar, text="Export mix", bg=PRIMARY, fg=BG, relief="flat", padx=12, pady=6, command=self.export_mix).pack(
            side="right", padx=12
        )
        tk.Button(bar, text="Export stems", bg=INPUT, fg=FG, relief="flat", padx=10, pady=6, command=self.export_stems).pack(
            side="right"
        )

        self.trim_label = tk.Label(wrap, text="", bg=BG, fg=WARN, font=_font(9, mono=True), anchor="w")
        self.trim_label.pack(fill="x", padx=16)

        self.mixer_host = tk.Frame(wrap, bg=BG)
        self.mixer_host.pack(fill="both", expand=True)
        self.mixer_canvas = tk.Canvas(self.mixer_host, bg=BG, highlightthickness=0)
        mscroll = ttk.Scrollbar(self.mixer_host, orient="vertical", command=self.mixer_canvas.yview)
        self.mixer_inner = tk.Frame(self.mixer_canvas, bg=BG)
        self.mixer_inner.bind(
            "<Configure>", lambda e: self.mixer_canvas.configure(scrollregion=self.mixer_canvas.bbox("all"))
        )
        self.mixer_canvas.create_window((0, 0), window=self.mixer_inner, anchor="nw")
        self.mixer_canvas.configure(yscrollcommand=mscroll.set)
        self.mixer_canvas.pack(side="left", fill="both", expand=True)
        mscroll.pack(side="right", fill="y")
        return wrap

    def _build_report(self) -> tk.Frame:
        wrap = tk.Frame(self.body, bg=BG)
        canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        scroll = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.report_inner = inner

        card = self._card(inner)
        card.pack(fill="x", padx=24, pady=16)
        tk.Label(card, text="DSP SELF-TEST", bg=CARD, fg=MUTED, font=_font(8, mono=True)).pack(
            anchor="w", padx=16, pady=(12, 4)
        )
        tk.Label(
            card,
            text="Known tones at 0.1, 0.5, 3.7, and 19 Hz, time-compressed 200×.",
            bg=CARD,
            fg=MUTED,
            font=_font(9),
        ).pack(anchor="w", padx=16)
        tk.Button(
            card, text="Run self-test", bg=INPUT, fg=FG, relief="flat", padx=12, pady=6, command=self.do_selftest
        ).pack(anchor="w", padx=16, pady=10)
        self.selftest_box = tk.Text(
            card, height=12, bg=INPUT, fg=FG, relief="flat", font=_font(9, mono=True), wrap="word"
        )
        self.selftest_box.pack(fill="x", padx=16, pady=(0, 12))

        self.report_text = tk.Text(
            inner, height=28, bg=CARD, fg=MEASURE, relief="flat", font=_font(9, mono=True), wrap="word"
        )
        self.report_text.pack(fill="both", expand=True, padx=24, pady=(0, 8))
        btnrow = tk.Frame(inner, bg=BG)
        btnrow.pack(anchor="w", padx=24, pady=(0, 20))
        tk.Button(btnrow, text="Save report files", bg=PRIMARY, fg=BG, relief="flat", padx=12, pady=8, command=self.export_report).pack(
            side="left"
        )
        return wrap

    # ----- navigation / banners -----

    def _show(self, tab: str) -> None:
        if tab in ("setup", "mixer", "report") and not self.source and tab != "report":
            tab = "load"
        if tab == "mixer" and not self.result:
            tab = "setup" if self.source else "load"
        self.tab = tab
        for name, fr in self.screens.items():
            if name == tab:
                fr.lift()
        for k, b in self.nav_btns.items():
            b.configure(fg=FG if k == tab else MUTED, bg=INPUT if k == tab else BG)
        self._refresh_banner()
        self._refresh_status()
        if tab == "setup":
            self._refresh_setup()
        if tab == "mixer":
            self._refresh_mixer()
        if tab == "report":
            self._refresh_report()

    def _refresh_banner(self) -> None:
        if self.error:
            self.banner.configure(text=self.error, bg="#2a1816", fg=DANGER)
        else:
            self.banner.configure(text=self.notice, bg=INPUT, fg=MUTED)
        if self.processing and self.progress:
            self.prog_wrap.pack(fill="x", after=self.banner)
            p = self.progress
            self.prog_label.configure(
                text=f"Stage {p['stage_index'] + 1} of {p['stage_count']}: {p['stage']}   {p['detail']}   {p['fraction'] * 100:.0f}%"
            )
            self.prog_bar["value"] = p["fraction"] * 100
        else:
            self.prog_wrap.pack_forget()

    def _refresh_status(self) -> None:
        if not self.source:
            self.status.configure(text="No source loaded · WAV / FLAC / AIFF")
            return
        s = self.source
        extra = ""
        if self.result:
            extra = f"  ·  cache {format_bytes(sum(c.nbytes for c in self.result.channels))}"
        self.status.configure(
            text=(
                f"Source {format_duration(s.duration)} {s.channels}ch · {s.sample_rate:,} Hz · {s.sample_format}"
                f"   Analysis {format_hz(self.f_min)}–{format_hz(self.f_max)} · {self.working_rate} Hz working"
                f"   Speed {self.speed:g}×{extra}"
            )
        )

    def _read_channel_names(self) -> None:
        self.channel_names = [v.get().strip() or f"Channel {i + 1}" for i, v in enumerate(self.ch_vars)]

    def _apply_setup_vars(self) -> None:
        try:
            self.f_min = float(self.var_fmin.get())
            self.f_max = float(self.var_fmax.get())
            self.speed = float(self.var_speed.get())
        except ValueError as exc:
            raise ValueError("Frequency and speed must be numbers.") from exc
        self.preset = self.var_preset.get()  # type: ignore[assignment]
        self.gain_mode = self.var_gain.get()  # type: ignore[assignment]
        mapping = {"full": None, "30s test": 30.0, "2 min test": 120.0, "10 min test": 600.0}
        self.test_duration = mapping.get(self.var_test.get())
        self.bands = preset_bands(self.preset, self.f_min, self.f_max, self.bands)  # type: ignore[arg-type]
        if self.working_auto:
            self.working_rate = auto_working_rate(self.f_max)
        self._read_channel_names()
        for k, var in self.hw_vars.items():
            self.hardware[k] = var.get()

    def _refresh_setup(self) -> None:
        if not self.source:
            return
        s = self.source
        self.setup_name.configure(text=s.name)
        self.setup_meta.configure(
            text=(
                f"{format_duration(s.duration)}   {s.channels} ch   {s.sample_rate:,} Hz   {s.format} · {s.sample_format}   {format_bytes(s.bytes)}\n"
                f"Peak  " + "   ".join(format_db(p) for p in s.peaks)
            )
        )
        try:
            self._apply_setup_vars()
        except ValueError:
            return
        safety = speed_safety(self.f_min, self.f_max, self.speed, s.duration, DEFAULT_OUTPUT_RATE)
        self.warn_label.configure(text="\n".join(safety["warnings"]))

    # ----- load / process -----

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open recording",
            filetypes=[
                (
                    "Lossless audio",
                    "*.flac *.FLAC *.wav *.WAV *.rf64 *.RF64 *.aiff *.AIFF *.aif *.AIF *.w64 *.W64 *.caf *.CAF",
                ),
                ("FLAC", "*.flac *.FLAC"),
                ("WAV / RF64", "*.wav *.WAV *.rf64 *.RF64"),
                ("AIFF", "*.aiff *.AIFF *.aif *.AIF"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._bg(lambda: self._load_path(path), "Reading file…")

    def load_synthetic(self, kind: str) -> None:
        self._bg(lambda: self._load_synth(kind), "Generating session…")

    def _load_synth(self, kind: str) -> None:
        channels, rate, spec = generate_synthetic(kind)  # type: ignore[arg-type]
        self.raw = channels
        self.raw_rate = rate
        self.source = source_from_synthetic(kind, channels, rate)  # type: ignore[arg-type]
        n = len(channels)
        self.channel_names = (
            ["Metal Marshmallow #1", "Metal Marshmallow #2"] if n == 2 else [f"Channel {i + 1}" for i in range(n)]
        )
        for i, var in enumerate(self.ch_vars):
            var.set(self.channel_names[i] if i < n else f"Channel {i + 1}")
        self.result = None
        self.notice = f"Loaded {spec.title}. Generated in memory — no original recording is modified."
        self.error = None
        self.log = [f"Loaded synthetic {spec.title} ({self.source.duration:.0f}s @ {rate} Hz)"]
        self.after(0, lambda: self._show("setup"))

    def _load_path(self, path: str) -> None:
        max_sec = self.test_duration
        channels, rate, info = decode_audio(
            path,
            max_seconds=max_sec,
            ingest_rate=INGEST_RATE_CAP,
            on_progress=lambda f: self._set_progress("Validate input", 0, f, os.path.basename(path)),
        )
        note = " The source file is read-only and will never be overwritten."
        if rate != info.sample_rate:
            note = (
                f" Ingested at {rate} Hz (source was {info.sample_rate} Hz)."
                " Original file is not modified."
            )
        self.raw = channels
        self.raw_rate = rate
        self.source = source_from_file(path, channels, rate, info)
        n = len(channels)
        self.channel_names = [f"Channel {i + 1}" for i in range(n)]
        for i, var in enumerate(self.ch_vars):
            var.set(self.channel_names[i] if i < n else f"Channel {i + 1}")
        self.result = None
        extra = ""
        ext = os.path.splitext(path)[1].lower()
        if ext in {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma"}:
            extra = " Lossy source — prefer the original WAV/FLAC for low-frequency work."
        self.notice = f"Loaded {os.path.basename(path)}.{note}{extra}"
        self.error = None
        self.log = [f"Loaded {path}"]
        self.after(0, lambda: self._show("setup"))

    def unload(self) -> None:
        self._stop_play()
        self.raw = None
        self.raw_rate = 0
        self.source = None
        self.result = None
        self.error = None
        self.notice = "No source loaded. Original files are never overwritten."
        self._show("load")

    def process(self) -> None:
        if not self.raw or not self.source:
            self.error = "Load a recording or a synthetic session first."
            self._refresh_banner()
            return
        try:
            self._apply_setup_vars()
        except ValueError as e:
            self.error = str(e)
            self._refresh_banner()
            return
        nyq = self.raw_rate / 2
        if self.f_max > nyq:
            self.error = f"The input Nyquist frequency is {nyq} Hz. Frequencies above {nyq} Hz were never recorded."
            self._refresh_banner()
            return

        raw = self.raw
        rate = self.raw_rate
        cfg = ProcessConfig(
            bands=self.bands,
            f_min=self.f_min,
            f_max=self.f_max,
            working_rate=self.working_rate,
            speed=self.speed,
            output_rate=DEFAULT_OUTPUT_RATE,
            gain_mode=self.gain_mode,  # type: ignore[arg-type]
            test_duration=self.test_duration,
            channel_names=self.channel_names,
        )

        def work() -> None:
            result = process_channels(
                raw,
                rate,
                cfg,
                on_progress=lambda p: self.after(0, lambda p=p: self._progress_dict(p)),
            )
            self.result = result
            n_ch = len(result.band_audio)
            n_b = len(result.band_audio[0]) if n_ch else 0
            self.gains = [row[:] for row in result.suggested_gains]
            self.mute = [[False] * n_b for _ in range(n_ch)]
            self.solo = [[False] * n_b for _ in range(n_ch)]
            mix0 = mix_bands(result.band_audio[0], self.gains[0], self.mute[0], self.solo[0])
            compressed = time_compress(mix0, result.working_rate, self.speed, DEFAULT_OUTPUT_RATE)
            trim, _ = master_trim(compressed, HEADROOM_DB)
            self.master_trim_db = trim
            self.log.extend(result.log)
            self.notice = "Processing complete. Mixing is cheap — gain changes do not re-read the source."
            self.error = None
            self.mix_cache = None
            self.after(0, lambda: self._show("mixer"))

        self._bg(work, "Processing…")

    def _set_progress(self, stage: str, index: int, fraction: float, detail: str) -> None:
        self.progress = {
            "stage": stage,
            "stage_index": index,
            "stage_count": 7,
            "fraction": fraction,
            "detail": detail,
            "elapsed_ms": 0,
        }
        self.after(0, self._refresh_banner)

    def _progress_dict(self, p: dict) -> None:
        self.progress = p
        self._refresh_banner()

    def _bg(self, fn: Callable[[], None], notice: str) -> None:
        if self.processing:
            return
        self.processing = True
        self.notice = notice
        self.error = None
        self._refresh_banner()

        def runner() -> None:
            try:
                fn()
            except Exception as exc:
                self.error = str(exc)
                self.notice = ""
                self.after(0, self._refresh_banner)
            finally:
                self.processing = False
                self.progress = None
                self.after(0, self._refresh_banner)

        threading.Thread(target=runner, daemon=True).start()

    # ----- mixer -----

    def _live_bands(self) -> list[Band]:
        return [b for b in self.bands if b.enabled and b.hi > b.lo]

    def set_preview(self, mode: str) -> None:
        self.preview_mode = mode
        self.mix_cache = None
        self._refresh_mixer()

    def set_layout(self, mode: str) -> None:
        self.layout_mode = mode
        self.mix_cache = None

    def reset_gains(self, to: str) -> None:
        if not self.result:
            return
        if to == "original":
            self.gains = [[0.0 for _ in row] for row in self.result.suggested_gains]
            self.preview_mode = "honest"
        else:
            self.gains = [row[:] for row in self.result.suggested_gains]
            self.preview_mode = "balanced"
        self.mix_cache = None
        self._refresh_mixer()

    def _refresh_mixer(self) -> None:
        for w in self.mixer_inner.winfo_children():
            w.destroy()
        if not self.result:
            tk.Label(self.mixer_inner, text="Process a recording to open the mixer.", bg=BG, fg=MUTED).pack(pady=40)
            return
        live = self._live_bands()
        spec = self.result.spectrum[0] if self.result.spectrum else None
        self.spec_plot.set_data(
            spec.freqs if spec else None,
            spec.mean_dbfs if spec else None,
            self.f_min,
            self.f_max,
            live,
        )
        self.gram_plot.set_data(self.result.channels[0] if self.result.channels else None, self.result.working_rate, self.f_min, self.f_max)
        self.mode_label.configure(
            text=("LISTENING" if self.preview_mode == "balanced" else "MEASUREMENT")
            + f"   ·   layout {self.layout_mode}"
        )
        if self.master_trim_db < -0.1:
            self.trim_label.configure(
                text=f"Master trim {format_db(self.master_trim_db)} applied for headroom. No limiter. No compressor."
            )
        else:
            self.trim_label.configure(text="")

        for ci, ms in enumerate(self.result.measurements):
            block = self._card(self.mixer_inner)
            block.pack(fill="x", padx=16, pady=8)
            name = self.channel_names[ci] if ci < len(self.channel_names) else f"Channel {ci + 1}"
            tk.Label(block, text=name, bg=CARD, fg=FG, font=_font(11, medium=True)).pack(anchor="w", padx=12, pady=(10, 4))
            for bi, m in enumerate(ms):
                row = tk.Frame(block, bg=CARD)
                row.pack(fill="x", padx=12, pady=4)
                b = live[bi] if bi < len(live) else Band(0, 0, "?")
                tk.Label(row, text=format_band(b.lo, b.hi), bg=CARD, fg=FG, width=14, anchor="w", font=_font(9, mono=True)).pack(
                    side="left"
                )
                tk.Label(
                    row,
                    text=f"{CLASS_LABEL.get(m.classification, m.classification)}  {format_db(m.rms_dbfs)}",
                    bg=CARD,
                    fg=MUTED,
                    width=22,
                    anchor="w",
                    font=_font(8, mono=True),
                ).pack(side="left")
                gain = self.gains[ci][bi] if self.preview_mode == "balanced" else 0.0
                var = tk.DoubleVar(value=gain)
                cap = GAIN_CAPS.get(self.gain_mode, 48)  # type: ignore[arg-type]

                def on_slide(val: str, c=ci, b=bi, v=var) -> None:
                    self.gains[c][b] = float(val)
                    self.preview_mode = "balanced"
                    self.mix_cache = None

                sc = tk.Scale(
                    row,
                    from_=0,
                    to=cap,
                    orient="horizontal",
                    resolution=0.5,
                    bg=CARD,
                    fg=FG,
                    troughcolor=INPUT,
                    highlightthickness=0,
                    length=280,
                    variable=var,
                    command=on_slide,
                    showvalue=0,
                )
                sc.pack(side="left", padx=8)
                val_lbl = tk.Label(row, text=format_db(gain), bg=CARD, fg=ACCENT, width=9, font=_font(8, mono=True))
                val_lbl.pack(side="left")

                def bind_lbl(v=var, lbl=val_lbl):
                    lbl.configure(text=format_db(float(v.get())))

                var.trace_add("write", lambda *_a, fn=bind_lbl: fn())

                mute_v = tk.BooleanVar(value=self.mute[ci][bi])
                solo_v = tk.BooleanVar(value=self.solo[ci][bi])

                def tog_mute(_=None, c=ci, b=bi, v=mute_v) -> None:
                    self.mute[c][b] = bool(v.get())
                    self.mix_cache = None

                def tog_solo(_=None, c=ci, b=bi, v=solo_v) -> None:
                    self.solo[c][b] = bool(v.get())
                    self.mix_cache = None

                tk.Checkbutton(
                    row, text="M", variable=mute_v, bg=CARD, fg=FG, selectcolor=INPUT, command=tog_mute
                ).pack(side="left")
                tk.Checkbutton(
                    row, text="S", variable=solo_v, bg=CARD, fg=FG, selectcolor=INPUT, command=tog_solo
                ).pack(side="left")
                if m.low_confidence:
                    tk.Label(row, text="low conf", bg=CARD, fg=WARN, font=_font(8, mono=True)).pack(side="left", padx=6)

    def _current_mix(self) -> list[np.ndarray]:
        if self.mix_cache is not None:
            return self.mix_cache
        assert self.result
        n_ch = len(self.result.band_audio)
        outs: list[np.ndarray] = []
        for ci in range(n_ch):
            g = self.gains[ci] if self.preview_mode == "balanced" else [0.0] * len(self.gains[ci])
            mixed = mix_bands(self.result.band_audio[ci], g, self.mute[ci], self.solo[ci])
            out = time_compress(mixed, self.result.working_rate, self.speed, DEFAULT_OUTPUT_RATE)
            master_trim(out, HEADROOM_DB)
            outs.append(out)
        if self.layout_mode == "ab" and len(outs) >= 2:
            n = min(outs[0].size, outs[1].size)
            self.mix_cache = [outs[0][:n], outs[1][:n]]
        elif len(outs) == 1:
            self.mix_cache = [outs[0], outs[0]]
        else:
            stacked = np.mean(np.stack(outs), axis=0)
            self.mix_cache = [stacked, stacked]
        self.mix_duration = self.mix_cache[0].size / DEFAULT_OUTPUT_RATE
        return self.mix_cache

    def toggle_play(self) -> None:
        if self.playing:
            self._stop_play()
            return
        if not self.result:
            return
        mix = self._current_mix()
        fd, path = tempfile.mkstemp(prefix="infraexplorer_", suffix=".wav")
        os.close(fd)
        encode_wav32f(mix, DEFAULT_OUTPUT_RATE, path)
        if self.play_path and os.path.exists(self.play_path):
            try:
                os.remove(self.play_path)
            except OSError:
                pass
        self.play_path = path
        self.play_offset = 0.0
        self.play_started = time.time()
        self.playing = True
        self.play_btn.configure(text="Pause")
        _play_wav(path)

    def _stop_play(self) -> None:
        self.playing = False
        self.play_btn.configure(text="Play")
        _stop_wav()
        if self.play_started:
            self.play_offset = min(self.mix_duration, time.time() - self.play_started)

    def toggle_loop(self) -> None:
        self.loop = not self.loop
        self.loop_btn.configure(fg=FG if self.loop else MUTED, bg=INPUT)

    def _tick(self) -> None:
        if self.playing:
            t = min(self.mix_duration, time.time() - self.play_started)
            self.time_label.configure(text=f"{format_duration(t)} / {format_duration(self.mix_duration)}   orig {format_duration(t * self.speed)}")
            if t >= self.mix_duration - 0.05:
                if self.loop:
                    if self.play_path:
                        _play_wav(self.play_path)
                    self.play_started = time.time()
                else:
                    self._stop_play()
                    self.time_label.configure(text=f"{format_duration(self.mix_duration)} / {format_duration(self.mix_duration)}")
        elif self.result:
            self.time_label.configure(text=f"{format_duration(self.play_offset)} / {format_duration(self.mix_duration)}")
        self.after(80, self._tick)

    # ----- export / report -----

    def export_mix(self) -> None:
        if not self.result or not self.source:
            return
        folder = filedialog.askdirectory(title="Save mix WAV to folder")
        if not folder:
            return
        mix = self._current_mix()
        suffix = "MIC1-L_MIC2-R_BALANCED" if self.layout_mode == "ab" else (
            "HONEST" if self.preview_mode == "honest" else "BALANCED"
        )
        name = f"{safe_filename(self.source.name)}_{suffix}_{int(self.speed)}x.wav"
        encode_wav32f(mix, DEFAULT_OUTPUT_RATE, os.path.join(folder, name))
        self.notice = f"Wrote {name}"
        self._refresh_banner()

    def export_stems(self) -> None:
        if not self.result or not self.source:
            return
        folder = filedialog.askdirectory(title="Save stems to folder")
        if not folder:
            return
        live = self._live_bands()
        count = 0
        for ci, ch_bands in enumerate(self.result.band_audio):
            for bi, band in enumerate(ch_bands):
                gained = band * (10 ** ((self.gains[ci][bi] if self.preview_mode == "balanced" else 0) / 20))
                out = time_compress(gained, self.result.working_rate, self.speed, DEFAULT_OUTPUT_RATE)
                master_trim(out, HEADROOM_DB)
                b = live[bi]
                name = (
                    f"{safe_filename(self.source.name)}_"
                    f"{safe_filename(self.channel_names[ci] if ci < len(self.channel_names) else f'MIC{ci+1}')}_"
                    f"{b.lo}to{b.hi}Hz_{int(self.speed)}x.wav"
                )
                encode_wav32f([out], DEFAULT_OUTPUT_RATE, os.path.join(folder, name))
                count += 1
        self.notice = f"Wrote {count} stem WAVs"
        self._refresh_banner()

    def _config(self) -> ProcessConfig:
        return ProcessConfig(
            bands=self._live_bands(),
            f_min=self.f_min,
            f_max=self.f_max,
            working_rate=self.working_rate,
            speed=self.speed,
            output_rate=DEFAULT_OUTPUT_RATE,
            gain_mode=self.gain_mode,  # type: ignore[arg-type]
            test_duration=self.test_duration,
            channel_names=self.channel_names,
        )

    def export_report(self) -> None:
        if not self.result or not self.source:
            messagebox.showinfo(APP_NAME, "Process a recording first.")
            return
        folder = filedialog.askdirectory(title="Save analysis files")
        if not folder:
            return
        cfg = self._config()
        live = self._live_bands()
        base = safe_filename(self.source.name)
        with open(os.path.join(folder, f"{base}_analysis.csv"), "w", encoding="utf-8") as f:
            f.write(analysis_csv(self.source, cfg, self.result.measurements, live))
        payload = {
            "application": APP_NAME,
            "version": APP_VERSION,
            "dsp": DSP_VERSION,
            "schema": PROJECT_SCHEMA,
            "source": {
                "name": self.source.name,
                "kind": self.source.kind,
                "sampleRate": self.source.sample_rate,
                "channels": self.source.channels,
                "duration": self.source.duration,
            },
            "hardware": self.hardware,
            "gains": self.gains,
        }
        import json

        with open(os.path.join(folder, f"{base}_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(os.path.join(folder, f"{base}_processing_log.txt"), "w", encoding="utf-8") as f:
            f.write(
                processing_log_text(
                    self.source,
                    cfg,
                    self.result.integrity,
                    self.result.measurements,
                    live,
                    self.result.reconstruction_db,
                    self.gains,
                    self.log,
                )
            )
        self.notice = f"Wrote CSV, JSON, and log next to {base}"
        self._refresh_banner()

    def _refresh_report(self) -> None:
        self.report_text.delete("1.0", "end")
        if not self.source:
            self.report_text.insert("end", "Load a session to generate a processing report.\n")
            return
        if not self.result:
            self.report_text.insert("end", "Process the recording to fill this report.\n")
            return
        text = processing_log_text(
            self.source,
            self._config(),
            self.result.integrity,
            self.result.measurements,
            self._live_bands(),
            self.result.reconstruction_db,
            self.gains,
            self.log,
        )
        self.report_text.insert("end", text)

    def do_selftest(self) -> None:
        def work() -> None:
            cases = run_self_test()
            self.selftest = cases
            lines = []
            for c in cases:
                mark = "PASS" if c["pass"] else "FAIL"
                lines.append(f"[{mark}]  {c['name']}\n         {c['detail']}")
            text = "\n".join(lines)
            self.after(0, lambda: self._fill_selftest(text))

        self._bg(work, "Running DSP self-test…")
        self._show("report")

    def _fill_selftest(self, text: str) -> None:
        self.selftest_box.delete("1.0", "end")
        self.selftest_box.insert("end", text)
        self.notice = "Self-test finished."
        self._refresh_banner()


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
