"""Best-combination benchmark for LiveTranslate (ASR x VAD x translation).

A separate tool with its own window (bench/window.py, opened from Settings → Home →
Maintenance); the real-time
subtitle app never imports this package. It reuses ASRClient, VADProcessor and
Translator as they are, without modifying them."""
