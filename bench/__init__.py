"""Best-combination benchmark for LiveTranslate (ASR x VAD x translation).

A separate tool driven from the launcher menu (launcher/bench.ps1); the real-time
subtitle app never imports this package. It reuses ASRClient, VADProcessor and
Translator as they are, without modifying them."""
