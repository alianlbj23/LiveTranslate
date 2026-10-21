"""Opens the best-combination benchmark window (bench/window.py). Started by LiveTranslate
(Settings → Home → Maintenance) with pythonw, so no console window appears; a failure is
written to logs\\bench_window.log and shown in a message box (pythonw has no console).

Usage: pythonw -I launcher/bench_window.py"""
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    logging.basicConfig(filename=os.path.join(ROOT, "logs", "bench_window.log"), level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", encoding="utf-8")
    try:
        from bench.window import main as window_main

        return window_main()
    except Exception as e:  # shown to the user: without a console nothing else would appear
        logging.getLogger("LiveTranslate.Bench.Window").exception("The benchmark window could not start")
        from PyQt6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "最佳組合評測", f"評測視窗無法開啟：{e}\n詳細原因在 logs\\bench_window.log。")
        del app
        return 1


if __name__ == "__main__":
    sys.exit(main())
