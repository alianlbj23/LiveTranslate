"""One QApplication for the whole test session.

Several test modules create a QApplication in a module-scoped fixture. When such a
module finishes, its application object can be garbage-collected and the next module
creates a new one; qfluentwidgets' global `qconfig` does not survive that, so any later
ControlPanel failed with "wrapped C/C++ object of type QConfig has been deleted"
depending on test order. Creating the application here, before any test module is
imported, and keeping a reference makes every `QApplication.instance()` return it."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])
