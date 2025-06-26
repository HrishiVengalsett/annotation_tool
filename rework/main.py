import sys
import os
import logging
from PyQt5.QtWidgets import QApplication
import traceback

# Configure logging
logging.basicConfig(
    filename='annotation_tool.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler"""
    logging.critical(
        "Unhandled exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    from PyQt5.QtWidgets import QMessageBox
    QMessageBox.critical(
        None,
        "Fatal Error",
        f"{exc_type.__name__}: {str(exc_value)}"
    )


sys.excepthook = handle_exception

if __name__ == "__main__":
    try:
        # Set memory limits on Unix-like systems
        if hasattr(os, 'setrlimit'):
            import resource

            resource.setrlimit(
                resource.RLIMIT_AS,
                (2 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024)  # 2GB
            )

        app = QApplication(sys.argv)
        app.setStyle('Fusion')

        from ui_mainwindow import AnnotatorMainWindow

        window = AnnotatorMainWindow()
        window.show()

        ret = app.exec_()
        # Cleanup on exit
        if hasattr(window, 'canvas'):
            window.canvas.clear_memory()
        sys.exit(ret)

    except Exception as e:
        logging.critical(f"Startup failed: {traceback.format_exc()}")
        raise