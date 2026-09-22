import os
import logging
from flask import Flask
from backend.db import init_db
from backend.services.storage import ensure_directories
from backend.worker.runner import start_worker_thread

from backend.routes.pages import pages_bp
from backend.routes.upload import upload_bp
from backend.routes.jobs import jobs_bp
from backend.routes.results import results_bp
from backend.routes.telemetry import telemetry_bp

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"
)
logger = logging.getLogger("minab.app")


def create_app(start_worker: bool = True) -> Flask:
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates"
    )

    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB video upload limit

    # Initialize DB schema & storage
    init_db()
    ensure_directories()

    # Register routes
    app.register_blueprint(pages_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(results_bp)
    app.register_blueprint(telemetry_bp)

    # Start background worker thread if enabled and not reloader child process duplicate
    if start_worker and not os.environ.get("WERKZEUG_RUN_MAIN") == "false":
        logger.info("Initializing background worker thread...")
        start_worker_thread()

    return app


if __name__ == "__main__":
    app = create_app(start_worker=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
