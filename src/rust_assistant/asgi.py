"""Public ASGI entrypoint."""

from rust_assistant.bootstrap.api import create_app

app = create_app()
