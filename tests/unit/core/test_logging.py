import json
import logging

import pytest

from rust_assistant.core.config import LoggingSettings
from rust_assistant.core.logging import JsonFormatter, configure_logging

pytestmark = pytest.mark.unit


def test_configure_logging_sets_root_level_and_text_formatter():
    configure_logging(logging_settings=LoggingSettings(level="DEBUG", format="text"))

    root_logger = logging.getLogger()

    assert root_logger.level == logging.DEBUG
    assert root_logger.handlers
    assert not isinstance(root_logger.handlers[0].formatter, JsonFormatter)


def test_configure_logging_supports_json_formatter():
    configure_logging(logging_settings=LoggingSettings(level="INFO", format="json"))

    root_logger = logging.getLogger()
    formatter = root_logger.handlers[0].formatter
    record = logging.LogRecord(
        name="rust_assistant.tests",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    payload = json.loads(formatted)

    assert isinstance(formatter, JsonFormatter)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "rust_assistant.tests"
    assert payload["message"] == "hello"


def test_configure_logging_rejects_empty_level():
    try:
        configure_logging(logging_settings=LoggingSettings(level="   ", format="text"))
    except ValueError as exc:
        assert "LOG_LEVEL" in str(exc)
    else:
        raise AssertionError("Expected configure_logging to reject empty log levels")


def test_configure_logging_rejects_unknown_format():
    try:
        configure_logging(logging_settings=LoggingSettings(level="INFO", format="yaml"))
    except ValueError as exc:
        assert "LOG_FORMAT" in str(exc)
    else:
        raise AssertionError("Expected configure_logging to reject unknown log formats")
