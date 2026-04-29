import pytest

from rust_assistant import __main__ as package_cli
from rust_assistant.infrastructure.entrypoints.cli import ingest as cli_ingest

pytestmark = pytest.mark.unit


def test_build_ingest_parser_exposes_env_based_in_memory_cli_contract():
    help_text = cli_ingest.build_parser().format_help()

    assert "--no-persist" in help_text
    assert "--allow-sample-persist" in help_text
    assert "--raw-dir" not in help_text
    assert "--parse-output" not in help_text
    assert "--clean-output" not in help_text
    assert "--dedup-output" not in help_text
    assert "--chunk-output" not in help_text
    assert "--chunk-dedup-output" not in help_text
    assert "--persist-postgres" not in help_text


def test_root_cli_routes_to_ingest_subcommand(monkeypatch):
    captured = {}

    def fake_run_ingest(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("rust_assistant.__main__.run_ingest", fake_run_ingest)

    assert package_cli.main(["ingest", "--stage", "discover", "--no-persist"]) == 0
    assert captured == {
        "stage": "discover",
        "persist": False,
        "crates": None,
        "limit": None,
        "allow_sample_persist": False,
        "verbose": False,
    }


def test_root_cli_routes_sample_persist_confirmation_to_ingest(monkeypatch):
    captured = {}

    def fake_run_ingest(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("rust_assistant.__main__.run_ingest", fake_run_ingest)

    assert package_cli.main(["ingest", "--limit", "10", "--allow-sample-persist"]) == 0
    assert captured["limit"] == 10
    assert captured["persist"] is True
    assert captured["allow_sample_persist"] is True


def test_root_cli_translates_ingest_errors_into_usage_errors(monkeypatch):
    def fail_run_ingest(**_kwargs):
        raise ValueError("bad options")

    monkeypatch.setattr("rust_assistant.__main__.run_ingest", fail_run_ingest)

    with pytest.raises(SystemExit):
        package_cli.main(["ingest"])


def test_root_cli_translates_tokenizer_errors_into_usage_errors(monkeypatch):
    def fail_run_ingest(**_kwargs):
        raise package_cli.IngestTokenizerUnavailableError("tokenizer unavailable")

    monkeypatch.setattr("rust_assistant.__main__.run_ingest", fail_run_ingest)

    with pytest.raises(SystemExit):
        package_cli.main(["ingest"])
