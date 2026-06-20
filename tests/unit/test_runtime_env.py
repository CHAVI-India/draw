"""RuntimeEnv loading: env vars override the YAML file (12-factor / container support)."""

from __future__ import annotations

from draw_pipeline.config import load_env

_YAML = (
    "DB_URL: sqlite:///file.db\n"
    "DB_NAME: filedb\n"
    "TABLE_NAME: dicom_log\n"
    "WATCH_DIR: /from/file\n"
    "MODEL_DEF_ROOT: config_yaml\n"
)


def _write_env(tmp_path) -> str:
    p = tmp_path / "env.draw.yml"
    p.write_text(_YAML)
    return str(p)


def test_loads_from_yaml_when_no_env_vars(tmp_path, monkeypatch):
    for k in ("DB_URL", "DB_NAME", "TABLE_NAME", "WATCH_DIR", "MODEL_DEF_ROOT"):
        monkeypatch.delenv(k, raising=False)

    env = load_env(_write_env(tmp_path))

    assert env.db_url == "sqlite:///file.db"
    assert env.watch_dir == "/from/file"


def test_env_vars_take_precedence_over_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_URL", "sqlite:////app/data/draw.db.sqlite")
    monkeypatch.setenv("WATCH_DIR", "/app/incoming")

    env = load_env(_write_env(tmp_path))

    # Overridden by env vars...
    assert env.db_url == "sqlite:////app/data/draw.db.sqlite"
    assert env.watch_dir == "/app/incoming"
    # ...while unset keys still come from the file.
    assert env.db_name == "filedb"


def test_all_env_vars_means_no_file_needed(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_URL", "sqlite://")
    monkeypatch.setenv("DB_NAME", "d")
    monkeypatch.setenv("TABLE_NAME", "t")
    monkeypatch.setenv("WATCH_DIR", "/w")
    monkeypatch.setenv("MODEL_DEF_ROOT", "config_yaml")

    # Point at a non-existent path: fully env-configured, so the file is never read.
    env = load_env(str(tmp_path / "does_not_exist.yml"))

    assert env.db_name == "d"
    assert env.watch_dir == "/w"
