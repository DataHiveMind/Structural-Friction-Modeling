from unittest.mock import MagicMock

import pytest

from db_client import MongoDBClient, MongoDBConfig


def test_config_loads_dotenv_and_defaults_database(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('MONGODB_URI="mongodb://example.test"\n')
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("MONGODB_DATABASE", raising=False)
    monkeypatch.delenv("MONGODB_DB_NAME", raising=False)

    config = MongoDBConfig.from_env(env_file)

    assert config.uri == "mongodb://example.test"
    assert config.database_name == "structural_friction_modeling"


def test_process_environment_overrides_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MONGODB_URI=mongodb://dotenv.test\nMONGODB_DATABASE=dotenv_db\n"
    )
    monkeypatch.setenv("MONGODB_URI", "mongodb://environment.test")
    monkeypatch.setenv("MONGODB_DATABASE", "environment_db")

    config = MongoDBConfig.from_env(env_file)

    assert config.uri == "mongodb://environment.test"
    assert config.database_name == "environment_db"


def test_config_uses_database_from_uri_when_not_overridden(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MONGODB_URI=mongodb://example.test/research_data\n")
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("MONGODB_DATABASE", raising=False)
    monkeypatch.delenv("MONGODB_DB_NAME", raising=False)

    config = MongoDBConfig.from_env(env_file)

    assert config.database_name == "research_data"


def test_config_requires_uri(tmp_path, monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)

    with pytest.raises(ValueError, match="MONGODB_URI is required"):
        MongoDBConfig.from_env(tmp_path / "missing.env")


def test_client_exposes_collection_and_checks_connectivity():
    pymongo_client = MagicMock()
    database = MagicMock()
    collection = MagicMock()
    pymongo_client.__getitem__.return_value = database
    database.__getitem__.return_value = collection
    config = MongoDBConfig(uri="mongodb://example.test", database_name="research")

    client = MongoDBClient(config, client=pymongo_client)

    assert client.database is database
    assert client.get_collection("prices") is collection
    assert client.ping() is True
    pymongo_client.admin.command.assert_called_once_with("ping")


def test_context_manager_closes_client():
    pymongo_client = MagicMock()
    config = MongoDBConfig(uri="mongodb://example.test")

    with MongoDBClient(config, client=pymongo_client):
        pass

    pymongo_client.close.assert_called_once_with()
