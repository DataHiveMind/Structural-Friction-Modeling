"""MongoDB configuration and client helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from dotenv import dotenv_values
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

DEFAULT_DATABASE_NAME = "structural_friction_modeling"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


@dataclass(frozen=True)
class MongoDBConfig:
    """Connection settings loaded from the environment or a dotenv file."""

    uri: str
    database_name: str = DEFAULT_DATABASE_NAME

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> MongoDBConfig:
        """Load settings, preferring process environment over the dotenv file."""
        values = dotenv_values(env_file or DEFAULT_ENV_FILE)

        def setting(name: str) -> str | None:
            return os.getenv(name) or values.get(name)

        uri = setting("MONGODB_URI")
        if not uri or not uri.strip():
            raise ValueError(
                "MONGODB_URI is required. Set it in the process environment "
                "or in the project's .env file."
            )

        uri_database = unquote(urlsplit(uri).path.lstrip("/")) or None
        database_name = (
            setting("MONGODB_DATABASE")
            or setting("MONGODB_DB_NAME")
            or uri_database
            or DEFAULT_DATABASE_NAME
        )
        return cls(uri=uri.strip(), database_name=database_name.strip())


class MongoDBClient:
    """Small lifecycle wrapper around PyMongo's lazy MongoClient."""

    def __init__(
        self,
        config: MongoDBConfig | None = None,
        *,
        client: MongoClient[Any] | None = None,
    ) -> None:
        self.config = config or MongoDBConfig.from_env()
        self._client = (
            client
            if client is not None
            else MongoClient(self.config.uri, serverSelectionTimeoutMS=5_000)
        )
        self._database: Database[Any] = self._client[self.config.database_name]

    @property
    def database(self) -> Database[Any]:
        """The configured MongoDB database."""
        return self._database

    def get_collection(self, name: str) -> Collection[Any]:
        """Return a collection from the configured database."""
        if not name.strip():
            raise ValueError("Collection name must not be empty.")
        return self._database[name]

    def ping(self) -> bool:
        """Check connectivity; PyMongo errors are allowed to propagate."""
        self._client.admin.command("ping")
        return True

    def close(self) -> None:
        """Close the underlying MongoDB client."""
        self._client.close()

    def __enter__(self) -> MongoDBClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
