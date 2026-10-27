import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    xtdb_host: str = "localhost"
    xtdb_port: int = 5432
    xtdb_database: str = "xtdb"
    xtdb_user: str = "xtdb"
    xtdb_password: str = ""
    memory_url: str = "http://127.0.0.1:8001"
    restate_ingress: str = "http://127.0.0.1:8080"
    scip_repo_path: str | None = None
    scip_project_name: str | None = None
    scip_cache_dir: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            xtdb_host=os.getenv("XTDB_HOST", "localhost"),
            xtdb_port=int(os.getenv("XTDB_PORT", "5432")),
            xtdb_database=os.getenv("XTDB_DATABASE", "xtdb"),
            xtdb_user=os.getenv("XTDB_USER", "xtdb"),
            xtdb_password=os.getenv("XTDB_PASSWORD", ""),
            memory_url=os.getenv("MEMORY_URL", "http://127.0.0.1:8001"),
            restate_ingress=os.getenv("RESTATE_INGRESS", "http://127.0.0.1:8080"),
            scip_repo_path=os.getenv("SCIP_REPO_PATH") or None,
            scip_project_name=os.getenv("SCIP_PROJECT_NAME") or None,
            scip_cache_dir=os.getenv("SCIP_CACHE_DIR") or None,
        )
