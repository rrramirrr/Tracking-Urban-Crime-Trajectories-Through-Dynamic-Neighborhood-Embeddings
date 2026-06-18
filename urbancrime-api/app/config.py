from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Resolved relative to the project root so Docker and local runs both work
    data_dir: Path = Path("/data/processed")
    models_dir: Path = Path("/data/models")

    app_title: str = "UrbanCrime API"
    app_version: str = "1.0.0"

    FEATURE_COLS: list[str] = [
        "cm_transeunte", "cm_vehiculo", "cm_negocio", "cm_repartidor",
        "cm_metro", "cm_violacion", "cm_homicidio", "cm_microbus",
        "cm_lesiones", "cm_casa",
        "hr_madrugada", "hr_manana", "hr_tarde", "hr_noche",
        "ds_0", "ds_1", "ds_2", "ds_3", "ds_4", "ds_5", "ds_6",
        "log_n",
    ]


settings = Settings()
