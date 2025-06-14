import pandas as pd
from pathlib import Path

class DataManager:
    def __init__(self, config):
        """
        Initialise le gestionnaire de données avec les chemins fournis dans la config.
        
        :param config: dictionnaire de configuration (chargé depuis config.yaml)
        """
        self.config = config
        self.proc_dir = Path(config["paths"]["data"]["submission"])
        self.data = None  # DataFrame chargé

    def load_data(self, zone: str, horizon: str) -> pd.DataFrame:
        """
        Charge les données prétraitées d'une zone et d'un horizon.
        
        :param zone: nom de la zone (ex: "madrid")
        :param horizon: granularité ("daily" ou "hourly")
        :return: DataFrame avec un index temporel
        """
        file_path = self.proc_dir / f"{zone}_processed_{horizon}.parquet"
        self.data = pd.read_parquet(file_path)
        self.data.index = pd.to_datetime(self.data.index)
        return self.data

    def filter_by_date_range(self, start_date, end_date) -> pd.DataFrame:
        """
        Filtre les données chargées selon une plage de dates.
        
        :param start_date: date de début (datetime.date)
        :param end_date: date de fin (datetime.date)
        :return: DataFrame filtré entre les deux dates
        """
        # Convertir les dates en pandas Timestamp
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Si l'index a une timezone, convertir start et end dans cette timezone
        if self.data.index.tz is not None:
            start = start.tz_localize(self.data.index.tz) if start.tzinfo is None else start.tz_convert(self.data.index.tz)
            end = end.tz_localize(self.data.index.tz) if end.tzinfo is None else end.tz_convert(self.data.index.tz)

        mask = (self.data.index >= start) & (self.data.index <= end)
        return self.data.loc[mask]
