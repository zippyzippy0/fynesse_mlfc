import os
import warnings
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
import librosa
import numpy as np
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning, module="osmnx")

class DataAccess:
    """
    Unified Data Access class for GIS, population, schools/hospitals, and ESC-50 audio datasets.
    """
    def __init__(self, place_name: str, latitude: float, longitude: float,
                 box_width: float = 0.1, box_height: float = 0.1):
        self.place_name = place_name
        self.latitude = latitude
        self.longitude = longitude
        self.box_width = box_width
        self.box_height = box_height

        # Bounding box
        self.north = latitude + box_height / 2
        self.south = latitude - box_height / 2
        self.west = longitude - box_width / 2
        self.east = longitude + box_width / 2
        self.bbox = (self.west, self.south, self.east, self.north)

        # Default tags
        self.default_tags = {
            "amenity": True,
            "building": True,
            "historic": True,
            "leisure": True,
            "shop": True,
            "tourism": True,
            "religion": True,
            "memorial": True,
        }

        # Data placeholders
        self.pois = None
        self.graph = None
        self.area = None
        self.nodes = None
        self.edges = None
        self.buildings = None
        self.schools = None
        self.hospitals = None
        self.population = None
        self.esc50_meta = None

    # ---------------- GIS Access ----------------
    def access_pois(self, tags=None):
        if tags is None:
            tags = self.default_tags
        self.pois = ox.features_from_bbox(self.bbox, tags)
        print(f"Retrieved {len(self.pois)} POIs")
        return self.pois

    def access_road_network(self):
        self.graph = ox.graph_from_bbox(self.bbox)
        self.nodes, self.edges = ox.graph_to_gdfs(self.graph)
        return self.nodes, self.edges

    def access_area_boundary(self):
        self.area = ox.geocode_to_gdf(self.place_name)
        return self.area

    def access_buildings(self):
        self.buildings = ox.features_from_bbox(self.bbox, tags={"building": True})
        return self.buildings

    def access_schools(self):
        tags = {"amenity": "school"}
        self.schools = ox.features_from_bbox(self.bbox, tags)
        print(f"Retrieved {len(self.schools)} schools")
        return self.schools

    def access_hospitals(self):
        tags = {"amenity": "hospital"}
        self.hospitals = ox.features_from_bbox(self.bbox, tags)
        print(f"Retrieved {len(self.hospitals)} hospitals")
        return self.hospitals

    def access_population(self, csv_path):
        self.population = pd.read_csv(csv_path)
        print(f"Loaded population data: {self.population.shape}")
        return self.population

    def access_all_data(self):
        self.access_pois()
        self.access_area_boundary()
        self.access_buildings()
        self.access_schools()
        self.access_hospitals()
        try:
            self.access_road_network()
        except Exception:
            pass

    # ---------------- ESC-50 Audio ----------------
    def access_esc50(self, meta_path: str, audio_folder: str):
        self.esc50_meta = pd.read_csv(meta_path)
        self.esc50_meta['file_path'] = self.esc50_meta['filename'].apply(
            lambda f: os.path.join(audio_folder, f)
        )
        print(f"Loaded ESC-50: {len(self.esc50_meta)} files")
        return self.esc50_meta

    def access_audio_features(self, file_path: str, sr: int = 22050):
        y, sr = librosa.load(file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        return mfccs

    # ---------------- Plotting ----------------
    def plot_city_map(self, zoom=1, show_pois=True, show_buildings=True,
                      show_roads=True, show_schools=True, show_hospitals=True):
        fig, ax = plt.subplots(figsize=(8 * zoom, 8 * zoom))

        # Area boundary
        if self.area is None:
            try:
                self.access_area_boundary()
            except Exception:
                pass
        if self.area is not None:
            self.area.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

        # Buildings
        if show_buildings and self.buildings is not None:
            self.buildings.plot(ax=ax, facecolor="lightgray", edgecolor="gray", alpha=0.6)

        # Roads
        if show_roads and self.edges is not None:
            self.edges.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=0.7)

        # POIs
        if show_pois and self.pois is not None:
            self.pois.plot(ax=ax, color="red", markersize=5, alpha=0.7)

        # Schools
        if show_schools and self.schools is not None:
            self.schools.plot(ax=ax, color="blue", markersize=10, label="Schools")

        # Hospitals
        if show_hospitals and self.hospitals is not None:
            self.hospitals.plot(ax=ax, color="green", markersize=10, label="Hospitals")

        plt.title(f"Map of {self.place_name}", fontsize=14)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.show()

# ---------------- ESC-50 Dataset for PyTorch ----------------
class ESC50Dataset(Dataset):
    def __init__(self, csv_path, audio_path, folds=None, sr=22050, n_mels=64, duration=5):
        self.df = pd.read_csv(csv_path)
        if folds is not None:
            self.df = self.df[self.df["fold"].isin(folds)]
        self.audio_path = audio_path
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.audio_path, row["filename"])
        waveform, _ = torchaudio.load(filepath)
        waveform = waveform.mean(dim=0, keepdim=True)
        num_samples = self.duration * self.sr
        if waveform.shape[1] < num_samples:
            pad_size = num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        else:
            waveform = waveform[:, :num_samples]
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_mels=self.n_mels
        )(waveform)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        return mel_spec_db, row["target"]
