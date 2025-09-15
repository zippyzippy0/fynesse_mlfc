import os
import torch
import warnings
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
import librosa
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning, module="osmnx")

class DataAccess:
    def __init__(self, place_name: str, latitude: float, longitude: float,
                 box_width: float = 0.1, box_height: float = 0.1):
        self.place_name = place_name
        self.latitude = latitude
        self.longitude = longitude
        self.box_width = box_width
        self.box_height = box_height

        self.north = latitude + box_height / 2
        self.south = latitude - box_height / 2
        self.west = longitude - box_width / 2
        self.east = longitude + box_width / 2
        self.bbox = (self.west, self.south, self.east, self.north)

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

        self.pois = None
        self.graph = None
        self.area = None
        self.nodes = None
        self.edges = None
        self.buildings = None

        # new
        self.esc50_meta = None  

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

    def access_all_data(self):
        self.access_pois()
        self.access_area_boundary()
        self.access_buildings()
        try:
            self.access_road_network()
        except Exception:
            pass

    def plot_city_map(self, zoom=1, show_pois=True, show_buildings=True, show_roads=True):
        fig, ax = plt.subplots(figsize=(8 * zoom, 8 * zoom))

        if self.area is None:
            try:
                self.access_area_boundary()
            except Exception:
                pass
        if self.area is not None:
            self.area.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

        if show_buildings:
            if self.buildings is None:
                try:
                    self.access_buildings()
                except Exception:
                    pass
            if self.buildings is not None and not self.buildings.empty:
                self.buildings.plot(ax=ax, facecolor="lightgray",
                                    edgecolor="gray", alpha=0.6)

        if show_roads:
            if self.nodes is None or self.edges is None:
                try:
                    self.access_road_network()
                except Exception:
                    pass
            if self.edges is not None and not self.edges.empty:
                self.edges.plot(ax=ax, linewidth=0.5,
                                edgecolor="black", alpha=0.7)

        if show_pois:
            if self.pois is None:
                try:
                    self.access_pois()
                except Exception:
                    pass
            if self.pois is not None and not self.pois.empty:
                self.pois.plot(ax=ax, color="red", markersize=5, alpha=0.7)

        plt.title(f"Map of {self.place_name}", fontsize=14)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

    def access_esc50(self, meta_path: str, audio_folder: str):
        """Load ESC-50 metadata and attach file paths."""
        self.esc50_meta = pd.read_csv(meta_path)
        self.esc50_meta['file_path'] = self.esc50_meta['filename'].apply(
            lambda f: os.path.join(audio_folder, f)
        )
        print(f"Loaded ESC-50: {len(self.esc50_meta)} files")
        return self.esc50_meta

    def access_audio_features(self, file_path: str, sr: int = 22050):
        """Extract MFCC features from an audio file."""
        y, sr = librosa.load(file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        return mfccs

