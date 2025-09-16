import os
import warnings
import pandas as pd
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import requests

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
        self.schools = None
        self.hospitals = None
        self.population_csv = None
        self.population_raster = None

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
        tags = {"amenity": ["hospital", "clinic"]}
        self.hospitals = ox.features_from_bbox(self.bbox, tags)
        print(f"Retrieved {len(self.hospitals)} hospitals/clinics")
        return self.hospitals

    def access_population_csv(self, csv_path):
        self.population_csv = pd.read_csv(csv_path)
        print(f"Loaded population data: {self.population_csv.shape}")
        return self.population_csv

    def access_population_raster(self, year=2020):
        url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/KEN/ken_ppp_{year}_1km_Aggregated.tif"
        output_path = f"data/population_{year}.tif"
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(output_path):
            print(f"Downloading WorldPop {year} population raster...")
            r = requests.get(url)
            with open(output_path, "wb") as f:
                f.write(r.content)
            print(f"Saved raster at {output_path}")
        self.population_raster = rasterio.open(output_path)
        return self.population_raster

    def plot_population_raster(self):
        if self.population_raster is None:
            raise ValueError("Population raster not loaded. Run access_population_raster() first.")
        show(self.population_raster, title="Population Density (WorldPop)")

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

    def plot_city_map(self, zoom=1, show_pois=True, show_buildings=True,
                      show_roads=True, show_schools=True, show_hospitals=True):
        fig, ax = plt.subplots(figsize=(8 * zoom, 8 * zoom))
        if self.area is None:
            try:
                self.access_area_boundary()
            except Exception:
                pass
        if self.area is not None:
            self.area.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)
        if show_buildings and self.buildings is not None:
            self.buildings.plot(ax=ax, facecolor="lightgray", edgecolor="gray", alpha=0.6)
        if show_roads and self.edges is not None:
            self.edges.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=0.7)
        if show_pois and self.pois is not None:
            self.pois.plot(ax=ax, color="red", markersize=5, alpha=0.7)
        if show_schools and self.schools is not None:
            self.schools.plot(ax=ax, color="blue", markersize=10, label="Schools")
        if show_hospitals and self.hospitals is not None:
            self.hospitals.plot(ax=ax, color="green", markersize=10, label="Hospitals")
        plt.title(f"Map of {self.place_name}", fontsize=14)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.show()
