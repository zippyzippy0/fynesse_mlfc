import osmnx as ox
import warnings
from typing import Dict, Optional
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

    def access_pois(self, tags: Optional[Dict] = None):
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

    def plot_city_map(self, zoom: int = 1, show_pois: bool = True,
                      show_buildings: bool = True, show_roads: bool = True):
        """
        Plot a city map with optional POIs, buildings, and road networks.

        Args:
            zoom (int): scaling factor for map size.
            show_pois (bool): if True, show Points of Interest.
            show_buildings (bool): if True, show building footprints.
            show_roads (bool): if True, show road networks.
        """
        fig, ax = plt.subplots(figsize=(8 * zoom, 8 * zoom))

        # Plot area boundary
        if self.area is None:
            try:
                self.access_area_boundary()
            except Exception:
                pass
        if self.area is not None:
            self.area.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

        # Plot buildings
        if show_buildings:
            if self.buildings is None:
                try:
                    self.access_buildings()
                except Exception:
                    pass
            if self.buildings is not None and not self.buildings.empty:
                self.buildings.plot(ax=ax, facecolor="lightgray",
                                    edgecolor="gray", alpha=0.6)

        # Plot roads
        if show_roads:
            if self.nodes is None or self.edges is None:
                try:
                    self.access_road_network()
                except Exception:
                    pass
            if self.edges is not None and not self.edges.empty:
                self.edges.plot(ax=ax, linewidth=0.5,
                                edgecolor="black", alpha=0.7)

        # Plot POIs
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


# --------------------------------------------------------------------
# Standalone wrapper function
# --------------------------------------------------------------------
def plot_city_map(place_name: str, latitude: float,
                  longitude: float, zoom: int = 1):
    """
    Standalone utility to plot a city map without manually
    creating a DataAccess instance.
    """
    da = DataAccess(place_name, latitude, longitude)
    da.plot_city_map(zoom=zoom)
