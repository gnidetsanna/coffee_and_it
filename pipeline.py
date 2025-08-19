import os
import time
import csv
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from tqdm import tqdm
import html
from sklearn.cluster import KMeans


# Input data 
ADDRESS_FILE_RAW = "adress_file.csv"  
COFFEE_SHOPS_GEOJSON = "coffee.geojson"   

# Output data
FINAL_ANALYSIS_CSV = "it_companies_with_counts.csv"
COVERAGE_MAP_HTML = "coffee_coverage_map.html"
ALL_COFFEE_MAP_HTML = "all_coffee_shops_map.html"
FINAL_MAP_HTML = "final_map.html"

BUFFER_RADIUS_METERS = 500
N_CLUSTERS = 5

# methods

def geocode_addresses(input_csv, cache_file="office_address.csv"):
    """
    Method to geocode addresses
    """
    print("--- Step 1: Geocoding  ---")
    
    geolocator = Nominatim(user_agent="kyiv_coffee_map_project")
    geocoded_data = [["Company", "Address", "Latitude", "Longitude"]]
    
    with open(input_csv, newline='', encoding="utf-8") as infile:
        reader = list(csv.reader(infile))
        for row in tqdm(reader, desc="geocoding of addresses"):
            if len(row) < 2:
                continue
            company, address = row[0].strip(), row[1].strip().strip('"')
            lat, lon = None, None
            try:
                full_address = f"{address}, Київ"
                location = geolocator.geocode(full_address, timeout=10)
                if location:
                    lat, lon = location.latitude, location.longitude
                time.sleep(1)
            except Exception as e:
                print(f"Не вдалося обробити адресу '{address}': {e}")
            geocoded_data.append([company, address, lat, lon])
    
    geocoded_df = pd.DataFrame(geocoded_data[1:], columns=geocoded_data[0])
    geocoded_df.to_csv(cache_file, index=False, encoding='utf-8-sig')
    print(f"Geocoded addresses file: {cache_file}")
    return geocoded_df

def perform_analysis(it_gdf, coffee_gdf, buffer_radius):
    """
    Method to perform geo analysis
    """
    print("\n--- Step 2: Data analysis ---")
    
    it_gdf_proj = it_gdf.to_crs("EPSG:32636")
    coffee_gdf_proj = coffee_gdf.to_crs("EPSG:32636")
    print("1. Data to metric")

    it_buffers_gdf = it_gdf_proj.copy()
    it_buffers_gdf['geometry'] = it_buffers_gdf.geometry.buffer(buffer_radius)
    print(f"2. Created {buffer_radius}- m buffer zones.")

    joined_gdf = gpd.sjoin(it_buffers_gdf, coffee_gdf_proj, how="left", predicate="intersects")
    print("3. Joined tables.")

    coffee_counts = joined_gdf.groupby('index')['index_right'].count()
    it_gdf = it_gdf.merge(coffee_counts.rename('coffee_shop_count'), left_on='index', right_index=True, how='left')
    it_gdf['coffee_shop_count'].fillna(0, inplace=True)
    it_gdf['coffee_shop_count'] = it_gdf['coffee_shop_count'].astype(int)
    print("4. Coffee shops count completed.")
    
    return it_gdf

def apply_ml_clustering(df, n_clusters):
    """
    Method for K-Means cluster-n
    """
    print("\n--- Cluster searching  ---")
    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    if df.empty:
        print("No offices to analyze.")
        return df
    
    coords = df[['Latitude', 'Longitude']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(coords)
    df['cluster'] = kmeans.labels_
    print(f"There are {n_clusters} clusters.")
    return df

def create_coverage_map(df, output_html):
    """Method to create a coverage map """

    if df.empty:
        print("No data to create a map")
        return None
    
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    kyiv_map = folium.Map(location=map_center, zoom_start=12, tiles="cartodbpositron")

    def get_marker_color(count):
        if count == 0: return 'red'
        elif 1 <= count <= 4: return 'orange'
        else: return 'green'

    for _, row in df.iterrows():
        popup_text = f"<b>Компанія:</b> {row['Company']}<br>" \
                     f"<b>Адреса:</b> {row['Address']}<br><hr>" \
                     f"<b>Кав'ярень в радіусі {BUFFER_RADIUS_METERS}м:</b> {row['coffee_shop_count']}"
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=6,
            color=get_marker_color(row['coffee_shop_count']),
            fill=True,
            fill_color=get_marker_color(row['coffee_shop_count']),
            fill_opacity=0.8,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(kyiv_map)

    kyiv_map.save(output_html)
    print(f"The coverage map is saved to '{output_html}'")
    return kyiv_map

def create_all_coffee_shops_map(gdf, output_html):
    """Create map with all coffee shops"""

    if gdf.empty:
        print("No data to create a map")
        return None
    
    kyiv_center = [50.4501, 30.5234]
    coffee_map = folium.Map(location=kyiv_center, zoom_start=12, tiles="cartodbpositron")
    marker_cluster = MarkerCluster().add_to(coffee_map)

    for _, row in gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        centroid = row.geometry.centroid
        coords = [centroid.y, centroid.x]
        name = html.escape(str(row.get('name', 'Без назви')))
        amenity = html.escape(str(row.get('amenity', 'N/A')))
        popup_text = f"<b>Назва:</b> {name}<br><b>Тип:</b> {amenity}"
        
        folium.CircleMarker(
            location=coords,
            radius=4, color="#6F4E37", fill=True, fill_color="#A0522D", fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=250)
        ).add_to(marker_cluster)

    coffee_map.save(output_html)
    print(f"Coffee shops map is saved to '{output_html}'")
    return coffee_map

def create_final_map(df, output_html):
    """Method to create final map with intersaction zones"""

    if df.empty:
        print("No data to create a map")
        return None
    
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    kyiv_map_final = folium.Map(location=map_center, zoom_start=12, tiles="cartodbpositron")

    colors = [
        'blue', 'green', 'purple', 'orange', 'darkred',
        'lightblue', 'darkgreen', 'pink', 'cadetblue', 'darkpurple'
    ]

    def get_marker_radius(count):
        base_radius = 4
        scale_factor = 0.5
        max_radius = 15
        calculated_radius = base_radius + (count * scale_factor)
        return min(calculated_radius, max_radius)

    for _, row in df.iterrows():
        cluster_color = colors[row['cluster'] % len(colors)]
        marker_radius = get_marker_radius(row['coffee_shop_count'])
        popup_text = f"""
        <b>Компанія:</b> {row['Company']}<br>
        <b>Адреса:</b> {row['Address']}<br>
        <hr>
        <b>Кав'ярень поруч:</b> {row['coffee_shop_count']}<br>
        <b>ІТ-кластер №:</b> {row['cluster']}
        """
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=marker_radius,
            color=cluster_color,
            fill=True,
            fill_color=cluster_color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(kyiv_map_final)

    kyiv_map_final.save(output_html)
    print(f" Final map is saved to '{output_html}'")
    return kyiv_map_final

if __name__ == "__main__":

    geocoded_df = geocode_addresses(ADDRESS_FILE_RAW)

    coffee_gdf = gpd.read_file(COFFEE_SHOPS_GEOJSON, encoding='utf-8')
    coffee_gdf.dropna(subset=['geometry'], inplace=True)
    
    geocoded_df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    if geocoded_df.empty:
        print("No company is found")
        exit()
        
    it_gdf = gpd.GeoDataFrame(
        geocoded_df,
        geometry=gpd.points_from_xy(geocoded_df.Longitude, geocoded_df.Latitude),
        crs="EPSG:4326"
    )
    it_gdf.reset_index(inplace=True)

    analyzed_gdf = perform_analysis(it_gdf, coffee_gdf, BUFFER_RADIUS_METERS)

    analyzed_df = pd.DataFrame(analyzed_gdf.drop(columns='geometry')) 
    clustered_df = apply_ml_clustering(analyzed_df, N_CLUSTERS)

    create_coverage_map(clustered_df, COVERAGE_MAP_HTML) 
    create_all_coffee_shops_map(coffee_gdf, ALL_COFFEE_MAP_HTML)
    create_final_map(clustered_df, FINAL_MAP_HTML)

    print("\nAll steps completed")