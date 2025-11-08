# ===== Import Required Libraries =====
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta

# Define NASA API key and the date range (last 7 days)
API_KEY = "z9pbsYCAucfqdLaN7yNHWYOBI5aIoCGrJBQo4O3g"   # Replace this with your personal NASA API key if needed
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=7)

# NASA Near-Earth Object Web Service API endpoint
URL = (
    f"https://api.nasa.gov/neo/rest/v1/feed?"
    f"start_date={START_DATE}&end_date={END_DATE}&api_key={API_KEY}"
)

# Make a GET request to the API endpoint and retrieve the JSON response
response = requests.get(URL)
data = response.json()

# Access the dictionary containing near-Earth object data by date
neo_data = data["near_earth_objects"]

# Lists to store date-wise asteroid counts
dates = []
asteroid_counts = []

# Iterate through each date and count the number of detected asteroids
for day, asteroids in neo_data.items():
    dates.append(day)
    asteroid_counts.append(len(asteroids))

# Sort the data chronologically (important for correct plotting order)
dates, asteroid_counts = zip(*sorted(zip(dates, asteroid_counts)))

# Configure Seaborn style for better aesthetics
sns.set(style="whitegrid")

# Create a figure for visualization
plt.figure(figsize=(10, 6))

# Generate a bar plot of asteroid counts vs. dates
sns.barplot(x=dates, y=asteroid_counts, palette="mako")

# Add appropriate titles and axis labels
plt.title("Number of Near-Earth Objects Detected Per Day (Last 7 Days)\n"
          "By Revan - CODTECH Internship Task 1", fontsize=12)
plt.xlabel("Date")
plt.ylabel("Asteroid Count")

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

# Adjust layout to avoid label overlap
plt.tight_layout()

# Display the visualization
plt.show()
