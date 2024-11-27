import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# Initial Data (you can expand or use an API later)
data = {
    "Name": ["Lousiana cafe", "Cafe mappe", "Cafe Anne"],
    "Address": ["123 Main St, City", "456 Side Rd, City", "789 Another Rd, City"],
    "WiFi Speed (Mbps)": [50, 100, 25],
    "Power Availability": ["Yes", "Yes", "No"],
    "Latitude": [37.7749, 37.7849, 37.7949],
    "Longitude": [-122.4194, -122.4094, -122.4294],
}

df = pd.DataFrame(data)

# Streamlit App
st.title("Remote Work Cafes Finder")
st.write("Find cafes with WiFi and power outlets for your remote working needs.")

# Search Filter
st.sidebar.header("Filter Cafes")
city_filter = st.sidebar.text_input("City", value="City")
wifi_filter = st.sidebar.slider("Minimum WiFi Speed (Mbps)", min_value=0, max_value=200, value=20)

filtered_df = df[(df["WiFi Speed (Mbps)"] >= wifi_filter)]

# Display Cafes
st.subheader("Available Cafes")
for _, row in filtered_df.iterrows():
    st.write(f"**{row['Name']}**")
    st.write(f"📍 {row['Address']}")
    st.write(f"💻 WiFi Speed: {row['WiFi Speed (Mbps)']} Mbps")
    st.write(f"🔌 Power Availability: {row['Power Availability']}")
    st.write("---")

# Map
st.subheader("Map of Cafes")
m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=13)

for _, row in filtered_df.iterrows():
    folium.Marker(
        [row["Latitude"], row["Longitude"]],
        popup=f"{row['Name']} - WiFi: {row['WiFi Speed (Mbps)']} Mbps - Power: {row['Power Availability']}",
    ).add_to(m)

st_folium(m, width=700)

# User Submission
st.subheader("Add a Cafe")
with st.form("add_cafe_form"):
    name = st.text_input("Cafe Name")
    address = st.text_input("Address")
    wifi_speed = st.number_input("WiFi Speed (Mbps)", min_value=0)
    power = st.selectbox("Power Availability", ["Yes", "No"])
    latitude = st.number_input("Latitude")
    longitude = st.number_input("Longitude")
    submitted = st.form_submit_button("Submit")

    if submitted:
        new_data = pd.DataFrame({
            "Name": [name],
            "Address": [address],
            "WiFi Speed (Mbps)": [wifi_speed],
            "Power Availability": [power],
            "Latitude": [latitude],
            "Longitude": [longitude],
        })
    df = pd.concat([df, new_data], ignore_index=True)
    st.success("Cafe added successfully!")

# Display the list of cafes
st.subheader("List of Added Cafes")
if not st.session_state.df.empty:
    st.dataframe(st.session_state.df)
else:
    st.write("No cafes added yet.")