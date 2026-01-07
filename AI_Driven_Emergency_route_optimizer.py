# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# import re
# from datetime import datetime
# import networkx as nx
# from xgboost import XGBRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.cluster import KMeans
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# # ----------------------------
# # APP CONFIG
# # ----------------------------
# st.set_page_config(
#     page_title="AI Driven Emergency Control System",
#     layout="wide"
# )
# # ----------------------------
# # CSS STYLING FOR DARK THEME
# # ----------------------------
# st.markdown("""
# <style>
# body {
#     background-color: #000000;
#     color: #98FB98;
#     font-family: Arial, sans-serif;
# }
# h1, h2, h3, h4, h5 {
#     color: #98FB98;
# }
# div.stButton > button {
#     background-color: #1d3557;
#     color: #98FB98;
#     border-radius: 5px;
#     height: 35px;
#     width: 100%;
#     font-weight: bold;
# }
# .card {
#     border: 2px solid #98FB98;
#     border-radius: 10px;
#     padding: 15px;
#     margin-bottom: 10px;
#     background-color: #ffffff;
#     color: black;
# }
# .emergency-box {
#     border: 2px solid #98FB98;
#     border-radius: 10px;
#     padding: 15px;
#     background-color: #f0fff0;
#     color: black;
#     font-weight: bold;
# }
# .stApp {
#     background-color: #000000;
# }
# </style>
# """, unsafe_allow_html=True)

# # ----------------------------
# # SEED
# # ----------------------------
# random.seed(42)
# np.random.seed(42)

# # ----------------------------
# # AREAS & ROADS
# # ----------------------------
# AREAS = [
#     "MG Road", "Market Square", "Central Mall", "Station Road", "Town Hall",
#     "Green Park", "Ring Road East", "Ring Road West", "Lake View",
#     "City Hospital 1", "City Hospital 2", "City Hospital 3"
# ]

# ROAD_LIST = [
#     ("MG Road", "Market Square", 1.2),
#     ("Market Square", "Central Mall", 1.0),
#     ("Central Mall", "Station Road", 1.8),
#     ("Station Road", "Town Hall", 2.0),
#     ("Town Hall", "Ring Road East", 2.2),
#     ("Ring Road East", "City Hospital 2", 1.5),
#     ("MG Road", "Ring Road West", 3.0),
#     ("Ring Road West", "Ring Road East", 2.1),
#     ("Ring Road West", "Lake View", 1.9),
#     ("Lake View", "City Hospital 2", 2.0),
#     ("Central Mall", "Green Park", 1.1),
#     ("Green Park", "City Hospital 1", 1.4),
#     ("Town Hall", "City Hospital 3", 1.7)
# ]

# # ----------------------------
# # GRAPH
# # ----------------------------
# G = nx.Graph()
# for i, a in enumerate(AREAS):
#     G.add_node(i, name=a)

# for idx, (a, b, dist) in enumerate(ROAD_LIST, start=1):
#     G.add_edge(
#         AREAS.index(a),
#         AREAS.index(b),
#         distance_km=dist,
#         road_id=f"R{idx}"
#     )

# POS = nx.spring_layout(G, seed=7)

# # ----------------------------
# # DATA SIMULATION
# # ----------------------------
# def simulate_traffic():
#     rows = []
#     for ridx, (_, _, dist) in enumerate(ROAD_LIST, start=1):
#         base_speed = random.choice([30, 40, 50])
#         for hour in range(24):
#             speed = max(5, base_speed * random.uniform(0.4, 1))
#             vehicles = random.randint(10, 60)
#             congestion = (100 - (speed/base_speed*100))*0.1 + vehicles*0.05
#             rows.append({
#                 "road_id": f"R{ridx}",
#                 "hour": hour,
#                 "avg_speed": speed,
#                 "vehicle_count": vehicles,
#                 "distance_km": dist,
#                 "congestion_score": congestion
#             })
#     return pd.DataFrame(rows)

# df = simulate_traffic()

# # ----------------------------
# # MODELS
# # ----------------------------
# X = df[['hour', 'vehicle_count', 'avg_speed']]
# y = df['congestion_score']

# # Random Forest baseline
# rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
# rf_model.fit(X, y)
# df['rf_pred'] = rf_model.predict(X)

# # XGBoost
# xgb_model = XGBRegressor(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=5,
#     objective='reg:squarederror',
#     random_state=42
# )
# xgb_model.fit(X, y)
# df['xgb_pred'] = xgb_model.predict(X)

# # ----------------------------
# # MODEL ACCURACY
# # ----------------------------
# metrics = {}
# for name, preds in zip(["Random Forest", "XGBoost"], [df['rf_pred'], df['xgb_pred']]):
#     mse = mean_squared_error(y, preds)
#     rmse = np.sqrt(mse)
#     metrics[name] = {
#         "R2": r2_score(y, preds),
#         "MAE": mean_absolute_error(y, preds),
#         "RMSE": rmse
#     }

# # ----------------------------
# # KMEANS CLUSTERING
# # ----------------------------
# kmeans = KMeans(n_clusters=3, random_state=42)
# df['cluster'] = kmeans.fit_predict(df[['xgb_pred']])

# # ----------------------------
# # ROUTE FUNCTIONS
# # ----------------------------
# def shortest_path(src, dst):
#     return nx.shortest_path(G, src, dst, weight="distance_km")

# def least_traffic_path(src, dst, scores):
#     Gtemp = G.copy()
#     for u, v, attr in Gtemp.edges(data=True):
#         attr['weight'] = scores.get(attr['road_id'], 3)
#     return nx.shortest_path(Gtemp, src, dst, weight="weight")

# # ----------------------------
# # TRAFFIC OFFICERS
# # ----------------------------
# OFFICERS = [
#     {"name": "Officer Arun", "area": "MG Road / Market Square", "phone": "9998887771", "email": "arun.tc@city.gov", "duty_time": "08:00-16:00"},
#     {"name": "Officer Priya", "area": "Central Mall / Green Park", "phone": "9998887772", "email": "priya.tc@city.gov", "duty_time": "12:00-20:00"},
#     {"name": "Officer Mohan", "area": "Town Hall / Ring Road East", "phone": "9998887773", "email": "mohan.tc@city.gov", "duty_time": "06:00-14:00"},
# ]

# # ----------------------------
# # CITY MAP FUNCTION
# # ----------------------------
# def draw_city_map(shortest_path=None, least_path=None, traffic_scores=None):
#     fig, ax = plt.subplots(figsize=(4, 2))
#     fig.patch.set_facecolor('black')
#     ax.set_facecolor('black')

#     # ✅ ADD BORDER (this works even when axis is off)
#     border = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
#                            fill=False, edgecolor="#98FB98", linewidth=2)
#     ax.add_patch(border)

#     # Draw nodes
#     nx.draw_networkx_nodes(
#         G, POS,
#         node_size=50,
#         node_color="black",
#         edgecolors="#98FB98",
#         ax=ax
#     )

#     # Draw labels
#     nx.draw_networkx_labels(
#         G, POS,
#         {i: AREAS[i] for i in range(len(AREAS))},
#         font_size=4,
#         font_weight="bold",
#         font_color="white",
#         ax=ax
#     )

#     # Draw edges (traffic based)
#     edge_colors = [
#         plt.cm.viridis(min(traffic_scores.get(attr["road_id"], 3)/10, 1))
#         for _, _, attr in G.edges(data=True)
#     ] if traffic_scores else "grey"

#     nx.draw_networkx_edges(G, POS, width=2, edge_color=edge_colors, ax=ax)

#     # Highlight paths
#     if shortest_path:
#         nx.draw_networkx_edges(
#             G, POS,
#             edgelist=list(zip(shortest_path, shortest_path[1:])),
#             width=3,
#             edge_color="#00BFFF",
#             ax=ax
#         )

#     if least_path:
#         nx.draw_networkx_edges(
#             G, POS,
#             edgelist=list(zip(least_path, least_path[1:])),
#             width=3,
#             edge_color="#FF69B4",
#             style="dashed",
#             ax=ax
#         )

#     ax.set_title(
#         "City Traffic Map\nLight Blue = Shortest | Pink Dashed = Least Traffic",
#         fontsize=6,
#         color="white"
#     )

#     ax.axis("off")
#     return fig



# # ----------------------------
# # STREAMLIT UI
# # ----------------------------
# st.title("🚑 AI Driven City Emergency Control System")

# col1, col2 = st.columns([1, 2])

# with col1:
#     st.subheader("Emergency Details")
#     name = st.text_input("Patient Name")
#     mobile = st.text_input("Mobile Number")
#     condition = st.selectbox("Condition", ["severe", "normal"])
#     src = st.selectbox("Source", AREAS)
#     dst = st.selectbox(
#         "Destination Hospital",
#         [a for a in AREAS if "Hospital" in a]
#     )
#     process = st.button("🚨 PROCESS EMERGENCY")

# with col2:
#     st.subheader("Output")
#     output_box = st.empty()

# if 'emergency_info' not in st.session_state:
#     st.session_state.emergency_info = None


# # ----------------------------
# # PROCESS EMERGENCY
# # ----------------------------
# if process:
#     if not re.match("^[A-Za-z ]+$", name):
#         st.error("Invalid Name")
#     elif not re.match(r"^\d{10}$", mobile):
#         st.error("Invalid Mobile Number")
#     else:
#         now_hour = datetime.now().hour
#         current = df[df['hour'] == now_hour]
#         scores = dict(zip(current['road_id'], current['xgb_pred']))

#         src_id = AREAS.index(src)
#         dst_id = AREAS.index(dst)

#         sp = shortest_path(src_id, dst_id)
#         ltp = least_traffic_path(src_id, dst_id, scores)

#         emergency_msg = f"""
# <div class='emergency-box'>
# <h3>🚨 EMERGENCY ALERT 🚨</h3>
# <b>Patient Name:</b> {name}<br>
# <b>Mobile:</b> {mobile}<br>
# <b>Condition:</b> {condition.upper()}<br>
# <b>From:</b> {src}<br>
# <b>To:</b> {dst}<br><br>
# <b>Shortest Path:</b> {[AREAS[i] for i in sp]}<br>
# <b>Least Traffic Path:</b> {[AREAS[i] for i in ltp]}<br>
# </div>
# """
#         st.session_state.emergency_info = {
#             'message': emergency_msg,
#             'scores': scores
#         }

#         st.markdown(emergency_msg, unsafe_allow_html=True)
#         st.subheader("🗺 Emergency Route – Traffic Map")
#         fig_map = draw_city_map(sp, ltp, scores)
#         st.pyplot(fig_map)

# # ----------------------------
# # SEND NOTIFICATION BUTTON
# # ----------------------------
# if st.button("📣 Send Notification to Traffic Officers"):
#     if st.session_state.emergency_info:
#         now_hour = datetime.now().hour
#         on_duty = []
#         for officer in OFFICERS:
#             start, end = [int(t.split(":")[0]) for t in officer["duty_time"].split("-")]
#             if start <= now_hour < end:
#                 on_duty.append(officer)
#         if on_duty:
#             st.success("✅ Traffic Alerts Sent To:")
#             for o in on_duty:
#                 st.write(f"{o['name']} | Area: {o['area']} | SMS: {o['phone']} | Email: {o['email']}")
            
#             st.markdown(
#                 "<h4>📨 Message Sent:</h4>" + 
#                 st.session_state.emergency_info['message'],
#                 unsafe_allow_html=True
#             )
#         else:
#             st.warning("⚠ No officers currently on duty!")
#     else:
#         st.warning("⚠ Process an emergency first!")

# # ----------------------------
# # XGBOOST PREDICTION VS ACTUAL GRAPH
# # ----------------------------
# st.subheader("📈 XGBoost: Predicted vs Actual Congestion")

# fig, ax = plt.subplots(figsize=(8,6))
# fig.patch.set_facecolor('black')
# ax.set_facecolor('black')

# # Scatter plot
# ax.scatter(df['congestion_score'], df['xgb_pred'], color='#98FB98', s=30, label='Data Points')

# # Diagonal line for perfect prediction
# max_val = max(df['congestion_score'].max(), df['xgb_pred'].max())
# ax.plot([0, max_val], [0, max_val], color='white', linestyle='--', linewidth=2, label='Perfect Prediction')

# # Labels & Title
# ax.set_xlabel("Actual Congestion", color='white')
# ax.set_ylabel("Predicted Congestion (XGBoost)", color='white')
# ax.set_title("Predicted vs Actual Congestion", color='white')

# # Axis ticks
# ax.tick_params(colors='white')

# # Pista border for the plot
# for spine in ax.spines.values():
#     spine.set_edgecolor('#98FB98')
#     spine.set_linewidth(2)

# # Legend
# ax.legend(facecolor='black', edgecolor='#98FB98', labelcolor='white')

# # Grid
# ax.grid(True, color='#555555', linestyle='--', linewidth=0.5)

# st.pyplot(fig)


# # ---------------------------- 
# # MODEL COMPARISON DASHBOARD
# # ----------------------------
# st.subheader("📊 Model Accuracy Comparison")
# metrics_df = pd.DataFrame(metrics).T
# st.dataframe(
#     metrics_df.style
#         .format("{:.3f}")
#         .set_properties(**{
#             'color': '#98FB98',   
#             'background-color': '#000000',
#             'border': '1px solid #98FB98'
#         })
# )


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import re
from datetime import datetime
import networkx as nx
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----------------------------
# APP CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI Driven Emergency Control System",
    layout="wide"
)

# ----------------------------
# CSS STYLING FOR DARK THEME
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #000000;
    color: #98FB98;
    font-family: Arial, sans-serif;
}
h1, h2, h3, h4, h5 {
    color: #98FB98;
}
div.stButton > button {
    background-color: #1d3557;
    color: #98FB98;
    border-radius: 5px;
    height: 35px;
    width: 100%;
    font-weight: bold;
}
.card {
    border: 2px solid #98FB98;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
    background-color: #ffffff;
    color: black;
}
.emergency-box {
    border: 2px solid #98FB98;
    border-radius: 10px;
    padding: 15px;
    background-color: #f0fff0;
    color: black;
    font-weight: bold;
}
.stApp {
    background-color: #000000;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# SEED
# ----------------------------
random.seed(42)
np.random.seed(42)

# ----------------------------
# AREAS & ROADS
# ----------------------------
AREAS = [
    "MG Road", "Market Square", "Central Mall", "Station Road", "Town Hall",
    "Green Park", "Ring Road East", "Ring Road West", "Lake View",
    "City Hospital 1", "City Hospital 2", "City Hospital 3"
]

ROAD_LIST = [
    ("MG Road", "Market Square", 1.2),
    ("Market Square", "Central Mall", 1.0),
    ("Central Mall", "Station Road", 1.8),
    ("Station Road", "Town Hall", 2.0),
    ("Town Hall", "Ring Road East", 2.2),
    ("Ring Road East", "City Hospital 2", 1.5),
    ("MG Road", "Ring Road West", 3.0),
    ("Ring Road West", "Ring Road East", 2.1),
    ("Ring Road West", "Lake View", 1.9),
    ("Lake View", "City Hospital 2", 2.0),
    ("Central Mall", "Green Park", 1.1),
    ("Green Park", "City Hospital 1", 1.4),
    ("Town Hall", "City Hospital 3", 1.7)
]

# ----------------------------
# GRAPH
# ----------------------------
G = nx.Graph()
for i, a in enumerate(AREAS):
    G.add_node(i, name=a)

for idx, (a, b, dist) in enumerate(ROAD_LIST, start=1):
    G.add_edge(
        AREAS.index(a),
        AREAS.index(b),
        distance_km=dist,
        road_id=f"R{idx}"
    )

POS = nx.spring_layout(G, seed=7)

# ----------------------------
# DATA SIMULATION
# ----------------------------
def simulate_traffic():
    rows = []
    for ridx, (_, _, dist) in enumerate(ROAD_LIST, start=1):
        base_speed = random.choice([30, 40, 50])
        for hour in range(24):
            speed = max(5, base_speed * random.uniform(0.4, 1))
            vehicles = random.randint(10, 60)
            congestion = (100 - (speed/base_speed*100))*0.1 + vehicles*0.05
            rows.append({
                "road_id": f"R{ridx}",
                "hour": hour,
                "avg_speed": speed,
                "vehicle_count": vehicles,
                "distance_km": dist,
                "congestion_score": congestion
            })
    return pd.DataFrame(rows)

df = simulate_traffic()

# ----------------------------
# MODELS
# ----------------------------
X = df[['hour', 'vehicle_count', 'avg_speed']]
y = df['congestion_score']

rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X, y)
df['rf_pred'] = rf_model.predict(X)

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='reg:squarederror',
    random_state=42
)
xgb_model.fit(X, y)
df['xgb_pred'] = xgb_model.predict(X)

# ----------------------------
# MODEL ACCURACY
# ----------------------------
metrics = {}
for name, preds in zip(["Random Forest", "XGBoost"], [df['rf_pred'], df['xgb_pred']]):
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    metrics[name] = {
        "R2": r2_score(y, preds),
        "MAE": mean_absolute_error(y, preds),
        "RMSE": rmse
    }

# ----------------------------
# KMEANS CLUSTERING
# ----------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['xgb_pred']])

# ----------------------------
# ROUTE FUNCTIONS (ONLY CHANGE HERE)
# ----------------------------
def shortest_path(src, dst):
    return nx.shortest_path(G, src, dst, weight="distance_km")

def least_traffic_path(src, dst, scores):
    Gtemp = G.copy()
    for u, v, attr in Gtemp.edges(data=True):
        congestion = scores.get(attr['road_id'], 3)
        distance = attr['distance_km']
        attr['weight'] = distance * (1 + congestion)
    return nx.shortest_path(Gtemp, src, dst, weight="weight")

# ----------------------------
# TRAFFIC OFFICERS
# ----------------------------
OFFICERS = [
    {"name": "Officer Arun", "area": "MG Road / Market Square", "phone": "9998887771", "email": "arun.tc@city.gov", "duty_time": "08:00-16:00"},
    {"name": "Officer Priya", "area": "Central Mall / Green Park", "phone": "9998887772", "email": "priya.tc@city.gov", "duty_time": "12:00-20:00"},
    {"name": "Officer Mohan", "area": "Town Hall / Ring Road East", "phone": "9998887773", "email": "mohan.tc@city.gov", "duty_time": "06:00-14:00"},
]

# ----------------------------
# CITY MAP FUNCTION
# ----------------------------
def draw_city_map(shortest_path=None, least_path=None, traffic_scores=None):
    fig, ax = plt.subplots(figsize=(4, 2))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    border = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           fill=False, edgecolor="#98FB98", linewidth=2)
    ax.add_patch(border)

    nx.draw_networkx_nodes(G, POS, node_size=50, node_color="black",
                           edgecolors="#98FB98", ax=ax)

    nx.draw_networkx_labels(
        G, POS, {i: AREAS[i] for i in range(len(AREAS))},
        font_size=4, font_weight="bold",
        font_color="white", ax=ax
    )

    edge_colors = [
        plt.cm.viridis(min(traffic_scores.get(attr["road_id"], 3)/10, 1))
        for _, _, attr in G.edges(data=True)
    ] if traffic_scores else "grey"

    nx.draw_networkx_edges(G, POS, width=2,
                           edge_color=edge_colors, ax=ax)

    if shortest_path:
        nx.draw_networkx_edges(
            G, POS, edgelist=list(zip(shortest_path, shortest_path[1:])),
            width=3, edge_color="#00BFFF", ax=ax
        )

    if least_path:
        nx.draw_networkx_edges(
            G, POS, edgelist=list(zip(least_path, least_path[1:])),
            width=3, edge_color="#FF69B4",
            style="dashed", ax=ax
        )

    ax.set_title(
        "City Traffic Map\nLight Blue = Shortest | Pink Dashed = Least Traffic",
        fontsize=6, color="white"
    )
    ax.axis("off")
    return fig

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("🚑 AI Driven City Emergency Control System")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Emergency Details")
    name = st.text_input("Patient Name")
    mobile = st.text_input("Mobile Number")
    condition = st.selectbox("Condition", ["severe", "normal"])
    src = st.selectbox("Source", AREAS)
    dst = st.selectbox("Destination Hospital", [a for a in AREAS if "Hospital" in a])
    process = st.button("🚨 PROCESS EMERGENCY")

with col2:
    st.subheader("Output")
    output_box = st.empty()

if 'emergency_info' not in st.session_state:
    st.session_state.emergency_info = None

# ----------------------------
# PROCESS EMERGENCY
# ----------------------------
if process:
    if not re.match("^[A-Za-z ]+$", name):
        st.error("Invalid Name")
    elif not re.match(r"^\d{10}$", mobile):
        st.error("Invalid Mobile Number")
    else:
        now_hour = datetime.now().hour
        current = df[df['hour'] == now_hour]
        scores = dict(zip(current['road_id'], current['xgb_pred']))

        src_id = AREAS.index(src)
        dst_id = AREAS.index(dst)

        sp = shortest_path(src_id, dst_id)
        ltp = least_traffic_path(src_id, dst_id, scores)

        emergency_msg = f"""
<div class='emergency-box'>
<h3>🚨 EMERGENCY ALERT 🚨</h3>
<b>Patient Name:</b> {name}<br>
<b>Mobile:</b> {mobile}<br>
<b>Condition:</b> {condition.upper()}<br>
<b>From:</b> {src}<br>
<b>To:</b> {dst}<br><br>
<b>Least Traffic Path:</b> {[AREAS[i] for i in ltp]}<br>
</div>
"""
        st.session_state.emergency_info = {
            'message': emergency_msg,
            'scores': scores
        }

        st.markdown(emergency_msg, unsafe_allow_html=True)
        st.subheader("🗺 Emergency Route – Traffic Map")
        fig_map = draw_city_map(sp, ltp, scores)
        st.pyplot(fig_map)

# ----------------------------
# SEND NOTIFICATION BUTTON
# ----------------------------
if st.button("📣 Send Notification to Traffic Officers"):
    if st.session_state.emergency_info:
        now_hour = datetime.now().hour
        on_duty = []
        for officer in OFFICERS:
            start, end = [int(t.split(":")[0]) for t in officer["duty_time"].split("-")]
            if start <= now_hour < end:
                on_duty.append(officer)
        if on_duty:
            st.success("✅ Traffic Alerts Sent To:")
            for o in on_duty:
                st.write(f"{o['name']} | Area: {o['area']} | SMS: {o['phone']} | Email: {o['email']}")
            st.markdown("<h4>📨 Message Sent:</h4>" +
                        st.session_state.emergency_info['message'],
                        unsafe_allow_html=True)
        else:
            st.warning("⚠ No officers currently on duty!")
    else:
        st.warning("⚠ Process an emergency first!")

# ----------------------------
# XGBOOST PREDICTION VS ACTUAL
# ----------------------------
st.subheader("📈 XGBoost: Predicted vs Actual Congestion")

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

ax.scatter(df['congestion_score'], df['xgb_pred'],
           color='#98FB98', s=30)

max_val = max(df['congestion_score'].max(), df['xgb_pred'].max())
ax.plot([0, max_val], [0, max_val],
        color='white', linestyle='--', linewidth=2)

ax.set_xlabel("Actual Congestion", color='white')
ax.set_ylabel("Predicted Congestion", color='white')
ax.set_title("Predicted vs Actual Congestion", color='white')
ax.tick_params(colors='white')

for spine in ax.spines.values():
    spine.set_edgecolor('#98FB98')
    spine.set_linewidth(2)

ax.grid(True, color='#555555', linestyle='--', linewidth=0.5)
st.pyplot(fig)

# ----------------------------
# MODEL COMPARISON DASHBOARD
# ----------------------------
st.subheader("📊 Model Accuracy Comparison")
metrics_df = pd.DataFrame(metrics).T
st.dataframe(
    metrics_df.style
        .format("{:.3f}")
        .set_properties(**{
            'color': '#98FB98',
            'background-color': '#000000',
            'border': '1px solid #98FB98'
        })
)

