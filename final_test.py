import sqlite3
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
from flask_caching import Cache
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from utils import process_data, returnvalues

# --------------------------
# Autoencoder Model
# --------------------------
current_date = datetime.now()
date_6_days_from_now = current_date + timedelta(days=30)
target_date = date_6_days_from_now.strftime("%Y-%m-%d")
df = returnvalues(current_date, target_date)
# print(df)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --------------------------
# Preprocessing and Model Loading
# --------------------------

# Define feature names
features = ['water_pressure', 'log_ambient_temp', 'humidity', 'water_temp', 'ambient_temp']
features_plot = ['pressure_ws', 'ambientTemp_ws', 'humidity_ws', 'groundTemp_ws', 'windSpeed_ws']

# cursor.execute("""
# CREATE TABLE IF NOT EXISTS realtime_data (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
#     pressure_ws REAL,
#     ambientTemp_ws REAL,
#     humidity_ws REAL,
#     groundTemp_ws REAL,
#     windDirection_ws STRING,
#     windSpeed_ws REAL,
#     rainandsnow_ws STRING,
#     anomaly INTEGER
# )
# """)


# Load the scaler
scaler = StandardScaler()
scaler.mean_ = np.array([1.61401178, -7.11591429, 43.1693017, 1.10039844, -7.54093996])
scaler.scale_ = np.array([1.57396511, 4.15318914, 18.77094551, 0.68505492, 4.60274569])

# Load the trained Autoencoder model
model = Autoencoder(len(features))
model.load_state_dict(torch.load("autoencoder_model.pth", weights_only=True))
model.eval()

# Load precomputed training reconstruction errors
training_errors = np.load("training_errors.npy")
threshold = np.percentile(training_errors, 95)

# --------------------------
# Database Setup
# --------------------------

# Create or connect to SQLite database
conn = sqlite3.connect("realtime_data.db", check_same_thread=False)
cursor = conn.cursor()

# Create a table for storing real-time data
cursor.execute("""
CREATE TABLE IF NOT EXISTS realtime_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    valve_state STRING,
    pressure_ws REAL,
    ambientTemp_ws REAL,
    uvIntensity_ws REAL,
    humidity_ws REAL,
    groundTemp_ws REAL,
    windDirection_ws STRING,
    windSpeed_ws REAL,
    rainandsnow_ws STRING,
    anomaly INTEGER
)
""")
# --------------------------
# Evaluate Sample Function
# --------------------------

def evaluate_sample(sample_input):
    """
    Evaluate if a sample is an anomaly and identify feature contributions.
    """
    sample_df = pd.DataFrame([sample_input])
    sample_scaled = scaler.transform(sample_df[features].values)
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

    with torch.no_grad():
        reconstruction = model(sample_tensor)
        reconstruction_error = (sample_tensor - reconstruction).numpy().flatten()

    total_error = np.mean(reconstruction_error ** 2)
    anomaly = int(total_error > threshold)
    feature_contributions = dict(zip(features, reconstruction_error))

    return anomaly, feature_contributions

# --------------------------
# Dash Application
# --------------------------

# Function to clear the database table
def clear_database():
    """
    Clears the 'realtime_data' table in the database.
    """
    try:
        with sqlite3.connect("realtime_data.db") as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM realtime_data")  # Deletes all records
            conn.commit()
            print("Database cleared successfully.")
    except Exception as e:
        print(f"Error clearing database: {e}")

# --------------------------
# Dash Application
# --------------------------

app = Dash(__name__)
app.title = "Ice Stupa Viewer"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Dropdown container */
            .modern-dropdown .Select-control {
                background-color: #2a2829 !important;
                border: 1px solid #3d3d3d !important;
                border-radius: 10px !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
                transition: all 0.3s ease !important;
            }

            /* Dropdown text */
            .modern-dropdown .Select-value-label {
                color: #FFFFFF !important;
            }

            /* Placeholder text */
            .modern-dropdown .Select-placeholder {
                color: #888888 !important;
            }

            /* Dropdown menu */
            .modern-dropdown .Select-menu-outer {
                background-color: #2a2829 !important;
                border: 1px solid #3d3d3d !important;
                border-radius: 10px !important;
                margin-top: 5px !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
            }

            /* Option hover state */
            .modern-dropdown .Select-option:hover {
                background-color: #DDA853 !important;
                color: white !important;
                transition: all 0.2s ease !important;
            }

            /* Selected option */
            .modern-dropdown .Select-option.is-selected {
                background-color: #168f41 !important;
                color: white !important;
            }

            /* Focused state */
            .modern-dropdown .Select-control:hover {
                border-color: #DDA853 !important;
                box-shadow: 0 2px 12px rgba(29,185,84,0.2) !important;
            }

            /* Multi-value chips */
            .modern-dropdown .Select-value {
                background-color: #DDA853 !important;
                border: none !important;
                border-radius: 8px !important;
                color: white !important;
                padding: 2px 8px !important;
                margin: 2px !important;
            }

            /* Remove value X button */
            .modern-dropdown .Select-value-icon {
                border: none !important;
                border-right: 1px solid rgba(255,255,255,0.2) !important;
                color: white !important;
            }

            .modern-dropdown .Select-value-icon:hover {
                background-color: rgba(255,255,255,0.1) !important;
                color: white !important;
            }

            /* Dropdown arrow */
            .modern-dropdown .Select-arrow {
                border-color: #DDA853 transparent transparent !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


app.layout = html.Div(
    style={
        "backgroundColor": "#232023",
        "color": "#FFFFFF",
        "fontFamily": "Consola, monospace",
        "padding": "10",
        "margin": "0",
        "minHeight": "100vh",
        "width": "100%",
        "position": "absolute",
        "top": "0",
        "left": "0",
    },
    children=[
        html.H1(
            children=["Ice   Stupa", html.Br(), "@Phaterakh"],
            style={
                "textAlign": "left",
                "color": "#232023",
                "background": "linear-gradient(to right, #DDA853, #16404D)",
                "fontSize": "42px",
                "fontWeight": "bold",
                "fontFamily": "Courier New",
                "padding": "20px",
                "margin": "0",
                "display": "flex",
                "borderRadius": "10px",
                # "wordWrap": "break-word",
            }
        ),
        html.H3("Monthly Forecast (℃)", style={"color": "#DDA853", "margin": "0", "padding":"10px"}),
        html.Div(
            id="scrollable-panel",
            style={
                "display": "flex",
                "overflowX": "auto",
                "whiteSpace": "nowrap",
                "padding": "30px",
                "margin": "0",
                "msOverflowStyle": "none",  # Hide scrollbar for IE and Edge
                "scrollbarWidth": "none",   # Hide scrollbar for Firefox
                "background": "linear-gradient(145deg, #1e1c1f, #272528)", # 1e1c1f
                "borderRadius": "20px",
                "boxShadow": "0 10px 20px rgba(0,0,0,0.2)",
            },
            # CSS for webkit browsers (Chrome, Safari, etc.)
            className="hide-scrollbar",
            children=[
                html.Div(
                    style={
                        "minWidth": "220px",
                        "padding": "20px",
                        "marginRight": "20px",
                        "background": "linear-gradient(145deg, #DDA853, #16404D)",
                        "borderRadius": "15px",
                        "textAlign": "center",
                        "boxShadow": "8px 8px 16px #1a181b, -8px -8px 16px #2c282d",
                        "transition": "transform 0.2s ease",
                        "cursor": "pointer",
                        ":hover": {
                            "transform": "translateY(-5px)",
                        }
                    },
                    children=[
                        html.H4(index, style={
                            "color": "#232023",
                            "fontSize": "24px",
                            "marginBottom": "15px",
                            "fontWeight": "bold"
                        }),
                        html.Div([
                            html.P(f"Mean: {str(mean)[0:5]}°", style={
                                "fontSize": "18px",
                                "marginBottom": "10px",
                                "color": "#FBF5DD"
                            }),
                            html.P(f"Min: {str(min_temp)[0:5]}°", style={
                                "fontSize": "16px",
                                "marginBottom": "10px",
                                "color": "#FBF5DD"
                            }),
                            html.P(f"Max: {str(max_temp)[0:5]}°", style={
                                "fontSize": "16px",
                                "color": "#FBF5DD"
                            }),
                        ], style={
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "5px"
                        })
                    ],
                )
                for index, mean, min_temp, max_temp in zip(
                    df.index, df["AirTC_Avg_mean"], df["AirTC_Avg_min"], df["AirTC_Avg_max"]
                )
            ],
        ),
        html.Div([
            html.H3(
                "Select State of the Valve:", 
                style={
                    "color": "#DDA853", 
                    "margin": "20px 0 10px 0", 
                    "fontFamily": "Courier New",
                    "fontSize": "18px"
                }
            ),
            dcc.Dropdown(
                id="valve-state-dropdown",
                options=[{"label": "All", "value": "All"}],
                value="All",
                multi=False,
                className="modern-dropdown",
                style={
                    "backgroundColor": "#2a2829",
                    "border": "none",
                    "borderRadius": "10px",
                    "color": "#FFFFFF",
                    "fontFamily": "Courier New",
                    "marginBottom": "15px",
                }
            ),
            html.H3(
                "Select Features to View:", 
                style={
                    "color": "#DDA853", 
                    "margin": "20px 0 10px 0", 
                    "fontFamily": "Courier New",
                    "fontSize": "18px"
                }
            ),
            dcc.Dropdown(
                id="feature-dropdown",
                options=[{"label": feature, "value": feature} for feature in features_plot],
                value=["humidity_ws"],
                multi=True,
                className="modern-dropdown",
                style={
                    "backgroundColor": "#2a2829",
                    "border": "none",
                    "borderRadius": "10px",
                    "color": "#FFFFFF",
                    "fontFamily": "Courier New",
                }
            )
        ], style={"padding": "20px", "background": "linear-gradient(145deg, #1e1c1f, #272528)", "borderRadius": "15px"}),

        html.Div(
            html.Div(
                [
                    html.H3(
                        "Know your Surroundings", 
                        style={
                            "color": "#DDA853",
                            "margin": "0 0 20px 0",
                            "padding": "0",
                            "fontSize": "24px",
                            "fontWeight": "bold",
                            "fontFamily": "Courier New",
                            "textShadow": "2px 2px 4px rgba(0,0,0,0.2)"
                        }
                    ),
                    html.Div(
                        id="real-time-values",
                        style={
                            "whiteSpace": "pre-line",
                            "margin": "0",
                            "padding": "20px",
                            "backgroundColor": "#2a2829",
                            "borderRadius": "15px",
                            "boxShadow": "inset 4px 4px 8px #1a181b, inset -4px -4px 8px #2c282d",
                            "color": "#ffffff",
                            "fontFamily": "Courier New",
                            "fontSize": "16px",
                            "lineHeight": "1.6",
                            "minHeight": "100px"
                        }
                    ),
                    html.H3(
                        id="output-result",
                        style={
                            "color": "#DDA853",
                            "margin": "20px 0 0 0",
                            "padding": "15px",
                            "fontSize": "18px",
                            "fontWeight": "500",
                            "fontFamily": "Courier New",
                            "backgroundColor": "#2a2829",
                            "borderRadius": "12px",
                            "boxShadow": "4px 4px 8px #1a181b, -4px -4px 8px #2c282d",
                            "transition": "all 0.3s ease"
                        }
                    ),
                ],
                style={
                    "padding": "30px",
                    "background": "linear-gradient(145deg, #1e1c1f, #272528)",
                    "borderRadius": "20px",
                    "boxShadow": "8px 8px 16px #1a181b, -8px -8px 16px #2c282d",
                    "margin": "20px 0"
                }
            ),
            style={"margin": "0"}
        ),

        html.Div([
            html.H3(
                "Message from the land of the Lama 🧘", 
                style={
                    "color": "#DDA853",
                    "margin": "0 0 20px 0",
                    "padding": "0",
                    "fontSize": "24px",
                    "fontWeight": "bold",
                    "fontFamily": "Courier New",
                    "textShadow": "2px 2px 4px rgba(0,0,0,0.2)"
                }
            ),
            html.Div(
                id="processed-data-output",
                children="The soldiers are bringing you the message... Please Wait.. 🪖🇮🇳💗",
                style={
                    "backgroundColor": "#2a2829",
                    "padding": "25px",
                    "borderRadius": "15px",
                    "boxShadow": "inset 4px 4px 8px #1a181b, inset -4px -4px 8px #2c282d",
                    "color": "#FFFFFF",
                    "whiteSpace": "pre-line",
                    "fontFamily": "Courier New",
                    "fontSize": "16px",
                    "lineHeight": "1.6",
                    "margin": "0",
                    "transition": "all 0.3s ease",
                    "minHeight": "50px"
                }
            ),
            dcc.Interval(
                id="hourly-interval",
                interval=1 * 60 * 60 * 1000,
                n_intervals=0
            )
        ], style={
            "padding": "30px",
            "background": "linear-gradient(145deg, #1e1c1f, #272528)",
            "borderRadius": "20px",
            "boxShadow": "8px 8px 16px #1a181b, -8px -8px 16px #2c282d",
            "margin": "20px 0"
        }),

        dcc.Graph(id="realtime-graph", style={"marginTop": "20px", "margin": "0"}),

        dcc.Interval(
            id="interval-component",
            interval=1 * 1000,
            n_intervals=0
        ),
    ]
)

@app.callback(
    [Output("valve-state-dropdown", "options"),
     Output("real-time-values", "children"),
     Output("output-result", "children"),
    #  Output("output-feature-contributions", "children"),
     Output("realtime-graph", "figure")],
    [Input("interval-component", "n_intervals"),
     Input("feature-dropdown", "value"),
     Input("valve-state-dropdown", "value")]
)




# features = ['water_pressure', 'log_ambient_temp', 'humidity', 'water_temp', 'ambient_temp']


def update_real_time_values(n_intervals, selected_features, selected_valve_state):
    try:
        # Fetch data from the server
        response = requests.get("http://10.69.0.185:80/data")
        # response = requests.get("http://127.0.0.1:5000/random-instance")  # Adjust the URL if necessary
        if response.status_code == 200:
            sample_input = response.json()
            sample_input["water_pressure"] = 1.2
            sample_input['log_ambient_temp'] = -10
            sample_input['humidity'] = 45
            sample_input['water_temp'] = 1
            sample_input['ambient_temp'] = -10
            sample_input["valve_state"] = "ALL"  # Example default
            # sample_input["uv_intensity"] = sample_input.get("uv_intensity", 0)  # Default to 0 if missing

            # Evaluate anomaly
            anomaly, contributions = evaluate_sample(sample_input)

            # Insert current data into the database with the current timestamp
            current_timestamp = datetime.now().isoformat()
            with sqlite3.connect("realtime_data.db") as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO realtime_data (timestamp, pressure_ws, ambientTemp_ws, humidity_ws, groundTemp_ws, uvIntensity_ws, windDirection_ws, windSpeed_ws, rainandsnow_ws,anomaly, valve_state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (current_timestamp, sample_input['pressure_ws'], sample_input['ambientTemp_ws'], 
                      sample_input['humidity_ws'], sample_input['groundTemp_ws'], sample_input['uvIntensity_ws'], 
                      sample_input['windDirection_ws'], sample_input['windSpeed_ws'], sample_input['rainandsnow_ws'], anomaly, "ALL"))
                conn.commit()

            # Fetch accumulated data
            query = "SELECT * FROM realtime_data"
            if selected_valve_state != "All":
                query += f" WHERE valve_state='{selected_valve_state}'"
            query += " ORDER BY timestamp ASC"
            df = pd.read_sql_query(query, conn)

        # Ensure DataFrame isn't empty
        if df.empty:
            return [], "No data available.", "No data available.", "", {}

        # Prepare graph
        fig = px.line(
            df,
            x="timestamp",
            y=selected_features,
            title="",
            labels={"timestamp": "Time"},
        )
        fig.update_layout(template="plotly_dark")

        # Update dropdown options with non-null valve states
        valve_states = [{"label": "All", "value": "All"}] + [{"label": state, "value": state} for state in df["valve_state"].dropna().unique()]

        return (
            valve_states,
            f"""Ambient Temperature: {sample_input['ambientTemp_ws']}\n
            Ground Temperature: {sample_input['groundTemp_ws']}\n
            Humidity: {sample_input['humidity_ws']}\n
            Pressure: {sample_input['pressure_ws']}\n
            UV Intensity: {sample_input['uvIntensity_ws']}\n
            Mode: {sample_input['valve_state']}""",
            f"The system might be experiencing issues because of {max(contributions, key=lambda k: abs(contributions[k]))} 🤧" if anomaly else "The system is as healthy as a Marmot that stole your cookie! 🦫",
            # "\n".join([f"{k}: {v:.4f}" for k, v in contributions.items()]),
            # "\n".join([max(contributions, key=lambda k: abs(contributions[k]))]),
            fig
        )

    except Exception as e:
        print(f"Error: {e}")
        return [], "Error fetching data.", "Error fetching data.", {}



cache = Cache(app.server, config={
    "CACHE_TYPE": "SimpleCache",  # Simple in-memory cache
    "CACHE_DEFAULT_TIMEOUT": 3600  # Cache timeout in seconds (1 hour)
})

@cache.memoize()
def get_processed_data():
    """
    Process data and cache the output.
    This function will be called once per hour, and the output will be cached.
    """
    try:
        output = process_data()
        return f"{output}\nfrom team HIAL with 💗"
    except Exception as e:
        return f"Error processing data: {str(e)}"

@app.callback(
    Output("processed-data-output", "children"),
    Input("hourly-interval", "n_intervals")
)
def update_processed_data(n_intervals):
    """
    Retrieve cached data and display it.
    """
    return get_processed_data()

if __name__ == "__main__":
    clear_database()  # Clear the database before starting the app
    app.run(debug=True)



