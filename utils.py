import requests
import time
from datetime import datetime, timedelta
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Ollama API URL (Replace with the actual endpoint)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Function to call Ollama API
import json
import html
import requests

def call_ollama(sensor_data):
    prompt = "Provide a short overview of Ladakh, including its history, cultural and historical significance, fun facts about its unique traditions or geography, and motivational reasons to visit, such as its natural beauty, adventure opportunities, and spiritual serenity. Make it engaging, concise, and inspiring."

    payload = {
        "model": "smollm:360m", #llama3.2:1b granite3.1-moe:1b deepseek-r1:1.5b qwen2.5:0.5b tinyllama
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()

        # Unescape and extract the meaningful response
        unescaped_response = html.unescape(data.get("response", "No response provided."))
        return unescaped_response

    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
    except json.JSONDecodeError as e:
        return f"JSON decode failed: {e}. Raw response: {response.text}"

# Function to fetch sensor data and call the API
def process_data():
    # Fetch current sensor data (replace with your actual data source)
    sensor_data = {
        "pressure": 63567.55,
        "ambient_temp": -0.00,
        "humidity": 39.80,
        "ground_temp": -3.56,
        "uv_intensity": 467.27,
        "wind_direction": 68,
        "wind_direction_cardinal": "East",
        "wind_speed": 1.23,
        "rain_snow": "No"
    }

    # Call the Ollama API
    response = call_ollama(sensor_data)

    # Log or display the response
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"[{timestamp}] Generated Response:\n{response}")
    # resp_edit = response.split("</think>")[1]
    # resp_out = resp_edit.split("\n")
    # x = ""
    # for i in resp_out:
    #     if len(i) > 30:
    #         x+=i
    #         x+="\n\n"
    # out = ""
    # for i in x:
    #     if i!="*" and i!="#":
    #         out+=i

    return response

#################### LSTM
sequence_length = 30  # Use past 30 days to predict the next day
# input_size = scaled_data.shape[1]  # Number of features
hidden_size = 64
num_layers = 2
# output_size = scaled_data.shape[1]
learning_rate = 0.0001
num_epochs = 10000

class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # Get the last time step's output
        return out


def load_model():
    model_path = "weather_lstm_model.pth"
    learning_rate = 0.0001

    # Load the model and optimizer states
    checkpoint = torch.load(model_path, weights_only=True)

    # Retrieve model parameters from the checkpoint
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']
    output_size = checkpoint['output_size']

    # Reinitialize the model with the saved parameters
    model = WeatherLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    # Reinitialize the optimizer and load its state
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model


def predict_future_weather(model, scaler, daily_stats, start_date, target_date, sequence_length=7):
    """
    Predict future weather from a starting date to a target date using the LSTM model.
    
    Parameters:
        model (torch.nn.Module): The trained LSTM model.
        scaler (MinMaxScaler): The scaler used for normalizing the data.
        daily_stats (pd.DataFrame): The dataset containing historical weather data.
        start_date (str or datetime): The starting date for predictions.
        target_date (str or datetime): The ending date for predictions.
        sequence_length (int): The length of the sequence used for predictions.
    
    Returns:
        pd.DataFrame: Predicted weather data for the specified range of dates.
    """
    # Ensure dates are in datetime format
    start_date = pd.to_datetime(start_date).date()
    target_date = pd.to_datetime(target_date).date()

    # Validate that start_date and target_date are valid
    if start_date <= daily_stats.index[-1].date():
        raise ValueError("Start date must be after the last available date in the dataset.")

    if target_date <= start_date:
        raise ValueError("Target date must be after the start date.")

    # Extract numeric columns for scaling
    numeric_data = daily_stats.select_dtypes(include=[np.number])
    last_date_in_data = daily_stats.index[-1]
    sequence_data = numeric_data.loc[last_date_in_data - timedelta(days=sequence_length - 1):last_date_in_data]

    # Normalize the sequence
    sequence_data_scaled = scaler.transform(sequence_data)

    # Prepare list to store predictions
    future_predictions = []

    # Iteratively predict day by day from start_date to target_date
    current_date = start_date
    while current_date <= target_date:
        # Prepare the input tensor for the model
        sequence_tensor = torch.tensor(sequence_data_scaled, dtype=torch.float32).unsqueeze(0)  # Shape: [1, sequence_length, num_features]

        # Predict the next day's weather
        model.eval()
        with torch.no_grad():
            next_day_scaled = model(sequence_tensor).cpu().numpy()

        # Inverse transform to get the prediction in the original scale
        next_day = scaler.inverse_transform(next_day_scaled)

        # Append the prediction to the result
        future_predictions.append(next_day.flatten())

        # Update the sequence: remove the oldest day and add the predicted day
        sequence_data_scaled = np.vstack([sequence_data_scaled[1:], next_day_scaled])

        # Move to the next day
        current_date += timedelta(days=1)

    # Convert predictions into a DataFrame
    prediction_dates = pd.date_range(start=start_date, end=target_date).date
    predictions_df = pd.DataFrame(future_predictions, columns=numeric_data.columns, index=prediction_dates)

    return predictions_df

def returnvalues(start_date, target_date):
    model = load_model()
    daily_stats = pd.read_csv("daily_stats.csv", index_col=0)
    daily_stats.index = pd.to_datetime(daily_stats.index)
    numeric_data = daily_stats.select_dtypes(include=[np.number])
    scaler = MinMaxScaler()
    scaler.fit(numeric_data)

    # Generate predictions
    predictions_df = predict_future_weather(model, scaler, daily_stats, start_date, target_date)
    return predictions_df
