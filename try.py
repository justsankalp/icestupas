import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.title("Ice Stupa Data Analysis")

# File uploaders
st.sidebar.header("Upload CSV Files")
logger_file = st.sidebar.file_uploader("Upload Logger CSV", type="csv")
automation_file = st.sidebar.file_uploader("Upload Automation CSV", type="csv")

# Date inputs
st.sidebar.header("Select Date Range")
default_start_date = datetime(2024, 12, 6)
default_end_date = datetime(2024, 12, 10)
start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", default_end_date)

start_date = datetime.combine(start_date, datetime.min.time())
end_date = datetime.combine(end_date, datetime.min.time())

# Function definitions
def loggerproc(file):
    log = pd.read_csv(file, names=["date_time", "s", "k", "water_pressure", "ambient_temp", "humidity"])
    log = log.drop(columns=["s", "k"])
    log = log[(log["ambient_temp"].between(-40, 30)) & (log["humidity"].between(0, 100))]
    log = log[log['date_time'] != '0/0/0 0:0:0']
    log['date_time'] = pd.to_datetime(log['date_time'], format='%d/%m/%y %H:%M:%S', errors='coerce')
    log = log.dropna(subset=['date_time'])
    log.set_index("date_time", inplace=True)
    return log

def autoproc(file):
    auto = pd.read_csv(file, usecols=range(5), names=["date_time", "water_temp", "ambient_temp", "s", "valve_state"])
    auto = auto.drop(columns=["s"])
    auto = auto[
        (auto["water_temp"].between(-40, 30)) &
        (auto["ambient_temp"].between(-40, 30))
    ]
    auto = auto[auto['date_time'] != '0/0/0 0:0:0']
    auto['date_time'] = pd.to_datetime(auto['date_time'], format='%y/%m/%d %H:%M:%S', errors='coerce')
    auto = auto.dropna(subset=['date_time'])
    auto.set_index("date_time", inplace=True)
    return auto

def anal_state(start_date, end_date, auto):
    auto = auto[(auto.index >= start_date) & (auto.index <= end_date)]
    drain_count = (auto["valve_state"] == "STUPA").sum()
    stupa_count = auto["valve_state"].count() - drain_count
    drain_p = drain_count / auto["valve_state"].count()
    stupa_p = 100 - drain_p
    return drain_count, stupa_count, drain_p, stupa_p

def plot_temp(start_date, end_date, auto, log):
    auto = auto[(auto.index >= start_date) & (auto.index <= end_date)]
    log = log[(log.index >= start_date) & (log.index <= end_date)]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(auto.index, auto['water_temp'], label='Water Temp', marker='.')
    ax.plot(log.index, log['ambient_temp'], label='Ambient Temp (Logger)', linestyle='solid')
    ax.plot(auto.index, auto['ambient_temp'], label='Ambient Temp', linestyle='--')
    ax.set_title("Automation: Water and Ambient Temperatures")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)

def plot_humidity(start_date, end_date, log):
    log = log[(log.index >= start_date) & (log.index <= end_date)]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(log.index, log['humidity'], label='Humidity')
    ax.set_title("Humidity")
    ax.set_ylabel("RH - %")
    st.pyplot(fig)

def plot_pressure(start_date, end_date, log):
    log = log[(log.index >= start_date) & (log.index <= end_date)]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(log.index, log['water_pressure'], label='Pressure')
    ax.set_title("Pressure")
    ax.set_ylabel("Pressure (bar)")
    st.pyplot(fig)

if logger_file and automation_file:
    logger = loggerproc(logger_file)
    automation = autoproc(automation_file)

    # Display summary stats
    st.header("Analytics")
    a, b, c, d = anal_state(start_date, end_date, automation)
    st.write(f"Drain Count: {a}")
    st.write(f"Stupa Count: {b}")
    st.write(f"Drain Percentage: {c:.2%}")
    st.write(f"Stupa Percentage: {d:.2%}")

    # Plot visualizations
    st.header("Visualizations")
    plot_temp(start_date, end_date, automation, logger)
    plot_humidity(start_date, end_date, logger)
    plot_pressure(start_date, end_date, logger)
else:
    st.warning("Please upload both Logger and Automation CSV files to proceed.")
