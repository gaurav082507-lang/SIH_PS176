import openmeteo_requests
import pandas as pd
import requests_cache

from retry_requests import retry
from langchain_core.tools import tool


# ============================================================
# OPEN-METEO CLIENT
# ============================================================

cache_session = requests_cache.CachedSession(
    ".cache",
    expire_after=3600
)

retry_session = retry(
    cache_session,
    retries=5,
    backoff_factor=0.2
)

openmeteo = openmeteo_requests.Client(
    session=retry_session
)


# ============================================================
# WEATHER TOOL
# ============================================================

@tool
def weather_agent(latitude: float, longitude: float) -> dict:
    """
    Get weather information for a given latitude and longitude.

    Returns:
    - Temperature
    - Rainfall
    - Precipitation probability
    - Wind speed
    - Wind direction
    - Wind gusts
    - Cloud cover
    - Visibility
    - Surface pressure
    - Daily weather forecast

    Use this tool when the user asks about weather,
    wind, rainfall, or atmospheric conditions at a location.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "hourly": [
            "temperature_2m",
            "precipitation",
            "precipitation_probability",
            "rain",
            "weather_code",
            "cloud_cover",
            "visibility",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "surface_pressure"
        ],

        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
            "weather_code"
        ],

        "forecast_days": 7,
        "timezone": "auto"
    }

    try:

        responses = openmeteo.weather_api(
            url,
            params=params
        )

        response = responses[0]

        # ----------------------------------------------------
        # HOURLY DATA
        # ----------------------------------------------------

        hourly = response.Hourly()

        hourly_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(
                    hourly.Time(),
                    unit="s",
                    utc=True
                ),
                end=pd.to_datetime(
                    hourly.TimeEnd(),
                    unit="s",
                    utc=True
                ),
                freq=pd.Timedelta(
                    seconds=hourly.Interval()
                ),
                inclusive="left"
            ),

            "temperature":
                hourly.Variables(0).ValuesAsNumpy(),

            "precipitation":
                hourly.Variables(1).ValuesAsNumpy(),

            "precipitation_probability":
                hourly.Variables(2).ValuesAsNumpy(),

            "rain":
                hourly.Variables(3).ValuesAsNumpy(),

            "weather_code":
                hourly.Variables(4).ValuesAsNumpy(),

            "cloud_cover":
                hourly.Variables(5).ValuesAsNumpy(),

            "visibility":
                hourly.Variables(6).ValuesAsNumpy(),

            "wind_speed":
                hourly.Variables(7).ValuesAsNumpy(),

            "wind_direction":
                hourly.Variables(8).ValuesAsNumpy(),

            "wind_gusts":
                hourly.Variables(9).ValuesAsNumpy(),

            "surface_pressure":
                hourly.Variables(10).ValuesAsNumpy()
        }

        hourly_df = pd.DataFrame(hourly_data)


        # ----------------------------------------------------
        # DAILY DATA
        # ----------------------------------------------------

        daily = response.Daily()

        daily_data = {
            "date": pd.date_range(
                start=pd.to_datetime(
                    daily.Time(),
                    unit="s",
                    utc=True
                ),
                end=pd.to_datetime(
                    daily.TimeEnd(),
                    unit="s",
                    utc=True
                ),
                freq=pd.Timedelta(
                    seconds=daily.Interval()
                ),
                inclusive="left"
            ),

            "temperature_max":
                daily.Variables(0).ValuesAsNumpy(),

            "temperature_min":
                daily.Variables(1).ValuesAsNumpy(),

            "precipitation_sum":
                daily.Variables(2).ValuesAsNumpy(),

            "precipitation_probability_max":
                daily.Variables(3).ValuesAsNumpy(),

            "wind_speed_max":
                daily.Variables(4).ValuesAsNumpy(),

            "wind_gusts_max":
                daily.Variables(5).ValuesAsNumpy(),

            "wind_direction":
                daily.Variables(6).ValuesAsNumpy(),

            "weather_code":
                daily.Variables(7).ValuesAsNumpy()
        }

        daily_df = pd.DataFrame(daily_data)


        # ----------------------------------------------------
        # RETURN STRUCTURED DATA
        # ----------------------------------------------------

        return {
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },

            "timezone": response.Timezone(),

            "hourly": hourly_df.to_dict(
                orient="records"
            ),

            "daily": daily_df.to_dict(
                orient="records"
            )
        }

    except Exception as e:

        return {
            "error": f"Unable to fetch weather data: {str(e)}"
        }


# ============================================================
# OCEAN DATA TOOL
# ============================================================

@tool
def get_ocean_data(latitude: float, longitude: float) -> dict:
    """
    Get marine and oceanographic conditions for a given
    latitude and longitude.

    Returns information about:
    - Wave height
    - Wave direction
    - Wave period
    - Swell wave height
    - Swell wave direction
    - Swell wave period
    - Sea surface temperature
    - Ocean current velocity
    - Ocean current direction

    Use this tool when the user asks about sea conditions,
    waves, ocean conditions, SST, currents, or marine
    conditions relevant to fishing and navigation.
    """

    url = "https://marine-api.open-meteo.com/v1/marine"

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "hourly": [
            "wave_height",
            "wave_direction",
            "wave_period",
            "swell_wave_height",
            "swell_wave_direction",
            "swell_wave_period",
            "sea_surface_temperature",
            "ocean_current_velocity",
            "ocean_current_direction"
        ],

        "forecast_days": 7,
        "timezone": "auto"
    }

    try:

        # ------------------------------------------------
        # API REQUEST
        # ------------------------------------------------

        responses = openmeteo.weather_api(
            url,
            params=params
        )

        response = responses[0]

        # ------------------------------------------------
        # HOURLY DATA
        # ------------------------------------------------

        hourly = response.Hourly()

        # Number of timestamps
        times = pd.date_range(
            start=pd.to_datetime(
                hourly.Time(),
                unit="s",
                utc=True
            ),
            end=pd.to_datetime(
                hourly.TimeEnd(),
                unit="s",
                utc=True
            ),
            freq=pd.Timedelta(
                seconds=hourly.Interval()
            ),
            inclusive="left"
        )

        # ------------------------------------------------
        # SAFELY EXTRACT VARIABLES
        # ------------------------------------------------

        variables = {}

        variable_names = [
            "wave_height",
            "wave_direction",
            "wave_period",
            "swell_wave_height",
            "swell_wave_direction",
            "swell_wave_period",
            "sea_surface_temperature",
            "ocean_current_velocity",
            "ocean_current_direction"
        ]

        for i, name in enumerate(variable_names):

            try:

                values = hourly.Variables(i).ValuesAsNumpy()

                variables[name] = values

            except Exception:

                # If a variable is unavailable
                variables[name] = [None] * len(times)

        # ------------------------------------------------
        # CREATE DATAFRAME
        # ------------------------------------------------

        hourly_dataframe = pd.DataFrame({
            "datetime": times
        })

        for name in variable_names:

            hourly_dataframe[name] = variables[name]

        # ------------------------------------------------
        # ROUND NUMERICAL VALUES
        # ------------------------------------------------

        numeric_columns = [
            "wave_height",
            "wave_direction",
            "wave_period",
            "swell_wave_height",
            "swell_wave_direction",
            "swell_wave_period",
            "sea_surface_temperature",
            "ocean_current_velocity",
            "ocean_current_direction"
        ]

        for column in numeric_columns:

            hourly_dataframe[column] = (
                hourly_dataframe[column]
                .round(2)
            )

        # ------------------------------------------------
        # RETURN STRUCTURED DATA
        # ------------------------------------------------

        return {
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },

            "timezone": response.Timezone(),

            "hourly": hourly_dataframe.to_dict(
                orient="records"
            )
        }

    except Exception as e:

        return {
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },

            "error": f"Unable to fetch ocean data: {str(e)}"
        }