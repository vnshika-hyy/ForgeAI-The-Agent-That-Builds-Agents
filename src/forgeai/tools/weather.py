import json

import httpx

from .registry import ToolResult


class WeatherTool:
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        city = arguments.get("city")
        if not isinstance(city, str) or not city.strip():
            return ToolResult(
                success=False,
                output="",
                error="Provide a city name for the weather lookup.",
            )
        try:
            with httpx.Client(timeout=10.0) as client:
                geocode = client.get(
                    self.GEOCODING_URL,
                    params={"name": city.strip(), "count": 1, "language": "en"},
                )
                geocode.raise_for_status()
                locations = geocode.json().get("results") or []
                if not locations:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"No location found for '{city}'.",
                    )
                location = locations[0]
                forecast = client.get(
                    self.FORECAST_URL,
                    params={
                        "latitude": location["latitude"],
                        "longitude": location["longitude"],
                        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                    },
                )
                forecast.raise_for_status()
                current = forecast.json().get("current")
                if not isinstance(current, dict):
                    return ToolResult(
                        success=False,
                        output="",
                        error="Weather provider returned no current conditions.",
                    )
                return ToolResult(
                    success=True,
                    output=json.dumps(
                        {
                            "location": location.get("name", city),
                            "country": location.get("country"),
                            "current": current,
                        },
                        ensure_ascii=False,
                    ),
                )
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Weather lookup failed: {exc}",
            )
