import json

from forgeai.tools.calculator import CalculatorTool
from forgeai.tools.csv_lookup import CSVLookupTool
from forgeai.tools.current_date import CurrentDateTool
from forgeai.tools.registry import ToolRegistry
from forgeai.tools.weather import WeatherTool


def test_calculator_supports_arithmetic():
    result = CalculatorTool().execute({"expression": "(12 * 8.5) + 40"})

    assert result.success
    assert result.output == "142.0"


def test_calculator_rejects_non_arithmetic_expressions():
    result = CalculatorTool().execute(
        {"expression": "__import__('os').system('echo unsafe')"}
    )

    assert not result.success
    assert result.error


def test_csv_lookup_only_uses_allowed_filename(tmp_path):
    allowed_file = tmp_path / "catalogue.csv"
    allowed_file.write_text(
        "part_number,description\nABC-123,Bearing\nXYZ-900,Seal\n",
        encoding="utf-8",
    )
    tool = CSVLookupTool()

    found = tool.execute(
        {
            "file": "catalogue.csv",
            "column": "part_number",
            "value": "ABC-123",
        },
        {"catalogue.csv": str(allowed_file)},
    )
    denied = tool.execute(
        {
            "file": str(allowed_file),
            "column": "part_number",
            "value": "ABC-123",
        },
        {"catalogue.csv": str(allowed_file)},
    )

    assert found.success
    assert json.loads(found.output) == [
        {"part_number": "ABC-123", "description": "Bearing"}
    ]
    assert not denied.success


def test_current_date_is_iso_formatted():
    result = CurrentDateTool().execute({})

    assert result.success
    assert len(result.output) == 10
    assert result.output[4] == "-"
    assert result.output[7] == "-"


def test_registry_requires_tool_to_be_selected_and_registered():
    registry = ToolRegistry()

    denied = registry.execute_tool(
        "weather",
        {"city": "London"},
        allowed_tools=["calculator"],
    )
    unknown = registry.execute_tool(
        "not_registered",
        {},
        allowed_tools=["not_registered"],
    )

    assert not denied.success
    assert not unknown.success


def test_weather_uses_fixed_api_endpoints_with_mocked_http(monkeypatch):
    seen_urls = []

    class FakeResponse:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self.data

    class FakeClient:
        def __init__(self, timeout):
            assert timeout == 10.0
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, params):
            seen_urls.append(url)
            if url == WeatherTool.GEOCODING_URL:
                return FakeResponse(
                    {"results": [{"name": "London", "latitude": 51.5, "longitude": -0.1}]}
                )
            return FakeResponse({"current": {"temperature_2m": 14}})

    monkeypatch.setattr("forgeai.tools.weather.httpx.Client", FakeClient)
    result = WeatherTool().execute({"city": "London"})

    assert result.success
    assert "temperature_2m" in result.output
    assert seen_urls == [WeatherTool.GEOCODING_URL, WeatherTool.FORECAST_URL]
