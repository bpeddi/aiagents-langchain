import requests
from typing import Dict
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

@tool
def get_weather(location: str) -> str:
    """
    Get current weather conditions for a location.
    
    Args:
        location: City name or coordinates (e.g., "New York", "Tokyo", "52.52,13.41")
    
    example : 
                print("\n" + "="*60)
                print("🚀 Get the weather")
                print("="*60)
                locations = [
                    "New York",
                    "London",
                    "Tokyo",
                    "Paris",
                    "Sydney"
                    "hyderabad",
                    "Dallas"
                ]
                for location in locations:
                    result = get_weather(location)
                    print(result)
                    print("="*60)

    Returns:
        Formatted weather report or error message
    """
    # Open-Meteo weather code mapping (WMO codes)
    WEATHER_CODES = {
        0: "Clear sky",
        1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        56: "Light freezing drizzle", 57: "Dense freezing drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        66: "Light freezing rain", 67: "Heavy freezing rain",
        71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
    }
    
    def geocode_location(location: str) -> Dict:
        """Convert location name to coordinates using Open-Meteo's geocoding API."""
        try:
            geo_url = "https://geocoding-api.open-meteo.com/v1/search"
            geo_params = {
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json"
            }
            geo_response = requests.get(geo_url, params=geo_params, timeout=10)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            
            if "results" not in geo_data or not geo_data["results"]:
                raise ValueError(f"Location '{location}' not found")
            
            result = geo_data["results"][0]
            return {
                "latitude": result["latitude"],
                "longitude": result["longitude"],
                "name": result.get("name", location),
                "country": result.get("country", "Unknown"),
                "admin1": result.get("admin1", "")  # State/region
            }
        except Exception as e:
            raise ValueError(f"Geocoding failed: {str(e)}")
    
    def get_weather_by_coords(lat: float, lon: float) -> Dict:
        """Get weather data for coordinates."""
        try:
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                          "weather_code,wind_speed_10m,wind_direction_10m,"
                          "is_day,precipitation",
                "temperature_unit": "celsius",  # or "fahrenheit"
                "wind_speed_unit": "ms",  # or "kmh", "mph", "kn"
                "timezone": "auto"
            }
            
            response = requests.get(weather_url, params=weather_params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ValueError(f"Weather API failed: {str(e)}")
    
    def format_weather_report(location_info: Dict, weather_data: Dict) -> str:
        """Format the weather data into a readable report."""
        current = weather_data.get("current", {})
        weather_code = current.get("weather_code", 0)
        
        # Get weather description
        weather_desc = WEATHER_CODES.get(weather_code, "Unknown conditions")
        
        # Format location name
        location_name = location_info["name"]
        if location_info.get("admin1"):
            location_name += f", {location_info['admin1']}"
        if location_info.get("country") and location_info["country"] != "Unknown":
            location_name += f", {location_info['country']}"
        
        # Build report
        report = [
            f"🌤️ Weather in {location_name}:",
            f"• Condition: {weather_desc}",
            f"• Temperature: {current.get('temperature_2m', 'N/A')}°C",
            f"• Feels like: {current.get('apparent_temperature', 'N/A')}°C",
            f"• Humidity: {current.get('relative_humidity_2m', 'N/A')}%",
            f"• Wind: {current.get('wind_speed_10m', 'N/A')} m/s",
            f"• Precipitation: {current.get('precipitation', 'N/A')} mm"
        ]
        
        # Add day/night indicator if available
        if "is_day" in current:
            report.append(f"• Time: {'Day' if current['is_day'] else 'Night'}")
        
        return "\n".join(report)
    
    try:
        # Step 1: Check if input is coordinates
        if "," in location and " " not in location:
            # Might be coordinates "lat,lon"
            try:
                lat, lon = map(float, location.split(","))
                location_info = {
                    "name": f"{lat:.2f}°N, {lon:.2f}°E",
                    "country": "Coordinates",
                    "latitude": lat,
                    "longitude": lon
                }
            except ValueError:
                # Not coordinates, treat as location name
                location_info = geocode_location(location)
        else:
            # Treat as location name
            location_info = geocode_location(location)
        
        # Step 2: Get weather data
        weather_data = get_weather_by_coords(
            location_info["latitude"], 
            location_info["longitude"]
        )
        
        # Step 3: Format and return report
        return format_weather_report(location_info, weather_data)
    
    except ValueError as e:
        return f"❌ {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"🌐 Network error: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"📊 Data parsing error: {str(e)}"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"

def search_tool():
    """
    Returns a Tavily web search tool for your LangGraph agent.
    
    HOW TO USE IN YOUR AGENT:
    ==========================
    1. Import this function in your agent code:
        from tools import search_tool
    
    2. Initialize the tool:
        search_tool = search_tool()
    
    3. Use it in your search_node function:
        results = search_tool.invoke({"query": query})
    
    Returns:
        TavilySearchResults tool object configured with:
        - max_results: 3 (returns top 3 search results)
        - description: Explains when to use the tool
    
    Used for questions about:
        - News and current events
        - Stock prices and market data
        - Weather reports
        - Sports scores
        - Product releases
        - Company announcements
        - Any information that changes frequently
    
    Example in agent:
        # In your search_node function
        from tools import search_tool
        search_tool_instance = search_tool()
        results = search_tool_instance.invoke({"query": query})
        # Results are then formatted and added to conversation context
    """
    return TavilySearchResults(
        max_results=3,
        description="Use this tool to search for current events, recent developments, real-time information, or facts that might change over time."
    )


# ===========================
# TOOL REGISTRY
# ===========================

def get_all_my_tools():
    """Returns a list of all available insurance tools for the agent."""
    return [
        search_tool,
        get_weather,
    ]


# Export TOOLS for compatibility
TOOLS = get_all_my_tools()