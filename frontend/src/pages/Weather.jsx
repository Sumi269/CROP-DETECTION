import { useState } from "react";
import WeatherCard from "../components/WeatherCard";

export default function Weather() {
  const [city, setCity] = useState("");
  const [data, setData] = useState(null);

  const API_KEY = "YOUR_API_KEY"; // put your API key

  const fetchWeather = async () => {
    try {
      const res = await fetch(
        `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${API_KEY}&units=metric`
      );

      const result = await res.json();

      // convert into your WeatherCard format
      const formatted = {
        temp: result.main.temp,
        humidity: result.main.humidity,
        rain: result.weather[0].description,
        alert:
          result.main.temp > 35
            ? "⚠ High temperature - risk for crops"
            : result.weather[0].main === "Rain"
            ? "🌧 Rain expected - reduce irrigation"
            : "✅ Weather is good for crops",
      };

      setData(formatted);
    } catch (err) {
      alert("Error fetching weather");
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-xl font-bold mb-4">🌦 Weather for Crops</h1>

      <input
        type="text"
        placeholder="Enter city"
        className="border p-2 mr-2"
        value={city}
        onChange={(e) => setCity(e.target.value)}
      />

      <button
        onClick={fetchWeather}
        className="bg-green-500 text-white px-4 py-2"
      >
        Get Weather
      </button>

      <div className="mt-5">
        <WeatherCard data={data} />
      </div>
    </div>
  );
}
