import SpeechAssistant from "./components/SpeechAssistant";
import bgImage from "./bg.jpg";
import React, { useState } from "react";
import "./styles.css";

const BASE_URL = "http://localhost:8000";

function App() {

  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const [mode, setMode] = useState(null);

  const [locationType, setLocationType] = useState("");
  const [locationValue, setLocationValue] = useState("");

  const [taste, setTaste] = useState("");
  const [diet, setDiet] = useState("");
  const [course, setCourse] = useState("");

  const [results, setResults] = useState([]);
  const [systemMessage, setSystemMessage] = useState("");

  const regions = ["South India","North India","East India","West India","Central India","North-East India"];

  const states = [
    "Tamil Nadu","Kerala","Karnataka","Telangana","Andhra Pradesh","Maharashtra","Gujarat",
    "Punjab","Rajasthan","Manipur","Himachal Pradesh","Meghalaya","Haryana","Uttarakhand",
    "West Bengal","Odisha","Sikkim","Bihar","Assam","Mizoram","Nagaland","Tripura",
    "Arunachal Pradesh","Madhya Pradesh","Chhattisgarh","Goa"
  ];

  const unionTerritories = [
    "Delhi","Puducherry","Chandigarh","Lakshadweep","Andaman & Nicobar",
    "Jammu & Kashmir","Daman & Diu","Ladakh"
  ];

  // 🔊 Voice feedback
  function speak(text) {
    const msg = new SpeechSynthesisUtterance(text);
    speechSynthesis.cancel();
    speechSynthesis.speak(msg);
  }

  // ================= TEXT =================
  const handleTextSubmit = async () => {

    if (isLoading) return;

    setIsLoading(true);
    setResults([]);
    setSystemMessage("🤖 Thinking...");
    setLoadingStep("Analyzing your preferences...");
    speak("Let me think");

    try {

      setLoadingStep("Finding best dishes...");
      const response = await fetch(`${BASE_URL}/recommend/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          location: locationValue,
          course,
          taste,
          diet
        })
      });

      const data = await response.json();

      setLoadingStep("Generating explanation...");
      setResults(data.results || []);
      setSystemMessage(data.system_message || "");

    } catch (err) {
      console.error(err);
      setSystemMessage("Something went wrong.");
    }

    setIsLoading(false);
    setLoadingStep("");
  };

  // ================= IMAGE =================
  const handleImageUpload = async () => {

    if (!selectedImage || isLoading) return;

    setIsLoading(true);
    setResults([]);
    setSystemMessage("🤖 Thinking...");
    setLoadingStep("Analyzing your image...");
    speak("Analyzing your dish");

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {

      setLoadingStep("Matching similar dishes...");
      const response = await fetch(`${BASE_URL}/recommend/image`, {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      setLoadingStep("Preparing recommendations...");
      setResults(data.results || []);
      setSystemMessage(data.system_message || "Here are some dishes");

    } catch (err) {
      console.error(err);
      setSystemMessage("Image processing failed.");
    }

    setIsLoading(false);
    setLoadingStep("");
  };

  return (
    <div
      className="container"
      style={{
        backgroundImage: `url(${bgImage})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundAttachment: "fixed"
      }}
    >

      <p className="subtitle">Discover food through voice, text, or image</p>

      {/* MODE BUTTONS */}
      <div className="mode-buttons">
        <button onClick={() => setMode("text")}>📝 Type</button>
        <button onClick={() => setMode("speech")}>🎤 Speak</button>
        <button onClick={() => setMode("image")}>📸 Upload</button>
      </div>

      <div className="input-panel">

        {/* TEXT MODE */}
        {mode === "text" && (
          <div className="form">

            <select
              value={locationType}
              onChange={(e) => {
                setLocationType(e.target.value);
                setLocationValue("");
              }}
            >
              <option value="">Select Location Type</option>
              <option value="region">Region</option>
              <option value="state">State</option>
              <option value="ut">Union Territory</option>
            </select>

            {locationType === "region" && (
              <select value={locationValue} onChange={(e) => setLocationValue(e.target.value)}>
                <option value="">Select Region</option>
                {regions.map((r) => <option key={r}>{r}</option>)}
              </select>
            )}

            {locationType === "state" && (
              <select value={locationValue} onChange={(e) => setLocationValue(e.target.value)}>
                <option value="">Select State</option>
                {states.map((s) => <option key={s}>{s}</option>)}
              </select>
            )}

            {locationType === "ut" && (
              <select value={locationValue} onChange={(e) => setLocationValue(e.target.value)}>
                <option value="">Select UT</option>
                {unionTerritories.map((u) => <option key={u}>{u}</option>)}
              </select>
            )}

            <select value={course} onChange={(e) => setCourse(e.target.value)}>
              <option value="">Select Course</option>
              <option value="main course">Main Course</option>
              <option value="snack">Snack</option>
              <option value="dessert">Dessert</option>
              <option value="beverage">Beverage</option>
            </select>

            {course === "main course" && (
              <>
                <select value={taste} onChange={(e) => setTaste(e.target.value)}>
                  <option value="">Select Taste</option>
                  <option value="spicy">Spicy</option>
                  <option value="sweet">Sweet</option>
                  <option value="savoury">Savoury</option>
                  <option value="tangy">Tangy</option>
                </select>

                <select value={diet} onChange={(e) => setDiet(e.target.value)}>
                  <option value="">Select Diet</option>
                  <option value="vegetarian">Vegetarian</option>
                  <option value="meat-based">Meat-based</option>
                </select>
              </>
            )}

            <button onClick={handleTextSubmit} disabled={isLoading}>
              {isLoading ? "Thinking..." : "Recommend"}
            </button>

          </div>
        )}

        {/* SPEECH */}
        {mode === "speech" && (
          <SpeechAssistant setResults={setResults} setSystemMessage={setSystemMessage} />
        )}

        {/* IMAGE */}
        {mode === "image" && (
          <div className="form">

            <input type="file" accept="image/*"
              onChange={(e) => setSelectedImage(e.target.files[0])}
            />

            {selectedImage && (
              <div className="preview-container">
                <p>Uploaded Image:</p>
                <img src={URL.createObjectURL(selectedImage)} alt="preview" className="preview-image" />
              </div>
            )}

            <button onClick={handleImageUpload} disabled={isLoading}>
              {isLoading ? "Processing..." : "Upload & Recommend"}
            </button>

          </div>
        )}

      </div>

      {/* SYSTEM MESSAGE */}
      {systemMessage && (
        <div className="system-message">
          <p>{systemMessage}</p>
        </div>
      )}

      {/* 🧠 STEP-BY-STEP + LOADER */}
      {isLoading && (
        <div className="thinking-indicator">
          🤖 {loadingStep}
          <span className="dots"></span>
        </div>
      )}

      {/* RESULTS */}
      <div className="results">
        {results.map((item, index) => (
          <div key={index} className="food-card">

            <img src={`http://localhost:8000${item.image}`} alt={item.name} />

            <div>
              <h3>{item.name}</h3>
              <p><strong>Taste:</strong> {item.taste}</p>
              <p><strong>Course:</strong> {item.course_type}</p>
              <p><strong>Diet:</strong> {item.diet}</p>
              <p><strong>Origin:</strong> {item.state}</p>

              <p>{item.about}</p>
            </div>

          </div>
        ))}
      </div>

    </div>
  );
}

export default App;