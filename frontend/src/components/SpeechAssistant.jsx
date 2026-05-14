import React, { useState, useEffect } from "react";

const BASE_URL = "http://localhost:8000";

function SpeechAssistant({ setResults, setSystemMessage }) {

  const [labels, setLabels] = useState([]);
  const [probabilities, setProbabilities] = useState(null);
  const [alternatives, setAlternatives] = useState([]);

  const [status, setStatus] = useState("idle");

  const [detectedState, setDetectedState] = useState(null);
  const [chooseStates, setChooseStates] = useState([]);

  const [readyForPreferences, setReadyForPreferences] = useState(false);
  const [regionForPreferences, setRegionForPreferences] = useState(null);


  useEffect(() => {

    const welcome =
      "Hello! I am AMUDHU. I can recommend Indian food using your voice.\n\nSpeech recognition currently works best for Tamil, Malayalam, Telugu and Kannada speakers.";

    setSystemMessage(welcome);
    speak(welcome);

  }, [setSystemMessage]);

  useEffect(() => {
      return () => {
          speechSynthesis.cancel(); // stop any ongoing speech
      };
  }, []);
    
  function resetAll() {
      setResults([]);
      setProbabilities(null);
      setAlternatives([]);
      setDetectedState(null);
      setChooseStates([]);
      setReadyForPreferences(false);
      setRegionForPreferences(null);
      setStatus("idle");
  }
    
  function speak(text) {

    speechSynthesis.cancel();

    const msg = new SpeechSynthesisUtterance(text);

    const voices = speechSynthesis.getVoices();

    const preferredVoice = voices.find(
      v => v.name === "Google UK English Female"
    );

    msg.voice = preferredVoice || voices[0];

    msg.rate = 0.95;
    msg.pitch = 1.05;

    speechSynthesis.speak(msg);
  }


  async function startAccentDetection() {

    resetAll();   
    setSystemMessage("");
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    setStatus("listening");
    
    const recorder = new MediaRecorder(stream);

    let chunks = [];

    recorder.ondataavailable = e => chunks.push(e.data);

    recorder.onstop = async () => {

      setStatus("processing");

      const blob = new Blob(chunks, { type: "audio/webm" });

      if (blob.size < 15000) {

        setStatus("idle");

        const message =
          "I could not hear any speech. Please try again and speak clearly.";

        setSystemMessage(message);
        speak(message);

        return;
      }

      const formData = new FormData();
      formData.append("audio", blob);

      try {

        const response = await fetch(`${BASE_URL}/recommend/speech`, {
          method: "POST",
          body: formData
        });

        const data = await response.json();

        setLabels(data.labels || []);
        setProbabilities(data.probabilities || null);
        setAlternatives(data.alternatives || []);

        if (data.stage === "low_confidence") {

          setStatus("idle");
          setSystemMessage(data.message);
          speak(data.message);
          return;
        }

        if (data.stage === "confirm_region") {

          const message = `I detected ${data.predicted_region}. Is that correct?`;

          setSystemMessage(message);
          speak(message);

          setDetectedState(data.predicted_region);
          setStatus("idle");
        }

        if (data.stage === "choose_state") {

          const message =
            "You sound like a Telugu speaker. Are you from Andhra Pradesh or Telangana?";

          setSystemMessage(message);
          speak(message);

          setChooseStates(data.options);
          setStatus("idle");
        }

      } catch (err) {

        console.error(err);
        setStatus("idle");

        const message = "Could not reach the speech service. Please try again.";

        setSystemMessage(message);
        speak(message);
      }

    };


    setSystemMessage("Listening will start in one second. Please prepare.");
    speak("Listening will start in one second.");

    setTimeout(() => {

      if (recorder.state === "inactive") {
        recorder.start();
      }

    }, 1000);

    setTimeout(() => {

      if (recorder.state === "recording") {
        recorder.stop();
      }

    }, 15000);
  }


  function handleAccentRejection() {

    setDetectedState(null);
    setChooseStates([]);

    const message =
      "Sorry about that. I might have misheard your accent. Please try recording again or use the text recommendation mode.";

    setSystemMessage(message);
    speak(message);
  }


  function confirmState(region) {

    setDetectedState(null);
    setChooseStates([]);

    setRegionForPreferences(region);

    const message = `
Please read the following sentence.

I would like spicy, tangy, sweet or savory.
Main course, snack, dessert or beverage.
I am vegetarian or meat based eater.

Example: I would like to eat spicy main course dish. I am vegetarian
`;

    setSystemMessage(message);
    speak(message);

    setReadyForPreferences(true);
  }


  function listenPreferences(region) {
    const SpeechRecognition = //speech to text converter- using default prebuilt text to speech converter for browser
      window.SpeechRecognition || window.webkitSpeechRecognition; 
    const recognition = new SpeechRecognition();
      
    recognition.onerror = function (event) {
        console.error("Speech error:", event.error);

        setStatus("idle");

        const message = "I couldn't understand that. Please try again.";
        setSystemMessage(message);
        speak(message);
    };

    recognition.onend = function () {
        setStatus(prev => (prev === "processing" ? prev : "idle"));
    };
    setReadyForPreferences(false);

    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    setStatus("listening");
    speechSynthesis.cancel();
    recognition.start();
    setTimeout(() => {
        if (recognition && recognition.state !== "inactive") {
            recognition.stop();
        }
    }, 10000);
      
    recognition.onresult = async function (event) {

      setStatus("processing");

      const transcript = event.results[0][0].transcript;

      const response = await fetch(`${BASE_URL}/recommend/speech_preferences`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          region: region,
          transcript: transcript
        })
      });

      const data = await response.json();

      setResults(data.results || []);

      if (data.system_message) {
          setSystemMessage(data.system_message);
          speak(data.system_message);
      } else {
          const message = "Here are some dishes you might enjoy.";
          setSystemMessage(message);
          speak(message);
      }

      setStatus("idle");
    };
  }


  return (

    <div className="voice-assistant">

      <p>🎤 Voice Assistant</p>

      <button
        className="mic-button"
        onClick={startAccentDetection}
        disabled={status === "listening"}
      >
        🎤 Start Accent Detection
      </button>


      {probabilities && labels.length > 0 && (

        <div className="accent-probabilities">

          <p><strong>Accent Confidence</strong></p>

          {labels.map((lang, i) => {

            const prob = probabilities[i] || 0;
            const percent = Math.round(prob * 100);

            return (

              <div key={lang} className="prob-row">

                <span className="prob-label">{lang}</span>

                <div className="prob-bar">
                  <div
                    className="prob-fill"
                    style={{ width: percent + "%" }}
                  ></div>
                </div>

                <span className="prob-value">{percent}%</span>

              </div>

            );
          })}

        </div>

      )}


      {alternatives.length > 0 && (

        <div className="accent-alternatives">

          <p><strong>Other possible accents:</strong></p>

          {alternatives.map((alt, i) => (
            <span key={i} className="alt-pill">{alt}</span>
          ))}

        </div>

      )}


      {status === "listening" && (
        <p className="listening-indicator">
          🎤 Listening... You have up to 15 seconds.
        </p>
      )}


      {status === "processing" && (
        <p className="thinking-indicator">🤖 Thinking...</p>
      )}


      {detectedState && (

        <div>

          <p>Confirm your state:</p>

          <button onClick={() => confirmState(detectedState)}>
            Yes
          </button>

          <button onClick={handleAccentRejection}>
            No
          </button>

        </div>

      )}


      {chooseStates.length > 0 && (

        <div>

          <p>Select your state:</p>

          {chooseStates.map((state) => (

            <button key={state} onClick={() => confirmState(state)}>
              {state}
            </button>

          ))}

        </div>

      )}


      {readyForPreferences && (

        <div style={{ marginTop: "10px" }}>

          <button onClick={() => listenPreferences(regionForPreferences)}>
            🎤 Speak Your Food Preferences
          </button>

        </div>

      )}

    </div>

  );
}

export default SpeechAssistant;