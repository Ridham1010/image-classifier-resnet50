import React, { useState } from 'react';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPrediction(null);
    setConfidence(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);
    setConfidence(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch('https://image-classifier-backend.onrender.com/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Something went wrong during prediction.');
      }

      const data = await response.json();
      setPrediction(data.prediction);
      setConfidence((data.confidence * 100).toFixed(2));

    } catch (err) {
      console.error("Upload error:", err);
      setError(err.message || "Failed to get prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`App ${darkMode ? 'dark-mode' : ''}`}>
      <header className="App-header d-flex justify-content-between align-items-center px-4">
        <h1>CIFAR-10 Image Classifier</h1>
        <div className="form-check form-switch text-white">
          <input
            className="form-check-input"
            type="checkbox"
            id="darkModeSwitch"
            checked={darkMode}
            onChange={() => setDarkMode(!darkMode)}
          />
          <label className="form-check-label" htmlFor="darkModeSwitch">
            {darkMode ? "Dark" : "Light"} Mode
          </label>
        </div>
      </header>

      <main>
        <div className="card">
          <h2>Upload an Image</h2>
          <input type="file" accept="image/*" onChange={handleFileChange} />
          {selectedFile && <p>Selected: {selectedFile.name}</p>}
          <button onClick={handleUpload} disabled={loading}>
            {loading ? 'Predicting...' : 'Predict Image'}
          </button>

          {error && <p className="error-message">Error: {error}</p>}

          {prediction && (
            <div className="prediction-result">
              <h3>Prediction: {prediction}</h3>
              <p>Confidence score: {confidence}%</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
