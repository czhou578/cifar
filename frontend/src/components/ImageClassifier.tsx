import React, { useState, useRef } from "react";
import "./ImageClassifier.css";

interface Prediction {
  class_name: string;
  class_id: number;
  confidence: number;
}

interface PredictionResponse {
  status: string;
  filename: string;
  predictions: Prediction[];
  top_prediction: Prediction;
}

const ImageClassifier: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Backend API URL - adjust this to match your FastAPI server
  const API_BASE_URL = "http://localhost:8000/api/v1";

  const handleFileSelect = (file: File) => {
    // Validate file type
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file");
      return;
    }

    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      setError("File size must be less than 10MB");
      return;
    }

    setSelectedFile(file);
    setError(null);
    setPredictions([]);

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(false);

    const file = event.dataTransfer.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleClassify = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE_URL}/predict?top_k=5`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: PredictionResponse = await response.json();

      if (data.status === "success") {
        setPredictions(data.predictions);
      } else {
        throw new Error("Classification failed");
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An error occurred during classification"
      );
      setPredictions([]);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setPredictions([]);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const formatConfidence = (confidence: number) => {
    return (confidence * 100).toFixed(1);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.7) return "#4CAF50";
    if (confidence > 0.4) return "#FF9800";
    return "#F44336";
  };

  return (
    <div className="image-classifier">
      <div className="upload-section">
        <div
          className={`drop-zone ${dragOver ? "drag-over" : ""} ${
            selectedFile ? "has-file" : ""
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleUploadClick}
        >
          {preview ? (
            <div className="preview-container">
              <img src={preview} alt="Preview" className="preview-image" />
              <div className="file-info">
                <p className="file-name">{selectedFile?.name}</p>
                <p className="file-size">
                  {selectedFile ? (selectedFile.size / 1024).toFixed(1) : 0} KB
                </p>
              </div>
            </div>
          ) : (
            <div className="drop-zone-content">
              <div className="upload-icon">📷</div>
              <p className="drop-text">
                Drop an image here or{" "}
                <span className="click-text">click to upload</span>
              </p>
              <p className="file-types">
                Supports: JPG, PNG, BMP, TIFF (max 10MB)
              </p>
            </div>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="file-input"
        />

        <div className="button-group">
          <button
            onClick={handleClassify}
            disabled={!selectedFile || loading}
            className="classify-button"
          >
            {loading ? "Classifying..." : "Classify Image"}
          </button>

          {selectedFile && (
            <button onClick={handleReset} className="reset-button">
              Reset
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {loading && (
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Analyzing your image...</p>
        </div>
      )}

      {predictions.length > 0 && (
        <div className="results-section">
          <h2>Classification Results</h2>
          <div className="predictions-list">
            {predictions.map((prediction, index) => (
              <div
                key={index}
                className={`prediction-item ${
                  index === 0 ? "top-prediction" : ""
                }`}
              >
                <div className="prediction-rank">
                  {index === 0 ? "🏆" : `#${index + 1}`}
                </div>
                <div className="prediction-details">
                  <div className="class-name">
                    {prediction.class_name.replace("_", " ")}
                  </div>
                  <div className="class-id">
                    Class ID: {prediction.class_id}
                  </div>
                </div>
                <div className="confidence-section">
                  <div
                    className="confidence-bar"
                    style={{
                      width: `${prediction.confidence * 100}%`,
                      backgroundColor: getConfidenceColor(
                        prediction.confidence
                      ),
                    }}
                  ></div>
                  <div className="confidence-text">
                    {formatConfidence(prediction.confidence)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageClassifier;
