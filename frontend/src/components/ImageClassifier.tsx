import React, { useState, useRef } from "react";
import "./ImageClassifier.css";

interface Prediction {
  class_name: string;
  class_id: number;
  confidence: number;
}

const ImageClassifier: React.FC = () => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [predictions, setPredictions] = useState<Prediction[][]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  // Backend API URL - adjust this to match your FastAPI server
  const API_BASE_URL = "http://localhost:8000/api/v1";

  const handleFileSelect = (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    const validFiles: File[] = [];
    const newPreviews: string[] = [];
    let loadedCount = 0;

    fileArray.forEach((file, index) => {
      // Validate file type
      if (!file.type.startsWith("image/")) {
        setError(`${file.name} is not a valid image file`);
        return;
      }

      // Validate file size (10MB limit)
      if (file.size > 10 * 1024 * 1024) {
        setError(`${file.name} is too large (max 10MB)`);
        return;
      }

      validFiles.push(file);

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        newPreviews[index] = e.target?.result as string;
        loadedCount++;

        // Update previews when all images are loaded
        if (loadedCount === validFiles.length) {
          setPreviews(newPreviews.filter(Boolean)); // Remove empty slots
        }
      };
      reader.readAsDataURL(file);
    });

    if (validFiles.length > 0) {
      setSelectedFiles(validFiles);
      setError(null);
      setPredictions([]);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files);
    }
  };

  const handleFolderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files);
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

    const files = event.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelect(files);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFolderClick = () => {
    folderInputRef.current?.click();
  };

  const handleClassify = async () => {
    if (selectedFiles.length === 0) return;

    setLoading(true);
    setError(null);

    try {
      // Create FormData with all files for batch processing
      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append("files", file);
      });

      const response = await fetch(`${API_BASE_URL}/predict-batch?top_k=5`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.status === "success" && data.results) {
        // Extract predictions in the same order as uploaded files
        const allPredictions: Prediction[][] = data.results.map(
          (result: any) => result.predictions
        );
        setPredictions(allPredictions);
      } else {
        throw new Error("Batch classification failed");
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
    setSelectedFiles([]);
    setPreviews([]);
    setPredictions([]);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    if (folderInputRef.current) {
      folderInputRef.current.value = "";
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
            selectedFiles.length > 0 ? "has-file" : ""
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="drop-zone-content">
            <div className="upload-icon">📤</div>
            <h3>Drop images here or click to upload</h3>
            <p className="drop-text">
              Support for multiple images at once. JPG, PNG, WEBP accepted.
            </p>
            <div className="upload-buttons">
              <button className="choose-files-btn" onClick={handleUploadClick}>
                📁 Choose Files
              </button>
              <button className="upload-folder-btn" onClick={handleFolderClick}>
                📂 Upload Folder
              </button>
            </div>
          </div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileChange}
          className="file-input"
        />

        <input
          ref={folderInputRef}
          type="file"
          accept="image/*"
          multiple
          {...({ webkitdirectory: "" } as any)}
          onChange={handleFolderChange}
          className="file-input"
        />
      </div>

      {/* Images Display Section */}
      {selectedFiles.length > 0 && (
        <div className="images-display-section">
          <div className="section-header">
            <div className="upload-icon">�</div>
            <div>
              <h3>
                {selectedFiles.length} image
                {selectedFiles.length > 1 ? "s" : ""} uploaded
              </h3>
              <p>Ready to classify your images</p>
            </div>
          </div>
          <div className="uploaded-images-grid">
            {previews.map((preview, index) => (
              <div key={index} className="uploaded-image-item">
                <img
                  src={preview}
                  alt={`Upload ${index + 1}`}
                  className="uploaded-image-preview"
                />
                <p className="uploaded-image-name">
                  {selectedFiles[index]?.name}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Classify Button */}
      {selectedFiles.length > 0 && (
        <div className="classify-section">
          <button
            onClick={handleClassify}
            disabled={selectedFiles.length === 0 || loading}
            className="classify-button"
          >
            {loading
              ? "Classifying..."
              : `Classify ${selectedFiles.length} Image${
                  selectedFiles.length > 1 ? "s" : ""
                }`}
          </button>

          <button onClick={handleReset} className="reset-button">
            Reset
          </button>
        </div>
      )}

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {loading && (
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Analyzing your images...</p>
        </div>
      )}

      {predictions.length > 0 && (
        <div className="results-section">
          <h2>Classification Results</h2>
          {predictions.map((predictionSet, imageIndex) => (
            <div key={imageIndex} className="image-results">
              <div className="image-info">
                <img
                  src={previews[imageIndex]}
                  alt={`Preview ${imageIndex + 1}`}
                  className="result-preview"
                />
                <h3>{selectedFiles[imageIndex]?.name}</h3>
              </div>
              <div className="predictions-list">
                {predictionSet.map((prediction, index) => (
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
          ))}
        </div>
      )}
    </div>
  );
};

export default ImageClassifier;
