import React, { useState, useRef, useEffect } from "react";
import "./ImageClassifier.css";
import { useWebSocket } from "../hooks/useWebSocket";

interface Prediction {
  class_name: string;
  class_id: number;
  confidence: number;
}

interface PredictionResult {
  predictions: Prediction[];
  caption?: string;
  streamingCaption?: string;
  captionComplete?: boolean;
}

const ImageClassifier: React.FC = () => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [generateCaptions, setGenerateCaptions] = useState(false);
  const [useStreaming, setUseStreaming] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const completedCaptionsRef = useRef<number>(0); // Track completed captions

  // Backend API URL - adjust this to match your FastAPI server
  const API_BASE_URL = "http://localhost:8000/api/v1";
  const WS_URL = "ws://localhost:8000/api/v1/ws/caption";

  // WebSocket connection
  const { sendMessage, lastMessage, isConnected, connect, disconnect } =
    useWebSocket(WS_URL, false);

  // Handle WebSocket messages
  useEffect(() => {
    console.log("Last WebSocket message:", lastMessage);

    if (!lastMessage) return;

    const fileIndex = selectedFiles.findIndex(
      (f) => f.name === lastMessage.filename
    );

    if (fileIndex === -1) {
      console.log("File not found:", lastMessage.filename);
      // Don't return early for caption_complete - we still need to count it
      if (lastMessage.type === "caption_complete") {
        completedCaptionsRef.current += 1;
        console.log(
          `Completed ${completedCaptionsRef.current} of ${selectedFiles.length} captions`
        );

        if (completedCaptionsRef.current >= selectedFiles.length) {
          console.log("All captions complete, setting loading to false");
          setLoading(false);
          completedCaptionsRef.current = 0; // Reset for next batch
        }
      }
      return;
    }

    console.log("Processing message for file index:", fileIndex);

    switch (lastMessage.type) {
      case "caption_start":
        setPredictions((prev) => {
          const newPredictions = [...prev];
          if (newPredictions[fileIndex]) {
            newPredictions[fileIndex] = {
              ...newPredictions[fileIndex],
              streamingCaption: "",
              captionComplete: false,
            };
            console.log("caption_start - Updated predictions:", newPredictions);
          } else {
            console.log("caption_start - No prediction at index:", fileIndex);
          }
          return newPredictions;
        });
        break;

      case "caption_token":
        setPredictions((prev) => {
          const newPredictions = [...prev];
          if (newPredictions[fileIndex]) {
            newPredictions[fileIndex] = {
              ...newPredictions[fileIndex],
              streamingCaption: lastMessage.partial,
            };
            console.log(
              "caption_token - Updated predictions:",
              newPredictions[fileIndex]
            );
          } else {
            console.log("caption_token - No prediction at index:", fileIndex);
          }
          return newPredictions;
        });
        break;

      case "caption_complete":
        setPredictions((prev) => {
          const newPredictions = [...prev];
          if (newPredictions[fileIndex]) {
            newPredictions[fileIndex] = {
              ...newPredictions[fileIndex],
              caption: lastMessage.caption,
              streamingCaption: lastMessage.caption,
              captionComplete: true,
            };
            console.log(
              "caption_complete - Final predictions:",
              newPredictions[fileIndex]
            );
          } else {
            console.log(
              "caption_complete - No prediction at index:",
              fileIndex
            );
          }
          return newPredictions;
        });

        // Increment completed count and check if all are done
        completedCaptionsRef.current += 1;
        console.log(
          `Completed ${completedCaptionsRef.current} of ${selectedFiles.length} captions`
        );

        if (completedCaptionsRef.current >= selectedFiles.length) {
          console.log("All captions complete, setting loading to false");
          setLoading(false);
          completedCaptionsRef.current = 0; // Reset for next batch
        }
        break;

      case "error":
        setError(lastMessage.error);
        setLoading(false);
        completedCaptionsRef.current = 0; // Reset on error
        break;
    }
  }, [lastMessage, selectedFiles]); // REMOVED predictions from dependencies

  useEffect(() => {
    console.log("Predictions state updated:", predictions);
  }, [predictions]);

  // Connect WebSocket when captions are enabled
  useEffect(() => {
    if (generateCaptions && useStreaming) {
      connect();
    } else {
      disconnect();
    }

    return () => disconnect();
  }, [generateCaptions, useStreaming, connect, disconnect]);

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
    if (selectedFiles.length === 0) {
      setError("Please select at least one image first");
      return;
    }

    setLoading(true);
    setError(null);

    // Use WebSocket for streaming captions
    if (generateCaptions && useStreaming && isConnected) {
      await handleClassifyWebSocket();
    } else {
      // Use HTTP for batch processing
      await handleClassifyHTTP();
    }
  };

  const handleClassifyWebSocket = async () => {
    try {
      // Reset completed count
      completedCaptionsRef.current = 0;

      // Initialize predictions array
      const initialPredictions: PredictionResult[] = selectedFiles.map(() => ({
        predictions: [],
        streamingCaption: "",
        captionComplete: false,
      }));
      setPredictions(initialPredictions);

      // Process each file
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];

        // First, get predictions via HTTP (fast)
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_BASE_URL}/predict`, {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();

          // Update predictions
          setPredictions((prev) => {
            const newPredictions = [...prev];
            newPredictions[i] = {
              ...newPredictions[i],
              predictions: data.predictions,
            };
            return newPredictions;
          });
        }

        // Then stream caption via WebSocket
        const reader = new FileReader();
        reader.onload = () => {
          const base64 = (reader.result as string).split(",")[1];

          sendMessage({
            type: "generate_caption",
            image_base64: base64,
            filename: file.name,
          });
        };
        reader.readAsDataURL(file);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Classification failed");
      setLoading(false);
      completedCaptionsRef.current = 0; // Reset on error
    }
  };

  const handleClassifyHTTP = async () => {
    if (selectedFiles.length === 0) {
      setError("Please select at least one image first");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append("files", file);
      });

      // Add caption generation parameter
      const url = generateCaptions
        ? `${API_BASE_URL}/predict-batch?generate_captions=true`
        : `${API_BASE_URL}/predict-batch`;

      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.status === "success" && data.results) {
        console.log("API Response:", data.results);

        // Store the full results including predictions and captions
        const fullResults: PredictionResult[] = data.results.map((r: any) => ({
          predictions: r.predictions,
          caption: r.caption,
        }));

        setPredictions(fullResults);
      } else {
        throw new Error("Invalid response format");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Classification failed");
      console.error("Classification error:", err);
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
      {/* WebSocket Connection Status */}
      {generateCaptions && useStreaming && (
        <div
          className={`ws-status ${isConnected ? "connected" : "disconnected"}`}
        >
          {isConnected ? "🟢 Streaming Ready" : "🔴 Connecting..."}
        </div>
      )}

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

      {/* Add caption toggle */}
      <div className="caption-toggle">
        <label>
          <input
            type="checkbox"
            checked={generateCaptions}
            onChange={(e) => setGenerateCaptions(e.target.checked)}
          />
          Generate creative captions ✨
        </label>
      </div>

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

      {/* Display results with captions */}
      {predictions.length > 0 && (
        <div className="results-section">
          {predictions.map((result, idx) => (
            <div key={idx} className="image-results">
              <div className="image-info">
                <img
                  src={previews[idx]}
                  alt={`Preview ${idx}`}
                  className="result-preview"
                />
                <h3>{selectedFiles[idx]?.name}</h3>
              </div>

              {/* Show caption with streaming effect */}
              {result.streamingCaption && (
                <div className="caption-box">
                  <p className="caption-text">
                    {result.streamingCaption}
                    {!result.captionComplete && (
                      <span className="cursor-blink">|</span>
                    )}
                  </p>
                </div>
              )}

              {/* Display all predictions */}
              <div className="predictions-list">
                {result.predictions.map((pred, predIdx) => (
                  <div
                    key={predIdx}
                    className={`prediction-item ${
                      predIdx === 0 ? "top-prediction" : ""
                    }`}
                  >
                    <div className="prediction-rank">
                      {predIdx === 0
                        ? "🥇"
                        : predIdx === 1
                        ? "🥈"
                        : predIdx === 2
                        ? "🥉"
                        : `${predIdx + 1}.`}
                    </div>
                    <div className="prediction-details">
                      <div className="class-name">{pred.class_name}</div>
                      <div className="class-id">Class ID: {pred.class_id}</div>
                    </div>
                    <div className="confidence-section">
                      <div
                        className="confidence-bar"
                        style={{
                          backgroundColor: getConfidenceColor(pred.confidence),
                          width: `${pred.confidence * 100}px`,
                        }}
                      ></div>
                      <div className="confidence-text">
                        {formatConfidence(pred.confidence)}%
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
