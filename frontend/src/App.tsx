import React from "react";
import "./App.css";
import ImageClassifier from "./components/ImageClassifier";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>CIFAR-100 Image Classifier</h1>
        <p>Upload an image to classify it using our trained model</p>
      </header>
      <main className="App-main">
        <ImageClassifier />
      </main>
    </div>
  );
}

export default App;
