import "./App.css";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useNavigate,
  useLocation,
} from "react-router-dom";
import ImageClassifier from "./components/ImageClassifier";
import Login from "./components/Login";
import AuthSuccess from "./AuthSuccess";

const AppContent: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleLoginClick = () => {
    navigate("/login");
  };

  // Don't show the main header on the login page
  if (location.pathname === "/login") {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
      </Routes>
    );
  }

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <div className="header-text">
            <h1>CIFAR-100 Image Classifier</h1>
            <p>Upload an image to classify it using our trained model</p>
          </div>
          <button onClick={handleLoginClick} className="login-button">
            Login
          </button>
        </div>
      </header>
      <main className="App-main">
        <Routes>
          <Route path="/" element={<ImageClassifier />} />
          <Route path="/auth/success" element={<AuthSuccess />} />
        </Routes>
      </main>
    </div>
  );
};

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;
