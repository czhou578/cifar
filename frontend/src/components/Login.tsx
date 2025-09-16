import React from "react";
import { useNavigate } from "react-router-dom";
import "./Login.css";

const Login: React.FC = () => {
  const navigate = useNavigate();

  const handleBackToHome = () => {
    navigate("/");
  };

  const handleLogin = () => {
    // Simply hit our FastAPI endpoint which redirects to GitHub
    window.location.href = "http://localhost:8000/login/github";
  };

  return (
    <div className="login-page">
      <div className="login-header">
        <button onClick={handleBackToHome} className="back-button">
          ← Back to Home
        </button>
        <h1>Login</h1>
      </div>
      <div className="login-container">
        <div className="login-form">
          <h2>Welcome Back</h2>
          <form>
            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input
                type="email"
                id="email"
                name="email"
                placeholder="Enter your email"
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                id="password"
                name="password"
                placeholder="Enter your password"
                required
              />
            </div>
            <button
              type="submit"
              className="login-submit-button"
              onClick={handleLogin}
            >
              Login
            </button>
          </form>
          <div className="login-footer">
            <p>
              Don't have an account? <a href="#signup">Sign up</a>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
