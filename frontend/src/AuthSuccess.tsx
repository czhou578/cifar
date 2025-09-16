import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

export default function AuthSuccess() {
  const [token, setToken] = useState<string | null>(null);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    const query = new URLSearchParams(location.search);
    const t = query.get("token");
    if (t) {
      localStorage.setItem("gh_token", t);
      setToken(t);

      // Redirect to home page after 2 seconds
      setTimeout(() => {
        navigate("/");
      }, 2000);
    }
  }, [location.search, navigate]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "60vh",
        textAlign: "center",
      }}
    >
      {token ? (
        <div>
          <h2>✅ Login Successful!</h2>
          <p>Welcome! You have been successfully authenticated.</p>
          <p>Redirecting to home page...</p>
        </div>
      ) : (
        <div>
          <h2>🔄 Completing Login...</h2>
          <p>Please wait while we complete your authentication.</p>
        </div>
      )}
    </div>
  );
}
