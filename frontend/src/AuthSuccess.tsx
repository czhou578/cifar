import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

export default function AuthSuccess() {
  const [token, setToken] = useState<string | null>(null);
  const location = useLocation();

  useEffect(() => {
    const query = new URLSearchParams(location.search);
    const t = query.get("token");
    if (t) {
      localStorage.setItem("gh_token", t);
      setToken(t);
    }
  }, [location.search]);

  return token ? <p>Logged in! Token stored.</p> : <p>Logging in…</p>;
}
