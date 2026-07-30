import React, { useState, useEffect } from 'react';
import './index.css';
import { Navbar }    from './components/Navbar';
import { Login }     from './components/Login';
import { Dashboard } from './components/Dashboard';
import { api, type User } from './api';

export const App: React.FC = () => {
  const [user, setUser]         = useState<User | null>(null);
  const [activeTab, setActiveTab] = useState<string>('predict');
  const [loading, setLoading]   = useState(true);

  useEffect(() => { checkAuth(); }, []);

  const checkAuth = async () => {
    const token = localStorage.getItem('token');
    if (!token) { setLoading(false); return; }
    try {
      const u = await api.getMe();
      setUser(u);
    } catch {
      localStorage.removeItem('token');
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setUser(null);
    setActiveTab('predict');
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', background: '#fafafa', flexDirection: 'column', gap: 16 }}>
        <div style={{ width: 36, height: 36, border: '2px solid #e5e5e5', borderTopColor: '#171717', borderRadius: '50%', animation: 'spin 0.9s linear infinite' }} />
        <p style={{ fontSize: 13, color: '#a3a3a3' }}>Initialising system…</p>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#fafafa' }}>
      <Navbar user={user} activeTab={activeTab} setActiveTab={setActiveTab} onLogout={handleLogout} />

      <main style={{ flex: 1 }}>
        {user ? (
          <Dashboard activeTab={activeTab} />
        ) : (
          <Login onLoginSuccess={u => setUser(u)} />
        )}
      </main>

      <footer style={{ borderTop: '1px solid #e5e5e5', background: '#ffffff', padding: '14px 0' }}>
        <div className="container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
          <span style={{ fontSize: 12, color: '#a3a3a3' }}>Credit Card Fraud Detection System · XGBoost Inference Engine v2.0</span>
          <span style={{ fontSize: 12, color: '#a3a3a3', fontFamily: 'var(--font-mono)' }}>JWT &amp; BCrypt · FastAPI</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
