import React, { useState } from 'react';
import { Lock, User as UserIcon, Mail, Shield, ArrowRight, AlertCircle, CheckCircle } from 'lucide-react';
import { api, type User } from '../api';

interface LoginProps {
  onLoginSuccess: (user: User) => void;
}

type Tab = 'login' | 'register';

const tabStyle = (active: boolean): React.CSSProperties => ({
  flex: 1,
  padding: '0.5rem',
  background: active ? '#1e2130' : 'transparent',
  border: 'none',
  borderBottom: active ? '2px solid #3b82f6' : '2px solid transparent',
  color: active ? '#f4f4f5' : '#71717a',
  fontWeight: active ? 600 : 400,
  fontSize: '0.875rem',
  cursor: 'pointer',
  transition: 'all 0.15s ease',
});

export const Login: React.FC<LoginProps> = ({ onLoginSuccess }) => {
  const [tab, setTab] = useState<Tab>('login');

  // Login state
  const [loginUsername, setLoginUsername] = useState('');
  const [loginPassword, setLoginPassword] = useState('');

  // Register state
  const [regUsername, setRegUsername] = useState('');
  const [regEmail, setRegEmail]       = useState('');
  const [regPassword, setRegPassword] = useState('');
  const [regSuccess, setRegSuccess]   = useState(false);

  const [error, setError]   = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const switchTab = (t: Tab) => { setTab(t); setError(null); setRegSuccess(false); };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await api.login(loginUsername, loginPassword);
      const user = await api.getMe();
      onLoginSuccess(user);
    } catch (err: any) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await api.register(regUsername, regEmail, regPassword);
      // Auto-login after successful registration
      await api.login(regUsername, regPassword);
      const user = await api.getMe();
      onLoginSuccess(user);
    } catch (err: any) {
      setError(err.message || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  const inputIcon = { position: 'absolute' as const, left: 12, top: 11, color: '#71717a' };

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '80vh', padding: '1.5rem' }}>
      <div className="card" style={{ maxWidth: '420px', width: '100%', padding: '2rem', background: '#14161f', border: '1px solid #27272a' }}>

        {/* Brand */}
        <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            width: 48, height: 48, borderRadius: '10px',
            background: '#1c1e2b', border: '1px solid #27272a', marginBottom: '1rem'
          }}>
            <Shield size={24} color="#3b82f6" />
          </div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 600, color: '#f4f4f5', marginBottom: '0.375rem' }}>
            Credit Card Fraud Detection System
          </h2>
          <p style={{ color: '#71717a', fontSize: '0.875rem' }}>
            XGBoost-powered real-time fraud detection
          </p>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', borderBottom: '1px solid #27272a', marginBottom: '1.5rem' }}>
          <button style={tabStyle(tab === 'login')}  onClick={() => switchTab('login')}>Sign In</button>
          <button style={tabStyle(tab === 'register')} onClick={() => switchTab('register')}>Register</button>
        </div>

        {/* Error */}
        {error && (
          <div style={{
            background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)',
            color: '#ef4444', padding: '0.75rem 1rem', borderRadius: 'var(--radius-md)',
            marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.625rem', fontSize: '0.875rem'
          }}>
            <AlertCircle size={16} /><span>{error}</span>
          </div>
        )}

        {/* Success */}
        {regSuccess && (
          <div style={{
            background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.2)',
            color: '#22c55e', padding: '0.75rem 1rem', borderRadius: 'var(--radius-md)',
            marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.625rem', fontSize: '0.875rem'
          }}>
            <CheckCircle size={16} /><span>Account created! Signing you in…</span>
          </div>
        )}

        {/* ── Login Form ── */}
        {tab === 'login' && (
          <form onSubmit={handleLogin}>
            <div className="input-group">
              <label className="input-label">Username</label>
              <div style={{ position: 'relative' }}>
                <UserIcon size={16} style={inputIcon} />
                <input type="text" className="input-field" style={{ paddingLeft: '2.25rem' }}
                  value={loginUsername} onChange={e => setLoginUsername(e.target.value)}
                  required placeholder="Enter username" />
              </div>
            </div>

            <div className="input-group" style={{ marginBottom: '1.5rem' }}>
              <label className="input-label">Password</label>
              <div style={{ position: 'relative' }}>
                <Lock size={16} style={inputIcon} />
                <input type="password" className="input-field" style={{ paddingLeft: '2.25rem' }}
                  value={loginPassword} onChange={e => setLoginPassword(e.target.value)}
                  required placeholder="Enter password" />
              </div>
            </div>

            <button type="submit" className="btn btn-primary"
              style={{ width: '100%', padding: '0.75rem', fontSize: '0.875rem', justifyContent: 'center' }}
              disabled={loading}>
              {loading ? 'Authenticating…' : <><span>Sign in</span><ArrowRight size={16} /></>}
            </button>

            {/* Credentials hint */}
            <div style={{ marginTop: '1.25rem', textAlign: 'center', fontSize: '0.75rem', color: '#71717a' }}>
              <div>Default Admin Credentials</div>
              <div style={{ display: 'inline-flex', gap: '0.75rem', marginTop: '0.375rem', background: '#11131a', padding: '0.25rem 0.625rem', borderRadius: '4px', border: '1px solid #27272a', color: '#a1a1aa', fontFamily: 'var(--font-mono)' }}>
                <span>user: admin</span><span>|</span><span>pass: admin123</span>
              </div>
            </div>
          </form>
        )}

        {/* ── Register Form ── */}
        {tab === 'register' && (
          <form onSubmit={handleRegister}>
            <div className="input-group">
              <label className="input-label">Username</label>
              <div style={{ position: 'relative' }}>
                <UserIcon size={16} style={inputIcon} />
                <input type="text" className="input-field" style={{ paddingLeft: '2.25rem' }}
                  value={regUsername} onChange={e => setRegUsername(e.target.value)}
                  required placeholder="Choose a username" />
              </div>
            </div>

            <div className="input-group">
              <label className="input-label">Email</label>
              <div style={{ position: 'relative' }}>
                <Mail size={16} style={inputIcon} />
                <input type="email" className="input-field" style={{ paddingLeft: '2.25rem' }}
                  value={regEmail} onChange={e => setRegEmail(e.target.value)}
                  required placeholder="you@example.com" />
              </div>
            </div>

            <div className="input-group" style={{ marginBottom: '1.5rem' }}>
              <label className="input-label">Password</label>
              <div style={{ position: 'relative' }}>
                <Lock size={16} style={inputIcon} />
                <input type="password" className="input-field" style={{ paddingLeft: '2.25rem' }}
                  value={regPassword} onChange={e => setRegPassword(e.target.value)}
                  required minLength={6} placeholder="Min. 6 characters" />
              </div>
            </div>

            <button type="submit" className="btn btn-primary"
              style={{ width: '100%', padding: '0.75rem', fontSize: '0.875rem', justifyContent: 'center' }}
              disabled={loading}>
              {loading ? 'Creating account…' : <><span>Create Account</span><ArrowRight size={16} /></>}
            </button>

            <p style={{ marginTop: '1rem', textAlign: 'center', fontSize: '0.75rem', color: '#71717a' }}>
              New accounts are assigned the <span style={{ color: '#a1a1aa', fontFamily: 'var(--font-mono)' }}>analyst</span> role.
            </p>
          </form>
        )}

      </div>
    </div>
  );
};
