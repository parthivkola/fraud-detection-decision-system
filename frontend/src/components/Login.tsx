import React, { useState } from 'react';
import { Lock, User as UserIcon, Mail, Shield, ArrowRight, AlertCircle, CheckCircle } from 'lucide-react';
import { api, type User } from '../api';

interface LoginProps {
  onLoginSuccess: (user: User) => void;
}

type Tab = 'login' | 'register';

export const Login: React.FC<LoginProps> = ({ onLoginSuccess }) => {
  const [tab, setTab] = useState<Tab>('login');

  const [loginUsername, setLoginUsername] = useState('');
  const [loginPassword, setLoginPassword] = useState('');

  const [regUsername, setRegUsername] = useState('');
  const [regEmail, setRegEmail]       = useState('');
  const [regPassword, setRegPassword] = useState('');

  const [error, setError]   = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const switchTab = (t: Tab) => { setTab(t); setError(null); };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await api.login(loginUsername, loginPassword);
      const user = await api.getMe();
      onLoginSuccess(user);
    } catch (err: any) {
      setError(err.message || 'Login failed. Please check your credentials.');
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
      await api.login(regUsername, regPassword);
      const user = await api.getMe();
      onLoginSuccess(user);
    } catch (err: any) {
      setError(err.message || 'Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 'calc(100vh - 57px)', padding: 24, background: '#fafafa' }}>
      <div className="fade-in" style={{ width: '100%', maxWidth: 400 }}>

        {/* Brand header */}
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 48, height: 48, borderRadius: 12, background: '#171717', marginBottom: 16 }}>
            <Shield size={22} color="#fff" />
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, letterSpacing: '-0.02em', color: '#171717', marginBottom: 6 }}>
            Fraud Detection System
          </h1>
          <p style={{ fontSize: 13, color: '#737373' }}>
            Enterprise-grade transaction intelligence
          </p>
        </div>

        {/* Card */}
        <div className="card" style={{ padding: 28 }}>

          {/* Tabs */}
          <div style={{ display: 'flex', gap: 4, marginBottom: 24, background: '#f5f5f5', padding: 4, borderRadius: 8 }}>
            {(['login', 'register'] as Tab[]).map(t => (
              <button
                key={t}
                onClick={() => switchTab(t)}
                style={{
                  flex: 1,
                  padding: '6px 12px',
                  borderRadius: 6,
                  border: 'none',
                  background: tab === t ? '#ffffff' : 'transparent',
                  color: tab === t ? '#171717' : '#737373',
                  fontSize: 13,
                  fontWeight: tab === t ? 600 : 400,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  fontFamily: 'inherit',
                  boxShadow: tab === t ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                }}
              >
                {t === 'login' ? 'Sign In' : 'Register'}
              </button>
            ))}
          </div>

          {/* Error */}
          {error && (
            <div className="alert alert-error fade-in">
              <AlertCircle size={14} /><span>{error}</span>
            </div>
          )}

          {/* Login Form */}
          {tab === 'login' && (
            <form onSubmit={handleLogin}>
              <div className="input-group">
                <label className="input-label" htmlFor="login-username">Username</label>
                <div style={{ position: 'relative' }}>
                  <UserIcon size={14} style={{ position: 'absolute', left: 10, top: 10, color: '#a3a3a3' }} />
                  <input
                    id="login-username"
                    type="text"
                    className="input-field"
                    style={{ paddingLeft: 32 }}
                    value={loginUsername}
                    onChange={e => setLoginUsername(e.target.value)}
                    required
                    placeholder="Enter your username"
                    autoComplete="username"
                  />
                </div>
              </div>

              <div className="input-group" style={{ marginBottom: 20 }}>
                <label className="input-label" htmlFor="login-password">Password</label>
                <div style={{ position: 'relative' }}>
                  <Lock size={14} style={{ position: 'absolute', left: 10, top: 10, color: '#a3a3a3' }} />
                  <input
                    id="login-password"
                    type="password"
                    className="input-field"
                    style={{ paddingLeft: 32 }}
                    value={loginPassword}
                    onChange={e => setLoginPassword(e.target.value)}
                    required
                    placeholder="Enter your password"
                    autoComplete="current-password"
                  />
                </div>
              </div>

              <button
                type="submit"
                className="btn btn-primary btn-lg"
                style={{ width: '100%', justifyContent: 'center' }}
                disabled={loading}
              >
                {loading ? (
                  <><div className="spin" style={{ width: 14, height: 14, border: '2px solid rgba(255,255,255,0.3)', borderTopColor: '#fff', borderRadius: '50%' }} /><span>Signing in…</span></>
                ) : (
                  <><span>Sign in</span><ArrowRight size={14} /></>
                )}
              </button>

              {/* Hint */}
              <div style={{ marginTop: 20, padding: '10px 12px', background: '#f5f5f5', borderRadius: 8, border: '1px solid #e5e5e5' }}>
                <div style={{ fontSize: 11, color: '#a3a3a3', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 4 }}>Demo credentials</div>
                <div style={{ display: 'flex', gap: 12, fontSize: 12, fontFamily: 'var(--font-mono)', color: '#525252' }}>
                  <span>user: <strong style={{ color: '#171717' }}>admin</strong></span>
                  <span>pass: <strong style={{ color: '#171717' }}>admin123</strong></span>
                </div>
              </div>
            </form>
          )}

          {/* Register Form */}
          {tab === 'register' && (
            <form onSubmit={handleRegister}>
              <div className="input-group">
                <label className="input-label" htmlFor="reg-username">Username</label>
                <div style={{ position: 'relative' }}>
                  <UserIcon size={14} style={{ position: 'absolute', left: 10, top: 10, color: '#a3a3a3' }} />
                  <input
                    id="reg-username"
                    type="text"
                    className="input-field"
                    style={{ paddingLeft: 32 }}
                    value={regUsername}
                    onChange={e => setRegUsername(e.target.value)}
                    required
                    placeholder="Choose a username"
                    autoComplete="username"
                  />
                </div>
              </div>

              <div className="input-group">
                <label className="input-label" htmlFor="reg-email">Email</label>
                <div style={{ position: 'relative' }}>
                  <Mail size={14} style={{ position: 'absolute', left: 10, top: 10, color: '#a3a3a3' }} />
                  <input
                    id="reg-email"
                    type="email"
                    className="input-field"
                    style={{ paddingLeft: 32 }}
                    value={regEmail}
                    onChange={e => setRegEmail(e.target.value)}
                    required
                    placeholder="you@example.com"
                    autoComplete="email"
                  />
                </div>
              </div>

              <div className="input-group" style={{ marginBottom: 20 }}>
                <label className="input-label" htmlFor="reg-password">Password</label>
                <div style={{ position: 'relative' }}>
                  <Lock size={14} style={{ position: 'absolute', left: 10, top: 10, color: '#a3a3a3' }} />
                  <input
                    id="reg-password"
                    type="password"
                    className="input-field"
                    style={{ paddingLeft: 32 }}
                    value={regPassword}
                    onChange={e => setRegPassword(e.target.value)}
                    required
                    minLength={6}
                    placeholder="Minimum 6 characters"
                    autoComplete="new-password"
                  />
                </div>
              </div>

              <button
                type="submit"
                className="btn btn-primary btn-lg"
                style={{ width: '100%', justifyContent: 'center' }}
                disabled={loading}
              >
                {loading ? (
                  <><div className="spin" style={{ width: 14, height: 14, border: '2px solid rgba(255,255,255,0.3)', borderTopColor: '#fff', borderRadius: '50%' }} /><span>Creating account…</span></>
                ) : (
                  <><span>Create Account</span><ArrowRight size={14} /></>
                )}
              </button>

              <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#a3a3a3' }}>
                <CheckCircle size={12} color="#16a34a" />
                New accounts are assigned the <span style={{ fontFamily: 'var(--font-mono)', color: '#525252' }}>analyst</span> role.
              </div>
            </form>
          )}
        </div>

        <p style={{ textAlign: 'center', marginTop: 20, fontSize: 12, color: '#a3a3a3' }}>
          Protected by JWT &amp; BCrypt · XGBoost Inference Engine v2.0
        </p>
      </div>
    </div>
  );
};
