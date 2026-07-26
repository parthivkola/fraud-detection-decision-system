import React, { useState } from 'react';
import { Lock, User as UserIcon, Shield, ArrowRight, AlertCircle } from 'lucide-react';
import { api, type User } from '../api';

interface LoginProps {
  onLoginSuccess: (user: User) => void;
}

export const Login: React.FC<LoginProps> = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('admin123');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await api.login(username, password);
      const user = await api.getMe();
      onLoginSuccess(user);
    } catch (err: any) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '80vh', padding: '1.5rem' }}>
      <div className="card" style={{ maxWidth: '400px', width: '100%', padding: '2rem', background: '#14161f', border: '1px solid #27272a' }}>
        <div style={{ textAlign: 'center', marginBottom: '1.75rem' }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 48,
            height: 48,
            borderRadius: '10px',
            background: '#1c1e2b',
            border: '1px solid #27272a',
            marginBottom: '1rem'
          }}>
            <Shield size={24} color="#3b82f6" />
          </div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 600, color: '#f4f4f5', marginBottom: '0.375rem' }}>
            Sign in to Aegis
          </h2>
          <p style={{ color: '#71717a', fontSize: '0.875rem' }}>
            Enterprise XGBoost Fraud Detection System
          </p>
        </div>

        {error && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.2)',
            color: '#ef4444',
            padding: '0.75rem 1rem',
            borderRadius: 'var(--radius-md)',
            marginBottom: '1.25rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.625rem',
            fontSize: '0.875rem'
          }}>
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <label className="input-label">Username</label>
            <div style={{ position: 'relative' }}>
              <UserIcon size={16} style={{ position: 'absolute', left: 12, top: 11, color: '#71717a' }} />
              <input
                type="text"
                className="input-field"
                style={{ paddingLeft: '2.25rem' }}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                placeholder="Enter username"
              />
            </div>
          </div>

          <div className="input-group" style={{ marginBottom: '1.5rem' }}>
            <label className="input-label">Password</label>
            <div style={{ position: 'relative' }}>
              <Lock size={16} style={{ position: 'absolute', left: 12, top: 11, color: '#71717a' }} />
              <input
                type="password"
                className="input-field"
                style={{ paddingLeft: '2.25rem' }}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                placeholder="Enter password"
              />
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            style={{ width: '100%', padding: '0.75rem', fontSize: '0.875rem', justifyContent: 'center' }}
            disabled={loading}
          >
            {loading ? 'Authenticating...' : (
              <>
                <span>Sign in</span>
                <ArrowRight size={16} />
              </>
            )}
          </button>
        </form>

        <div style={{
          marginTop: '1.5rem',
          paddingTop: '1rem',
          borderTop: '1px solid #27272a',
          textAlign: 'center',
          fontSize: '0.75rem',
          color: '#71717a'
        }}>
          <div>Default Admin Credentials:</div>
          <div style={{ display: 'inline-flex', gap: '0.75rem', marginTop: '0.375rem', background: '#11131a', padding: '0.25rem 0.625rem', borderRadius: '4px', border: '1px solid #27272a', color: '#a1a1aa', fontFamily: 'var(--font-mono)' }}>
            <span>user: admin</span>
            <span>|</span>
            <span>pass: admin123</span>
          </div>
        </div>
      </div>
    </div>
  );
};
