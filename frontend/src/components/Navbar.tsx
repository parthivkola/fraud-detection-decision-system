import React from 'react';
import { Shield, LogOut, Activity, Cpu, UploadCloud, User as UserIcon } from 'lucide-react';
import type { User } from '../api';

interface NavbarProps {
  user: User | null;
  activeTab: string;
  setActiveTab: (tab: string) => void;
  onLogout: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ user, activeTab, setActiveTab, onLogout }) => {
  return (
    <header style={{
      background: '#0c0d12',
      borderBottom: '1px solid #27272a',
      position: 'sticky',
      top: 0,
      zIndex: 1000,
      padding: '0.75rem 0'
    }}>
      <div className="container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }} onClick={() => setActiveTab('predict')}>
          <div style={{
            background: '#1c1e2b',
            border: '1px solid #27272a',
            padding: '0.4rem',
            borderRadius: '6px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Shield size={20} color="#3b82f6" />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f4f4f5', letterSpacing: '-0.02em' }}>
              CC Fraud Detection
            </span>
            <span style={{ fontSize: '0.75rem', color: '#71717a', background: '#14161f', padding: '0.15rem 0.5rem', borderRadius: '4px', border: '1px solid #27272a', fontWeight: 500 }}>
              XGBoost Engine
            </span>
          </div>
        </div>

        {user ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
            <nav style={{ display: 'flex', gap: '0.25rem', background: '#14161f', padding: '0.2rem', borderRadius: '8px', border: '1px solid #27272a' }}>
              <button
                onClick={() => setActiveTab('predict')}
                style={{
                  background: activeTab === 'predict' ? '#232636' : 'transparent',
                  color: activeTab === 'predict' ? '#f4f4f5' : '#71717a',
                  border: 'none',
                  padding: '0.4rem 0.875rem',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontFamily: 'var(--font-main)',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  transition: 'all 0.15s ease'
                }}
              >
                <UploadCloud size={15} />
                Inference
              </button>

              <button
                onClick={() => setActiveTab('metrics')}
                style={{
                  background: activeTab === 'metrics' ? '#232636' : 'transparent',
                  color: activeTab === 'metrics' ? '#f4f4f5' : '#71717a',
                  border: 'none',
                  padding: '0.4rem 0.875rem',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontFamily: 'var(--font-main)',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  transition: 'all 0.15s ease'
                }}
              >
                <Activity size={15} />
                Telemetry
              </button>

              <button
                onClick={() => setActiveTab('models')}
                style={{
                  background: activeTab === 'models' ? '#232636' : 'transparent',
                  color: activeTab === 'models' ? '#f4f4f5' : '#71717a',
                  border: 'none',
                  padding: '0.4rem 0.875rem',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontFamily: 'var(--font-main)',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  transition: 'all 0.15s ease'
                }}
              >
                <Cpu size={15} />
                Model Registry
              </button>
            </nav>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', borderLeft: '1px solid #27272a', paddingLeft: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ width: 28, height: 28, borderRadius: '6px', background: '#1c1e2b', border: '1px solid #27272a', display: 'flex', alignItems: 'center', justifyItems: 'center', justifyContent: 'center' }}>
                  <UserIcon size={14} color="#a1a1aa" />
                </div>
                <div>
                  <div style={{ fontSize: '0.8125rem', fontWeight: 500, color: '#f4f4f5', lineHeight: 1.2 }}>{user.username}</div>
                  <div style={{ fontSize: '0.6875rem', color: '#10b981', textTransform: 'uppercase', fontWeight: 600 }}>{user.role}</div>
                </div>
              </div>

              <button
                onClick={onLogout}
                className="btn btn-secondary"
                style={{ padding: '0.4rem 0.6rem', background: 'transparent', border: '1px solid #27272a', color: '#71717a' }}
                title="Logout"
              >
                <LogOut size={16} />
              </button>
            </div>
          </div>
        ) : (
          <div style={{ fontSize: '0.8125rem', color: '#71717a' }}>Enterprise System Portal</div>
        )}
      </div>
    </header>
  );
};
