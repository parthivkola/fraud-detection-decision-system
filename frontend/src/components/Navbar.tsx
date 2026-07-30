import React from 'react';
import { Shield, LogOut, BarChart2, Cpu, Upload, User as UserIcon } from 'lucide-react';
import type { User } from '../api';

interface NavbarProps {
  user: User | null;
  activeTab: string;
  setActiveTab: (tab: string) => void;
  onLogout: () => void;
}

const tabs = [
  { id: 'predict',  label: 'Inference',      Icon: Upload    },
  { id: 'metrics',  label: 'Metrics',         Icon: BarChart2 },
  { id: 'models',   label: 'Model Registry',  Icon: Cpu       },
];

export const Navbar: React.FC<NavbarProps> = ({ user, activeTab, setActiveTab, onLogout }) => {
  return (
    <header style={{
      background: '#ffffff',
      borderBottom: '1px solid #e5e5e5',
      position: 'sticky',
      top: 0,
      zIndex: 1000,
    }}>
      <div className="container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: 56 }}>

        {/* Brand */}
        <button
          onClick={() => setActiveTab('predict')}
          style={{ display: 'flex', alignItems: 'center', gap: 10, background: 'none', border: 'none', cursor: 'pointer', padding: 0, textDecoration: 'none' }}
        >
          <div style={{ width: 32, height: 32, borderRadius: 8, background: '#171717', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Shield size={16} color="#fff" />
          </div>
          <div style={{ textAlign: 'left' }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#171717', letterSpacing: '-0.02em', lineHeight: 1.2 }}>
              Fraud Detection
            </div>
            <div style={{ fontSize: 10, color: '#a3a3a3', fontWeight: 500, letterSpacing: '0.04em', textTransform: 'uppercase' }}>
              Decision System
            </div>
          </div>
        </button>

        {user && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            {/* Tab nav */}
            <nav style={{ display: 'flex', gap: 2 }}>
              {tabs.map(({ id, label, Icon }) => {
                const active = activeTab === id;
                return (
                  <button
                    key={id}
                    onClick={() => setActiveTab(id)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 6,
                      padding: '6px 12px',
                      borderRadius: 6,
                      border: 'none',
                      background: active ? '#f5f5f5' : 'transparent',
                      color: active ? '#171717' : '#737373',
                      fontSize: 13,
                      fontWeight: active ? 600 : 400,
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                      fontFamily: 'inherit',
                    }}
                    onMouseEnter={e => { if (!active) (e.currentTarget as HTMLButtonElement).style.color = '#171717'; }}
                    onMouseLeave={e => { if (!active) (e.currentTarget as HTMLButtonElement).style.color = '#737373'; }}
                  >
                    <Icon size={14} />
                    <span>{label}</span>
                  </button>
                );
              })}
            </nav>

            {/* Divider */}
            <div style={{ width: 1, height: 20, background: '#e5e5e5', margin: '0 8px' }} />

            {/* User info */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ width: 28, height: 28, borderRadius: '50%', background: '#f5f5f5', border: '1px solid #e5e5e5', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <UserIcon size={13} color="#737373" />
              </div>
              <div style={{ lineHeight: 1.2 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#171717' }}>{user.username}</div>
                <div style={{ fontSize: 10, color: user.role === 'admin' ? '#16a34a' : '#737373', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{user.role}</div>
              </div>
              <button
                onClick={onLogout}
                title="Sign out"
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 28, height: 28, borderRadius: 6, background: 'none', border: '1px solid #e5e5e5', cursor: 'pointer', color: '#a3a3a3', transition: 'all 0.15s ease' }}
                onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = '#fef2f2'; (e.currentTarget as HTMLButtonElement).style.color = '#dc2626'; (e.currentTarget as HTMLButtonElement).style.borderColor = '#fecaca'; }}
                onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = 'none'; (e.currentTarget as HTMLButtonElement).style.color = '#a3a3a3'; (e.currentTarget as HTMLButtonElement).style.borderColor = '#e5e5e5'; }}
              >
                <LogOut size={13} />
              </button>
            </div>
          </div>
        )}

        {!user && (
          <div style={{ fontSize: 12, color: '#a3a3a3', fontWeight: 500 }}>XGBoost Inference Engine</div>
        )}
      </div>
    </header>
  );
};
