import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import {
  LayoutDashboard,
  Upload,
  Cpu,
  Radar,
  BarChart2,
  History,
  Info,
  Satellite,
  LogOut,
} from 'lucide-react'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/upload', icon: Upload, label: 'Upload Dataset' },
  { to: '/train', icon: Cpu, label: 'Model Training' },
  { to: '/detect', icon: Radar, label: 'Detection' },
  { to: '/evaluate', icon: BarChart2, label: 'Evaluation' },
  { to: '/history', icon: History, label: 'Prediction History' },
  { to: '/about', icon: Info, label: 'About Project' },
]

export default function Sidebar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
          <Satellite size={24} color="var(--accent-cyan)" />
          <h1>GNSS Spoofing<br />Detection</h1>
        </div>
        <p>Bi-GRU Deep Learning · MCA Project</p>
      </div>

      <nav className="sidebar-nav">
        <div className="sidebar-section-label">Navigation</div>
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-user">
          <strong>{user?.name || 'User'}</strong>
          {user?.email}
        </div>
        <button
          onClick={handleLogout}
          style={{
            marginTop: 12,
            width: '100%',
            padding: '8px 12px',
            background: 'rgba(239,68,68,0.1)',
            border: '1px solid rgba(239,68,68,0.2)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--accent-red)',
            cursor: 'pointer',
            fontSize: '0.825rem',
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
          }}
        >
          <LogOut size={15} />
          Sign Out
        </button>
      </div>
    </aside>
  )
}
