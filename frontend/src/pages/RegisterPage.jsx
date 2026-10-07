import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { authAPI } from '../services/api'
import { Satellite, User, Mail, Lock } from 'lucide-react'

export default function RegisterPage() {
  const [form, setForm] = useState({ name: '', email: '', password: '', confirm_password: '' })
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    if (form.password !== form.confirm_password) {
      setError('Passwords do not match.')
      return
    }
    if (form.password.length < 6) {
      setError('Password must be at least 6 characters.')
      return
    }
    setLoading(true)
    try {
      await authAPI.register(form)
      setSuccess('Registration successful! Redirecting to login...')
      setTimeout(() => navigate('/login'), 1800)
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-wrapper">
      <div className="auth-card">
        <div className="auth-logo">
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 12 }}>
            <div style={{ background: 'rgba(6,182,212,0.15)', padding: 14, borderRadius: 12 }}>
              <Satellite size={36} color="var(--accent-cyan)" />
            </div>
          </div>
          <h1>GNSS Spoofing Detection</h1>
          <p>Create your research account</p>
        </div>

        <h2 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: 24 }}>Register</h2>

        {error && <div className="alert alert-error">{error}</div>}
        {success && <div className="alert alert-success">{success}</div>}

        <form onSubmit={handleSubmit}>
          {[
            { field: 'name', label: 'Full Name', type: 'text', placeholder: 'Your name', Icon: User },
            { field: 'email', label: 'Email Address', type: 'email', placeholder: 'you@example.com', Icon: Mail },
            { field: 'password', label: 'Password', type: 'password', placeholder: '••••••••', Icon: Lock },
            { field: 'confirm_password', label: 'Confirm Password', type: 'password', placeholder: '••••••••', Icon: Lock },
          ].map(({ field, label, type, placeholder, Icon }) => (
            <div className="form-group" key={field}>
              <label className="form-label">{label}</label>
              <div style={{ position: 'relative' }}>
                <Icon size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                <input
                  className="form-input"
                  style={{ paddingLeft: 38 }}
                  type={type}
                  placeholder={placeholder}
                  value={form[field]}
                  onChange={e => setForm({ ...form, [field]: e.target.value })}
                  required
                />
              </div>
            </div>
          ))}
          <button className="btn btn-primary w-full btn-lg" type="submit" disabled={loading}>
            {loading ? <><span className="loading-spinner" /> Registering...</> : 'Create Account'}
          </button>
        </form>

        <div className="auth-footer">
          Already have an account? <Link to="/login">Sign in</Link>
        </div>
      </div>
    </div>
  )
}
