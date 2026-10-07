import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import Sidebar from './components/Sidebar'

// Pages
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import DashboardPage from './pages/DashboardPage'
import DatasetUploadPage from './pages/DatasetUploadPage'
import ModelTrainingPage from './pages/ModelTrainingPage'
import DetectionPage from './pages/DetectionPage'
import EvaluationPage from './pages/EvaluationPage'
import PredictionHistoryPage from './pages/PredictionHistoryPage'
import AboutPage from './pages/AboutPage'

function ProtectedRoute({ children }) {
  const { user, loading } = useAuth()
  if (loading) return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh' }}><div className="loading-spinner" /></div>
  if (!user) return <Navigate to="/login" replace />
  return children
}

function AppLayout({ children }) {
  return (
    <div className="app-layout">
      <Sidebar />
      <div className="main-content">
        <div className="page-content">
          {children}
        </div>
      </div>
    </div>
  )
}

function AppRoutes() {
  const { user } = useAuth()

  return (
    <Routes>
      <Route path="/login" element={user ? <Navigate to="/" replace /> : <LoginPage />} />
      <Route path="/register" element={user ? <Navigate to="/" replace /> : <RegisterPage />} />

      <Route path="/" element={
        <ProtectedRoute>
          <AppLayout><DashboardPage /></AppLayout>
        </ProtectedRoute>
      } />
      <Route path="/upload" element={
        <ProtectedRoute>
          <AppLayout><DatasetUploadPage /></AppLayout>
        </ProtectedRoute>
      } />
      <Route path="/train" element={
        <ProtectedRoute>
          <AppLayout><ModelTrainingPage /></AppLayout>
        </ProtectedRoute>
      } />
      <Route path="/detect" element={
        <ProtectedRoute>
          <AppLayout><DetectionPage /></AppLayout>
        </ProtectedRoute>
      } />
      <Route path="/evaluate" element={
        <ProtectedRoute>
          <AppLayout><EvaluationPage /></AppLayout>
        </ProtectedRoute>
      } />
      <Route path="/history" element={
        <ProtectedRoute>
          <AppLayout><PredictionHistoryPage /></AppLayout>
        </ProtectedRoute>
      } />
      <Route path="/about" element={
        <ProtectedRoute>
          <AppLayout><AboutPage /></AppLayout>
        </ProtectedRoute>
      } />

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  )
}
