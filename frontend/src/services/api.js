import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

// Attach JWT token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('gnss_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle 401 globally
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('gnss_token')
      localStorage.removeItem('gnss_user')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// ---- Auth ----
export const authAPI = {
  register: (data) => api.post('/auth/register', data),
  login: (data) => api.post('/auth/login', data),
}

// ---- Dataset ----
export const datasetAPI = {
  upload: (formData) =>
    api.post('/dataset/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  list: () => api.get('/dataset/list'),
  get: (id) => api.get(`/dataset/${id}`),
  downloadTemplate: () =>
    api.get('/dataset/sample-template', { responseType: 'blob' }),
}

// ---- Model ----
export const modelAPI = {
  train: (datasetId) => api.post('/model/train', { dataset_id: datasetId }),
  status: () => api.get('/model/status'),
  metrics: () => api.get('/model/metrics'),
}

// ---- Prediction ----
export const predictionAPI = {
  predict: (formData) =>
    api.post('/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  history: () => api.get('/predictions'),
}

// ---- Dashboard ----
export const dashboardAPI = {
  summary: () => api.get('/dashboard/summary'),
}

export default api
