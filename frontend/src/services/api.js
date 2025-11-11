import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const sendMessage = async (data) => {
  const response = await api.post('/api/chat', data)
  return response.data
}

export const searchCourses = async (query, topK = 10) => {
  const response = await api.post('/api/search', {
    query,
    top_k: topK,
    use_gpt: true,
    use_id_matching: true,
  })
  return response.data
}

export const getRecommendations = async (query, topK = 50) => {
  const response = await api.post('/api/recommend', {
    query,
    top_k: topK,
    use_gpt_query: true,
  })
  return response.data
}

// Session Management Functions
export const createSession = async (title = null) => {
  const response = await api.post('/api/session', { title })
  return response.data
}

export const listSessions = async () => {
  const response = await api.get('/api/sessions')
  return response.data
}

export const getSessionHistory = async (sessionId) => {
  const response = await api.get(`/api/session/${sessionId}`)
  return response.data
}

export const deleteSession = async (sessionId) => {
  const response = await api.delete(`/api/session/${sessionId}`)
  return response.data
}

export const renameSession = async (sessionId, title) => {
  const response = await api.patch(`/api/session/${sessionId}`, { title })
  return response.data
}

export default api

