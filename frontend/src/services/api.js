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

export default api

