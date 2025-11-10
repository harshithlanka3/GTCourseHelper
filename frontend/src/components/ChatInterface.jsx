import React, { useState, useRef, useEffect } from 'react'
import { sendMessage } from '../services/api'
import MessageList from './MessageList'
import MessageInput from './MessageInput'
import './ChatInterface.css'

function ChatInterface() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m your GT Course Helper. I can help you find courses, get recommendations, and build your schedule. What are you looking for?'
    }
  ])
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async (messageText) => {
    if (!messageText.trim() || isLoading) return

    // Add user message
    const userMessage = { role: 'user', content: messageText }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      // Build conversation history
      const conversationHistory = messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }))

      // Send to API
      const response = await sendMessage({
        message: messageText,
        session_id: sessionId,
        conversation_history: conversationHistory
      })

      // Update session ID if new
      if (response.session_id && !sessionId) {
        setSessionId(response.session_id)
      }

      // Add assistant response
      const assistantMessage = {
        role: 'assistant',
        content: response.response || response.recommendations || 'I apologize, but I couldn\'t generate a response.'
      }
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="ChatInterface">
      <MessageList messages={messages} isLoading={isLoading} />
      <div ref={messagesEndRef} />
      <MessageInput onSend={handleSendMessage} disabled={isLoading} />
    </div>
  )
}

export default ChatInterface

