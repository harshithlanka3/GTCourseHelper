import React, { useState, useRef, useEffect } from 'react'
import { sendMessage } from '../services/api'
import MessageList from './MessageList'
import MessageInput from './MessageInput'
import './ChatInterface.css'

const INITIAL_ASSISTANT_MESSAGE = {
  role: 'assistant',
  content: 'Hello! I\'m your GT Course Helper. I can help you find courses, get recommendations, and build your schedule. What are you looking for?'
}

const createSessionId = () => {
  if (typeof window !== 'undefined' && window.crypto?.randomUUID) {
    return window.crypto.randomUUID()
  }
  return `session_${Date.now()}_${Math.random().toString(16).slice(2)}`
}

function ChatInterface() {
  const [messages, setMessages] = useState([INITIAL_ASSISTANT_MESSAGE])
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(() => createSessionId())
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async (messageText) => {
    if (!messageText.trim() || isLoading) return

    const userMessage = { role: 'user', content: messageText }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      const response = await sendMessage({
        message: messageText,
        session_id: sessionId
      })

      if (response.session_id && response.session_id !== sessionId) {
        setSessionId(response.session_id)
      }

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

  const handleResetConversation = () => {
    setMessages([INITIAL_ASSISTANT_MESSAGE])
    setSessionId(createSessionId())
  }

  return (
    <div className="ChatInterface">
      <MessageList messages={messages} isLoading={isLoading} />
      <div ref={messagesEndRef} />
      <div className="ChatInterface-footer">
        <div className="ChatInterface-controls">
          <button
            type="button"
            className="ChatInterface-reset"
            onClick={handleResetConversation}
            disabled={isLoading}
          >
            New Conversation
          </button>
        </div>
        <MessageInput onSend={handleSendMessage} disabled={isLoading} />
      </div>
    </div>
  )
}

export default ChatInterface

