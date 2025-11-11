import React, { useState, useRef, useEffect } from 'react'
import { sendMessage } from '../services/api'
import MessageList from './MessageList'
import MessageInput from './MessageInput'
import './ChatInterface.css'

function ChatInterface({ sessionId: propSessionId = null, initialMessages = null, onMessageSent = null }) {
  const defaultMessage = {
    role: 'assistant',
    content: 'Hello! I\'m your GT Course Helper. I can help you find courses, get recommendations, and build your schedule. What are you looking for?'
  }
  
  const [messages, setMessages] = useState(
    initialMessages && initialMessages.length > 0 
      ? initialMessages 
      : [defaultMessage]
  )
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(propSessionId)
  const messagesEndRef = useRef(null)

  // Update messages when initialMessages prop changes (session switch)
  useEffect(() => {
    if (initialMessages !== null) {
      setMessages(initialMessages.length > 0 ? initialMessages : [defaultMessage])
    }
  }, [initialMessages])

  // Update sessionId when prop changes
  useEffect(() => {
    setSessionId(propSessionId)
  }, [propSessionId])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async (messageText) => {
    if (!messageText.trim() || isLoading) return

    // Add user message to local state immediately for UI
    const userMessage = { role: 'user', content: messageText }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      // Build conversation history from current messages (before adding user message)
      // The backend stores messages independently, so this is just for context
      const conversationHistory = messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }))

      // Send to API - backend will store both user and assistant messages
      const response = await sendMessage({
        message: messageText,
        session_id: sessionId,
        conversation_history: conversationHistory
      })

      // Update session ID if new
      if (response.session_id && !sessionId) {
        setSessionId(response.session_id)
      }

      // Add assistant response to local state
      const assistantMessage = {
        role: 'assistant',
        content: response.response || response.recommendations || 'I apologize, but I couldn\'t generate a response.'
      }
      setMessages(prev => [...prev, assistantMessage])
      
      // Notify parent that a message was sent (for session list refresh)
      if (onMessageSent) {
        onMessageSent()
      }
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

