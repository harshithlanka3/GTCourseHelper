import React, { useState, useRef, useEffect } from 'react'
import { sendMessage } from '../services/api'
import MessageList from './MessageList'
import MessageInput from './MessageInput'
import ConversationSidebar from './ConversationSidebar'
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

const createNewConversation = () => ({
  id: createSessionId(),
  sessionId: createSessionId(),
  messages: [INITIAL_ASSISTANT_MESSAGE],
  createdAt: Date.now(),
  updatedAt: Date.now(),
  title: null
})

const STORAGE_KEY = 'gt_course_helper_conversations'
const CURRENT_CONVERSATION_KEY = 'gt_course_helper_current_conversation'

function ChatInterface() {
  const [conversations, setConversations] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        const parsed = JSON.parse(stored)
        return parsed.length > 0 ? parsed : [createNewConversation()]
      }
    } catch (error) {
      console.error('Error loading conversations:', error)
    }
    return [createNewConversation()]
  })

  const [currentConversationId, setCurrentConversationId] = useState(() => {
    try {
      const stored = localStorage.getItem(CURRENT_CONVERSATION_KEY)
      if (stored && conversations.find(c => c.id === stored)) {
        return stored
      }
    } catch (error) {
      console.error('Error loading current conversation:', error)
    }
    return conversations[0]?.id
  })

  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const currentConversation = conversations.find(c => c.id === currentConversationId) || conversations[0]
  const messages = currentConversation?.messages || [INITIAL_ASSISTANT_MESSAGE]
  const sessionId = currentConversation?.sessionId || createSessionId()

  // Save conversations to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations))
    } catch (error) {
      console.error('Error saving conversations:', error)
    }
  }, [conversations])

  // Save current conversation ID
  useEffect(() => {
    try {
      localStorage.setItem(CURRENT_CONVERSATION_KEY, currentConversationId)
    } catch (error) {
      console.error('Error saving current conversation:', error)
    }
  }, [currentConversationId])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async (messageText) => {
    if (!messageText.trim() || isLoading) return

    const userMessage = { role: 'user', content: messageText }
    
    // Update current conversation with new message
    setConversations(prev => prev.map(conv => 
      conv.id === currentConversationId
        ? { ...conv, messages: [...conv.messages, userMessage], updatedAt: Date.now() }
        : conv
    ))
    
    setIsLoading(true)

    try {
      const response = await sendMessage({
        message: messageText,
        session_id: sessionId
      })

      const assistantMessage = {
        role: 'assistant',
        content: response.response || response.recommendations || 'I apologize, but I couldn\'t generate a response.'
      }
      
      // Update conversation with assistant response
      setConversations(prev => prev.map(conv => 
        conv.id === currentConversationId
          ? { 
              ...conv, 
              messages: [...conv.messages, assistantMessage],
              updatedAt: Date.now(),
              sessionId: response.session_id || conv.sessionId
            }
          : conv
      ))
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      }
      setConversations(prev => prev.map(conv => 
        conv.id === currentConversationId
          ? { ...conv, messages: [...conv.messages, errorMessage], updatedAt: Date.now() }
          : conv
      ))
    } finally {
      setIsLoading(false)
    }
  }

  const handleNewConversation = () => {
    const newConv = createNewConversation()
    setConversations(prev => [newConv, ...prev])
    setCurrentConversationId(newConv.id)
  }

  const handleSelectConversation = (conversationId) => {
    setCurrentConversationId(conversationId)
    setIsSidebarOpen(false)
  }

  const handleDeleteConversation = (conversationId) => {
    setConversations(prev => {
      const filtered = prev.filter(c => c.id !== conversationId)
      
      // If deleting current conversation, switch to another one
      if (conversationId === currentConversationId) {
        const nextConv = filtered[0] || createNewConversation()
        setCurrentConversationId(nextConv.id)
        if (filtered.length === 0) {
          return [nextConv]
        }
      }
      
      return filtered.length > 0 ? filtered : [createNewConversation()]
    })
  }

  const handleToggleSidebar = () => {
    setIsSidebarOpen(prev => !prev)
  }

  return (
    <>
      <ConversationSidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        onDeleteConversation={handleDeleteConversation}
        isOpen={isSidebarOpen}
        onToggle={handleToggleSidebar}
      />
      
      <div className={`ChatInterface ${isSidebarOpen ? 'sidebar-open' : ''}`}>
        <MessageList messages={messages} isLoading={isLoading} />
        <div ref={messagesEndRef} />
        <div className="ChatInterface-footer">
          <div className="ChatInterface-controls">
            <button
              type="button"
              className="ChatInterface-reset"
              onClick={handleNewConversation}
              disabled={isLoading}
            >
              New Conversation
            </button>
          </div>
          <MessageInput onSend={handleSendMessage} disabled={isLoading} />
        </div>
      </div>
    </>
  )
}

export default ChatInterface

