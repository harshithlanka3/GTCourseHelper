import React, { useState, useEffect } from 'react'
import Sidebar from '../components/Sidebar'
import ChatInterface from '../components/ChatInterface'
import { createSession, listSessions, getSessionHistory, deleteSession, renameSession } from '../services/api'
import './ChatPage.css'

function ChatPage() {
  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState(null)
  const [initialMessages, setInitialMessages] = useState(null)
  const [isLoadingSessions, setIsLoadingSessions] = useState(true)

  // Load sessions on mount
  useEffect(() => {
    loadSessions()
  }, [])

  // Load session history when active session changes
  useEffect(() => {
    if (activeSessionId) {
      // Reset to null first to trigger re-render, then load history
      setInitialMessages(null)
      loadSessionHistory(activeSessionId)
    } else {
      setInitialMessages(null)
    }
  }, [activeSessionId])

  const loadSessions = async () => {
    try {
      setIsLoadingSessions(true)
      const sessionList = await listSessions()
      setSessions(sessionList)
      
      // If no active session and we have sessions, select the first one
      // Only auto-select if we're not in the middle of creating a new session
      if (!activeSessionId && sessionList.length > 0) {
        const firstSession = sessionList[0]
        if (firstSession) {
          setActiveSessionId(firstSession.session_id)
        }
      }
    } catch (error) {
      console.error('Error loading sessions:', error)
    } finally {
      setIsLoadingSessions(false)
    }
  }

  const loadSessionHistory = async (sessionId) => {
    try {
      const history = await getSessionHistory(sessionId)
      // Ensure we have the messages array - it should include both user and assistant messages
      if (history && history.messages) {
        console.log('Loaded session history:', history.messages.length, 'messages')
        console.log('Message roles:', history.messages.map(m => m.role))
        setInitialMessages(history.messages)
      } else {
        console.warn('No messages found in history for session:', sessionId)
        setInitialMessages([])
      }
    } catch (error) {
      console.error('Error loading session history:', error)
      setInitialMessages([])
    }
  }

  const handleNewChat = async () => {
    try {
      const newSession = await createSession()
      setActiveSessionId(newSession.session_id)
      await loadSessions()
    } catch (error) {
      console.error('Error creating new session:', error)
    }
  }

  const handleSelectSession = (sessionId) => {
    setActiveSessionId(sessionId)
  }

  const handleDeleteSession = async (sessionId) => {
    try {
      await deleteSession(sessionId)
      await loadSessions()
      
      // If we deleted the active session, clear it or select another
      if (sessionId === activeSessionId) {
        if (sessions.length > 1) {
          const remainingSessions = sessions.filter(s => s.session_id !== sessionId)
          setActiveSessionId(remainingSessions[0]?.session_id || null)
        } else {
          setActiveSessionId(null)
        }
      }
    } catch (error) {
      console.error('Error deleting session:', error)
    }
  }

  const handleRenameSession = async (sessionId, newTitle) => {
    try {
      await renameSession(sessionId, newTitle)
      await loadSessions()
    } catch (error) {
      console.error('Error renaming session:', error)
    }
  }

  // Update session list when a message is sent (to update timestamps)
  const handleMessageSent = () => {
    loadSessions()
  }

  return (
    <div className="ChatPage">
      <Sidebar
        conversations={sessions}
        activeSessionId={activeSessionId}
        onSelect={handleSelectSession}
        onNew={handleNewChat}
        onDelete={handleDeleteSession}
        onRename={handleRenameSession}
      />
      <div className="ChatPage-main">
        {activeSessionId ? (
          <ChatInterface
            key={activeSessionId}
            sessionId={activeSessionId}
            initialMessages={initialMessages}
            onMessageSent={handleMessageSent}
          />
        ) : (
          <div className="ChatPage-empty">
            <p>No active chat. Create a new chat to get started!</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default ChatPage

