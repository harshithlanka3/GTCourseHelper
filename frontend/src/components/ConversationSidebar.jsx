import React, { useState } from 'react'
import './ConversationSidebar.css'

function ConversationSidebar({ 
  conversations, 
  currentConversationId, 
  onSelectConversation, 
  onNewConversation, 
  onDeleteConversation,
  isOpen,
  onToggle 
}) {
  const [hoveredId, setHoveredId] = useState(null)

  const formatDate = (timestamp) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now - date
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays === 1) return 'Yesterday'
    if (diffDays < 7) return `${diffDays}d ago`
    
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  }

  const getConversationTitle = (conv) => {
    if (conv.title) return conv.title
    
    const userMessages = conv.messages.filter(m => m.role === 'user')
    if (userMessages.length > 0) {
      const firstMessage = userMessages[0].content
      return firstMessage.length > 30 ? firstMessage.substring(0, 30) + '...' : firstMessage
    }
    
    return 'New Conversation'
  }

  return (
    <>
      <button 
        className={`ConversationSidebar-toggle ${isOpen ? 'open' : ''}`}
        onClick={onToggle}
        aria-label="Toggle sidebar"
      >
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <path d="M3 10h14M3 5h14M3 15h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
        </svg>
      </button>

      <div className={`ConversationSidebar ${isOpen ? 'open' : ''}`}>
        <div className="ConversationSidebar-header">
          <h2>Conversations</h2>
          <button 
            className="ConversationSidebar-new"
            onClick={onNewConversation}
            aria-label="New conversation"
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M10 5v10M5 10h10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>

        <div className="ConversationSidebar-list">
          {conversations.length === 0 ? (
            <div className="ConversationSidebar-empty">
              No conversations yet
            </div>
          ) : (
            conversations.map((conv) => (
              <div
                key={conv.id}
                className={`ConversationSidebar-item ${
                  conv.id === currentConversationId ? 'active' : ''
                }`}
                onClick={() => onSelectConversation(conv.id)}
                onMouseEnter={() => setHoveredId(conv.id)}
                onMouseLeave={() => setHoveredId(null)}
              >
                <div className="ConversationSidebar-item-content">
                  <div className="ConversationSidebar-item-title">
                    {getConversationTitle(conv)}
                  </div>
                  <div className="ConversationSidebar-item-meta">
                    {formatDate(conv.updatedAt)}
                  </div>
                </div>
                {hoveredId === conv.id && conv.id !== currentConversationId && (
                  <button
                    className="ConversationSidebar-item-delete"
                    onClick={(e) => {
                      e.stopPropagation()
                      onDeleteConversation(conv.id)
                    }}
                    aria-label="Delete conversation"
                  >
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                      <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                  </button>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </>
  )
}

export default ConversationSidebar
