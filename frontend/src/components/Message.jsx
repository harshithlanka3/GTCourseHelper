import React from 'react'
import ReactMarkdown from 'react-markdown'
import './Message.css'

function Message({ message }) {
  const isUser = message.role === 'user'
  
  return (
    <div className={`Message Message-${message.role}`}>
      <div className="Message-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="Message-content">
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <div className="Message-markdown">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  )
}

export default Message

