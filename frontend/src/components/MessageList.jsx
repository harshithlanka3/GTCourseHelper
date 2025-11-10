import React from 'react'
import Message from './Message'
import './MessageList.css'

function MessageList({ messages, isLoading }) {
  return (
    <div className="MessageList">
      {messages.map((message, index) => (
        <Message key={index} message={message} />
      ))}
      {isLoading && (
        <div className="Message Message-assistant">
          <div className="Message-content">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default MessageList

