import React, { useState } from 'react'
import './MessageInput.css'

function MessageInput({ onSend, disabled }) {
  const [input, setInput] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (input.trim() && !disabled) {
      onSend(input)
      setInput('')
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <form className="MessageInput" onSubmit={handleSubmit}>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder="Ask about courses, get recommendations, or mention specific course IDs..."
        disabled={disabled}
        className="MessageInput-field"
      />
      <button
        type="submit"
        disabled={disabled || !input.trim()}
        className="MessageInput-button"
      >
        Send
      </button>
    </form>
  )
}

export default MessageInput

