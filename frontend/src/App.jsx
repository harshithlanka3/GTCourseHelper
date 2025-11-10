import React, { useState } from 'react'
import ChatInterface from './components/ChatInterface'
import './App.css'

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>GT Course Helper</h1>
        <p>Your intelligent course recommendation assistant</p>
      </header>
      <main className="App-main">
        <ChatInterface />
      </main>
    </div>
  )
}

export default App

