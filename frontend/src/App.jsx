import React from 'react'
import ChatPage from './pages/ChatPage'
import './App.css'

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>GT Course Helper</h1>
        <p>Your intelligent course recommendation assistant</p>
      </header>
      <main className="App-main">
        <ChatPage />
      </main>
    </div>
  )
}

export default App

