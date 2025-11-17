import React, { useState } from 'react'
import ChatInterface from './components/ChatInterface'
import FeedbackButton from './components/FeedbackButton'
import ViewReviews from './components/ViewReviews'
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
        <FeedbackButton />
        <ViewReviews />
      </main>
    </div>
  )
}

export default App

