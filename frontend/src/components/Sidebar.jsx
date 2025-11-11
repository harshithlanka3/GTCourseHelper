import React, { useState } from "react";
import "./Sidebar.css";

export default function Sidebar({ conversations, activeSessionId, onSelect, onNew, onDelete, onRename }) {
  const [editingId, setEditingId] = useState(null);
  const [editValue, setEditValue] = useState("");

  const handleRenameClick = (e, conv) => {
    e.stopPropagation();
    setEditingId(conv.session_id);
    setEditValue(conv.title);
  };

  const handleRenameSubmit = (e, sessionId) => {
    e.stopPropagation();
    if (editValue.trim() && onRename) {
      onRename(sessionId, editValue.trim());
    }
    setEditingId(null);
    setEditValue("");
  };

  const handleRenameCancel = (e) => {
    e.stopPropagation();
    setEditingId(null);
    setEditValue("");
  };

  const handleDeleteClick = (e, sessionId) => {
    e.stopPropagation();
    if (window.confirm("Are you sure you want to delete this chat?")) {
      onDelete(sessionId);
    }
  };

  return (
    <div className="Sidebar">
      <h2>Conversations</h2>
      
      <button className="Sidebar-new" onClick={onNew}>
        + New Chat
      </button>

      <ul className="Sidebar-list">
        {conversations.map(conv => (
          <li
            key={conv.session_id}
            className={conv.session_id === activeSessionId ? "active" : ""}
            onClick={() => onSelect(conv.session_id)}
          >
            {editingId === conv.session_id ? (
              <div className="Sidebar-edit" onClick={(e) => e.stopPropagation()}>
                <input
                  type="text"
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      handleRenameSubmit(e, conv.session_id);
                    } else if (e.key === "Escape") {
                      handleRenameCancel(e);
                    }
                  }}
                  autoFocus
                  className="Sidebar-edit-input"
                />
                <button
                  className="Sidebar-edit-save"
                  onClick={(e) => handleRenameSubmit(e, conv.session_id)}
                >
                  ✓
                </button>
                <button
                  className="Sidebar-edit-cancel"
                  onClick={handleRenameCancel}
                >
                  ✕
                </button>
              </div>
            ) : (
              <div className="Sidebar-item">
                <span className="Sidebar-item-title">{conv.title || "Untitled Chat"}</span>
                <div className="Sidebar-item-actions">
                  <button
                    className="Sidebar-item-rename"
                    onClick={(e) => handleRenameClick(e, conv)}
                    title="Rename"
                  >
                    ✎
                  </button>
                  <button
                    className="Sidebar-item-delete"
                    onClick={(e) => handleDeleteClick(e, conv.session_id)}
                    title="Delete"
                  >
                    ×
                  </button>
                </div>
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}