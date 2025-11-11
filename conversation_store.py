# conversation_store.py  (root)
from typing import List, Dict, Optional
from threading import Lock
from uuid import uuid4
from datetime import datetime

class ConversationStore:
    def __init__(self):
        self._store: Dict[str, List[Dict[str, str]]] = {}
        self._meta: Dict[str, Dict[str, str]] = {}
        self._lock = Lock()

    def create(self, title: Optional[str] = None) -> str:
        with self._lock:
            sid = str(uuid4())
            self._store[sid] = []
            self._meta[sid] = {
                "title": title or "New chat",
                "updated_at": datetime.utcnow().isoformat()
            }
            return sid

    def rename(self, session_id: str, title: str):
        with self._lock:
            if session_id in self._meta:
                self._meta[session_id]["title"] = title

    def list(self) -> List[Dict[str, str]]:
        with self._lock:
            out = []
            for sid, meta in self._meta.items():
                out.append({
                    "session_id": sid,
                    "title": meta.get("title", "New chat"),
                    "updated_at": meta.get("updated_at", "")
                })
            # newest first
            out.sort(key=lambda x: x["updated_at"], reverse=True)
            return out

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            return self._store.get(session_id, []).copy()

    def add_message(self, session_id: str, role: str, content: str):
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = []
                self._meta[session_id] = {"title": "New chat", "updated_at": datetime.utcnow().isoformat()}
            self._store[session_id].append({"role": role, "content": content})
            # First user message becomes a better title
            if role == "user" and (self._meta[session_id].get("title") == "New chat"):
                snippet = content.strip().splitlines()[0][:40]
                self._meta[session_id]["title"] = snippet or "New chat"
            self._meta[session_id]["updated_at"] = datetime.utcnow().isoformat()

    def clear(self, session_id: str):
        with self._lock:
            self._store.pop(session_id, None)
            self._meta.pop(session_id, None)

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._meta

conversation_store = ConversationStore()