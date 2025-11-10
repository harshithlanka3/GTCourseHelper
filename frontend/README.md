# GT Course Helper - Frontend

React-based chat interface for GT Course Helper.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm run dev
```

The frontend will run on `http://localhost:3000`

## Environment Variables

Create a `.env` file in the frontend directory (optional):
```
VITE_API_URL=http://localhost:8000
```

If not set, it defaults to `http://localhost:8000`

## Building for Production

```bash
npm run build
```