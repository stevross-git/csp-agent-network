# Enhanced CSP Frontend

This directory contains the single-page application (SPA) used to administer the Enhanced CSP system.  It integrates with Azure Active Directory for authentication and communicates with the FastAPI backend.

## Requirements
- Node.js 18+ and npm
- Python 3.8+ (for the bundled development server)

## Installation
Install the JavaScript dependencies and prepare environment variables:

```bash
cd enhanced_csp/frontend
npm install
cp .env.example .env  # then edit values as needed
```

## Running the Development Server
Launch the local server which serves the static files and proxies API requests:

```bash
npm start
```

By default it listens on `http://localhost:3000`. Useful pages include:

- `http://localhost:3000/pages/login.html` – login screen
- `http://localhost:3000/csp_admin_portal.html` – main dashboard

The server exposes `/health` and `/config.js` endpoints for basic status checks.

## Testing
End‑to‑end tests reside in the `cypress/` directory and can be started with:

```bash
npx cypress open
```

## Directory Overview

```
frontend/
├── components/          # React UI components
├── config/              # Authentication and role definitions
├── css/                 # Stylesheets
├── cypress/             # Cypress tests
├── hooks/               # Custom React hooks
├── js/                  # Core logic and page scripts
├── middleware/          # Express-style middleware
├── pages/               # HTML entry points
├── services/            # API wrappers and session handling
├── stores/              # Zustand state stores
├── test-server.py       # Development server
└── package.json
```

For backend setup details see [`../backend/README.md`](../backend/README.md).
