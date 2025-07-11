# Frontend to Backend API Mapping

This document lists the API endpoints invoked by the frontend and the corresponding backend routes. It serves as a quick reference to ensure parity between the two layers.

| Frontend Service / Page | Endpoint | Backend Route |
|-------------------------|----------|---------------|
| `authService.js` | `POST /api/auth/local/register` | `main.py::register_local_user` |
|                     | `POST /api/auth/local/login` | `main.py::login_local_user` |
|                     | `GET /api/auth/me` | `main.py::get_current_user_info_unified_endpoint` |
|                     | `POST /api/auth/logout` | `main.py::logout_unified` |
| `cspApiService.js`  | `POST /api/auth/azure-login` | `main.py::azure_login` |
|                     | `POST /api/auth/refresh` | `main.py::refresh_token_local` |
|                     | `GET /api/designs` | `api/endpoints/designs.py::list_designs` |
|                     | `GET /api/designs/{id}` | `api/endpoints/designs.py::get_design` |
|                     | `POST /api/designs` | `api/endpoints/designs.py::create_design` |
|                     | `PUT /api/designs/{id}` | `api/endpoints/designs.py::update_design` |
|                     | `DELETE /api/designs/{id}` | `api/endpoints/designs.py::delete_design` |
|                     | `GET /api/components` | `api/endpoints/designs.py::list_components` |
|                     | `GET /api/components/{type}` | `api/endpoints/designs.py::get_component` |
|                     | `POST /api/executions/execute` | `main.py::start_execution` |
|                     | `GET /api/executions/{id}/status` | `main.py::get_execution_status` |
|                     | `GET /api/executions/{id}/results` | `main.py::get_execution_results` |

If new frontend features require additional endpoints, they should be added to the backend and documented here to maintain alignment.
