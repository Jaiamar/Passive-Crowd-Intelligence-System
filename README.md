# Passive Crowd Intelligence System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/poython-3.10%2B-blue)
![React](https://img.shields.io/badge/react-19.0-blue)
![Next.js](https://img.shields.io/badge/next.js-16.2-black)
![FastAPI](https://img.shields.io/badge/fastapi-1.0-green)
![YOLO](https://img.shields.io/badge/YOLO-26-orange)

Passive Crowd Intelligence System is a full-stack platform designed to analyze crowd density and patterns using a combination of computer vision and cellular network data. The application features a modern real-time dashboard for displaying video analytics and insights.

## 🚀 Features

- **Real-Time Video Analytics**: Integrates the state-of-the-art YOLO26 model and OpenCV to detect and track individuals in real-time video feeds.
- **Cellular Network Density**: Combines visual data with cellular network density analytics for a holistic understanding of crowd movements.
- **Interactive Dashboard**: A responsive, dynamic Next.js React frontend to visualize crowd patterns and statistics.
- **High-Performance Backend**: FastAPI and Uvicorn providing a scalable, asynchronous API.

## 🛠️ Tech Stack

### Frontend
- [Next.js 16](https://nextjs.org/)
- [React 19](https://react.dev/)
- [TailwindCSS 4](https://tailwindcss.com/)
- TypeScript

### Backend
- [FastAPI](https://fastapi.tiangolo.com/)
- [YOLO26 / Ultralytics](https://github.com/ultralytics/ultralytics) (Computer Vision)
- [OpenCV](https://opencv.org/) (Image Processing)
- Python 3.10+
- Uvicorn & WebSockets

## 📂 Project Structure

```
.
├── backend/              # FastAPI Python backend
│   ├── api/              # API routes (cellular_routes, video_routes)
│   ├── main.py           # FastAPI entry point
│   ├── requirements.txt  # Python dependencies
│   └── venv/             # Python Virtual Environment
├── frontend/             # Next.js React frontend
│   ├── src/              # Source code, components (VideoAnalytics.tsx)
│   ├── package.json      # Node.js dependencies
│   └── ...
├── start-backend.bat     # Helper script to launch the backend
├── start-frontend.bat    # Helper script to launch the frontend
└── README.md             # Project documentation
```

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.10+
- Node.js & npm
- (Optional) Iriun Webcam for remote video camera integration

### 1. Clone the repository
```bash
git clone https://github.com/Jaiamar/Passive-Crowd-Intelligence-System.git
cd Passive-Crowd-Intelligence-System
```

### 2. Configure Environment variables
Create a `.env` file in the `backend/` directory and configure the necessary variables for your setup (if applicable).

### 3. Startup Scripts (Windows Recommended)
We provide batch scripts to make running the project seamless automatically.

**Run the Backend:**
Simply execute:
```bash
.\start-backend.bat
```
*(This will activate the virtual environment and start the Uvicorn server on port 8000)*

**Run the Frontend:**
Simply execute:
```bash
.\start-frontend.bat
```
*(This will install npm dependencies if needed and start the Next.js dev server on port 3000)*

### Manual Startup

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## 🔗 APIs
- Backend API Docs (Swagger UI) are available at `http://localhost:8000/docs` while the backend is running.

## 📄 License
This project is licensed under the MIT License.