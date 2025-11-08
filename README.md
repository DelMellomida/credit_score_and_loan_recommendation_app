# Credit Score and Loan Recommendation System

A modern, AI-powered loan processing system that combines credit scoring with intelligent loan product recommendations. This application helps financial institutions streamline their loan application process while providing data-driven decisions and personalized loan recommendations.

Link: <https://thesis-credit-score-loan-applicatio.vercel.app/>
API: <https://thesis-credit-score-loan-application-6haa.onrender.com>

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Environment Setup](#environment-setup)
  - [Frontend Setup](#frontend-setup)
  - [Backend Setup](#backend-setup)
- [Usage](#usage)
  - [Development Mode](#development-mode)
  - [Docker Deployment](#docker-deployment)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Secure Authentication**: JWT-based user authentication system
- **Smart Loan Processing**: AI-powered credit scoring and loan recommendation
- **Document Management**: Secure document upload and storage system
- **Real-time Analysis**: Instant credit scoring and loan product matching
- **Interactive Dashboard**: User-friendly interface for loan officers
- **Comprehensive Reporting**: Detailed applicant assessment reports
- **Multi-factor Analysis**: Considers various factors for credit scoring
- **Automated Workflows**: Streamlined loan application processing

## Technologies Used

### Frontend
- Next.js 15.5.3
- React 19.1.0
- TypeScript
- Tailwind CSS
- Radix UI Components
- Sonner for Notifications
- JWT Authentication

### Backend
- FastAPI
- Python 3.11+
- MongoDB with Beanie ODM
- Google Generative AI
- scikit-learn
- JWT for Authentication
- Supabase for File Storage

### DevOps
- Docker
- Docker Compose
- Git

## Project Structure

```
credit_score_and_loan_recommendation_app/
├── client/                 # Frontend Next.js application
│   ├── src/
│   │   ├── app/           # Next.js pages and layouts
│   │   ├── components/    # React components
│   │   ├── context/       # React context providers
│   │   ├── lib/           # Utility functions and types
│   │   └── styles/        # Styles
│   └── public/            # Static assets
├── server/                # Backend FastAPI application
│   ├── app/
│   │   ├── api/          # API routes
│   │   ├── core/         # Core configurations
│   │   ├── database/     # Database models and connection
│   │   ├── services/     # Business logic
│   │   └── utils/        # Utility functions
│   ├── models/           # ML models and analysis
│   └── scripts/          # Data processing scripts
```

## Prerequisites

- Python 3.11 or higher
- Node.js 18.x or higher
- MongoDB
- Google Cloud API key for Generative AI
- Supabase account for file storage

## Installation

### Environment Setup

1. Clone the repository:
\`\`\`bash
git clone https://github.com/DelMellomida/credit_score_and_loan_recommendation_app.git
cd credit_score_and_loan_recommendation_app
\`\`\`

### Frontend Setup

1. Navigate to the client directory:
\`\`\`bash
cd client
\`\`\`

2. Install dependencies:
\`\`\`bash
npm install
\`\`\`

3. Create a .env.local file:
\`\`\`
NEXT_PUBLIC_API_URL=http://localhost:8000
\`\`\`

### Backend Setup

1. Navigate to the server directory:
\`\`\`bash
cd server
\`\`\`

2. Create a virtual environment:
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. Create a .env file:
\`\`\`
MONGODB_URI=your_mongodb_uri
MONGODB_DB_NAME=credit-score-and-loan-recommendation
GEMINI_API_KEY=your_gemini_api_key
JWT_SECRET_KEY=your_jwt_secret
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_PUBLIC=your_supabase_anon_key
SUPABASE_SERVICE_ROLE=your_supabase_service_key
CLIENT_URL=http://localhost:3000,http://localhost:3001
\`\`\`

## Usage

### Development Mode

1. Start the backend server:
\`\`\`bash
cd server
uvicorn main:app --reload --port 8000
\`\`\`

2. Start the frontend development server:
\`\`\`bash
cd client
npm run dev
\`\`\`

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

### Docker Deployment

1. Build and start the containers:
\`\`\`bash
docker compose up --build
\`\`\`

The application will be available at:
- Frontend: http://localhost:3001
- Backend API: http://localhost:9003

## API Documentation

### Authentication Endpoints
- POST /auth/register - Register a new user
- POST /auth/login - User login
- POST /auth/refresh - Refresh access token

### Loan Application Endpoints
- POST /loan/apply - Submit new loan application
- GET /loan/applications - Get all loan applications
- GET /loan/application/{id} - Get specific application
- PUT /loan/application/{id} - Update application
- DELETE /loan/application/{id} - Delete application

### Document Endpoints
- POST /documents/upload - Upload application documents
- GET /documents/{application_id} - Get application documents
- DELETE /documents/{application_id} - Delete application documents

### Model Endpoints
- POST /model/predict - Get credit score prediction
- GET /model/recommendations - Get loan recommendations

## Contributing

1. Fork the repository
2. Create your feature branch (\`git checkout -b feature/YourFeature\`)
3. Commit your changes (\`git commit -m 'Add some feature'\`)
4. Push to the branch (\`git push origin feature/YourFeature\`)
5. Open a Pull Request

---

Thesis by: 
- Shana Eve Gonzales
- Louise Guiaya
- Jhondel Mellomida
