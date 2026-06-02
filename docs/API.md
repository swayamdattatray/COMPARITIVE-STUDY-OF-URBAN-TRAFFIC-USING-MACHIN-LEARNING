# CodeMentor AI API Routes

Base URL: `/api/v1`

| Method | Route | Auth | Purpose |
| --- | --- | --- | --- |
| GET | `/health` | Public | Health check |
| POST | `/auth/demo-login` | Public | Demo JWT login placeholder for email/Google flows |
| GET | `/trainers` | Public | List specialized AI trainers |
| POST | `/trainers/:slug/chat` | Student/Trainer/Admin | Ask trainer questions and receive progress-aware guidance |
| POST | `/trainers/:slug/learning-path` | Student/Trainer/Admin | Generate personalized learning paths |
| GET | `/tests/milestone?progress=75` | Student/Trainer/Admin | Return milestone tests unlocked every 25% |
| POST | `/tests/weekly-saturday` | Trainer/Admin | Create weekly Saturday assessment coverage |
| POST | `/assessments/analyze` | Student/Trainer/Admin | Detect weak areas, revision plan, next topics, careers |
| POST | `/certificates` | Admin | Generate certificate with unique ID and QR verification |
| GET | `/admin/analytics` | Admin | View platform analytics |
