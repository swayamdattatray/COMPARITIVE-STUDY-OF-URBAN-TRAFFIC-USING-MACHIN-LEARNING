# CodeMentor AI

CodeMentor AI is a premium AI-powered EdTech SaaS platform for programming and technology education. It includes specialized AI trainers, adaptive learning paths, automated milestone and weekly Saturday tests, a coding playground, gamification, certificates, RBAC authentication, admin analytics, and production-oriented deployment assets.

## Project Architecture

```text
.
├── apps
│   ├── web                 # Next.js 15 + React + TypeScript + Tailwind UI
│   │   ├── app             # App Router pages and global styles
│   │   ├── components      # Premium dashboard/marketing components
│   │   └── lib             # Static SaaS data/configuration
│   └── api                 # Node.js + Express + TypeScript API
│       └── src
│           ├── config      # Environment validation
│           ├── controllers # Request handlers
│           ├── middleware  # JWT/RBAC protection
│           ├── routes      # API route registry
│           ├── services    # AI trainer, assessment, certificate logic
│           └── types       # Domain types
├── prisma/schema.prisma    # PostgreSQL database schema
├── docs/API.md             # API documentation
├── docs/ER_DIAGRAM.md      # Mermaid ER diagram
├── docker-compose.yml      # Local PostgreSQL + API orchestration
├── Dockerfile              # Production API image build
├── index.html              # Static landing-page preview
└── styles.css              # Static preview styling
```

## Core Features

- **AI Trainers:** Python, C, PHP, HTML/CSS/JavaScript, Full Stack, Backend, Machine Learning, DSA, Cyber Security, and Database Management trainers with OpenAI and LangChain/vector-database integration points.
- **Student Dashboard:** Profile data, enrolled courses, learning progress, weekly hours, upcoming tests, certificates, AI recommendations, and recent activity.
- **Course Management:** Beginner, intermediate, advanced levels with videos, AI tutor chat, notes, resources, coding playground, assignments, and progress tracking.
- **Automated Testing:** 25% milestone tests and AI-generated weekly Saturday tests with grading and performance analysis.
- **AI Assessment Engine:** Weak-area detection, revision plans, custom quizzes, completion prediction, and career-path recommendations.
- **Coding Playground:** Architecture supports Python, C, PHP, and JavaScript execution with AI review and bug-fix suggestions.
- **Gamification:** XP points, levels, badges, weekly leaderboard, streaks, and achievements.
- **Certificates:** Auto-generated after 100% completion and final assessment with unique certificate IDs and QR verification.
- **Admin Panel:** Students, courses, AI trainers, analytics, announcements, tests, and reports.

## API Routes

See [`docs/API.md`](docs/API.md) for complete routes, authorization requirements, and purpose.

## Database Schema

The PostgreSQL schema in [`prisma/schema.prisma`](prisma/schema.prisma) models users, RBAC roles, student profiles, courses, levels, lessons, AI trainers, enrollments, progress, assignments, tests, submissions, certificates, and activities. The ER diagram is available in [`docs/ER_DIAGRAM.md`](docs/ER_DIAGRAM.md).

## Local Development

```bash
npm install
npm run dev
```

Run services with Docker:

```bash
docker compose up --build
```

## Environment Variables

```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/codementor_ai
JWT_SECRET=replace-with-a-secure-secret
OPENAI_API_KEY=your-openai-api-key
PORT=4000
```

## Deployment Guide

1. Deploy `apps/web` to Vercel and set the production API base URL.
2. Deploy `apps/api` to Railway or AWS using the provided Dockerfile.
3. Provision PostgreSQL and set `DATABASE_URL`.
4. Run Prisma migrations from the API deployment pipeline.
5. Configure Google OAuth, email verification provider, and secure JWT secrets.
6. Connect a vector database for course knowledge retrieval and populate embeddings per course.
7. Enable HTTPS, rate limiting, monitoring, backups, and audit logging before production launch.
