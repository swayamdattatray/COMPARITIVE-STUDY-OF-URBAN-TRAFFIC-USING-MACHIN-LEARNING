import OpenAI from "openai";
import { env } from "../config/env.js";
import type { TrainerSlug } from "../types/domain.js";

const trainerProfiles: Record<TrainerSlug, string> = {
  python: "Python Development trainer focused on fundamentals, APIs, automation, OOP, testing, and production patterns.",
  c: "C Programming trainer focused on pointers, memory, files, algorithms, and systems-level thinking.",
  php: "PHP Development trainer focused on modern PHP, Laravel-ready patterns, security, and MySQL integration.",
  "html-css-js": "Frontend trainer focused on semantic HTML, responsive CSS, JavaScript, accessibility, and browser APIs.",
  "full-stack": "Full Stack trainer focused on end-to-end product building, deployments, architecture, and debugging.",
  backend: "Backend trainer focused on APIs, authentication, databases, queues, caching, observability, and security.",
  "machine-learning": "Machine Learning trainer focused on data prep, model selection, evaluation, deployment, and ethics.",
  dsa: "Data Structures & Algorithms trainer focused on patterns, complexity, problem solving, and interview drills.",
  "cyber-security": "Cyber Security trainer focused on secure coding, threat modeling, OWASP, and defensive labs.",
  database: "Database trainer focused on SQL/MySQL, schema design, indexing, transactions, and query optimization."
};

export class AiTrainerService {
  private client = env.OPENAI_API_KEY ? new OpenAI({ apiKey: env.OPENAI_API_KEY }) : undefined;

  listTrainers() {
    return Object.entries(trainerProfiles).map(([slug, description]) => ({ slug, description }));
  }

  async chat(slug: TrainerSlug, question: string, progress = 0) {
    const system = `${trainerProfiles[slug]} Teach step-by-step, include examples, challenges, assignments, progress-aware recommendations, and next topics.`;
    if (!this.client) {
      return { answer: `[Demo AI] ${system} Student progress: ${progress}%. Answering: ${question}`, nextTopics: ["Review fundamentals", "Complete a guided challenge", "Take a short adaptive quiz"] };
    }
    const response = await this.client.responses.create({
      model: "gpt-4.1-mini",
      input: [{ role: "system", content: system }, { role: "user", content: `Progress: ${progress}%. Question: ${question}` }]
    });
    return { answer: response.output_text, nextTopics: ["Practice assignment", "Milestone quiz", "Project review"] };
  }

  generateLearningPath(slug: TrainerSlug, goals: string[]) {
    return { trainer: slug, path: ["Diagnostic assessment", "Beginner foundations", "Guided projects", "Intermediate patterns", "Advanced capstone", "Final assessment"], goals };
  }
}
