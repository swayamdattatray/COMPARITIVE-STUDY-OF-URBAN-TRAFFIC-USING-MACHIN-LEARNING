export type Role = "STUDENT" | "TRAINER" | "ADMIN";
export type TrainerSlug = "python" | "c" | "php" | "html-css-js" | "full-stack" | "backend" | "machine-learning" | "dsa" | "cyber-security" | "database";

export interface AuthUser {
  id: string;
  email: string;
  role: Role;
}

export interface AssessmentResult {
  score: number;
  weakAreas: string[];
  revisionPlan: string[];
  nextTopics: string[];
  completionPrediction: string;
  careerPaths: string[];
}
