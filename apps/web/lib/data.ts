import { Bot, BrainCircuit, Code2, Database, Globe, GraduationCap, LockKeyhole, Server, Shield, Workflow } from "lucide-react";

export const trainers = [
  { name: "Python Development", icon: Code2, progress: 72, focus: "APIs, automation, OOP, testing" },
  { name: "C Programming", icon: Workflow, progress: 48, focus: "Pointers, memory, systems fundamentals" },
  { name: "PHP Development", icon: Server, progress: 39, focus: "Laravel-ready backend patterns" },
  { name: "HTML, CSS & JavaScript", icon: Globe, progress: 83, focus: "Responsive UI and DOM mastery" },
  { name: "Full Stack Web Development", icon: GraduationCap, progress: 61, focus: "Frontend, backend, deployments" },
  { name: "Backend Development", icon: Server, progress: 56, focus: "REST, auth, caching, queues" },
  { name: "Machine Learning", icon: BrainCircuit, progress: 45, focus: "Model training and evaluation" },
  { name: "Data Structures & Algorithms", icon: Bot, progress: 68, focus: "Patterns, complexity, interviews" },
  { name: "Cyber Security", icon: Shield, progress: 34, focus: "Secure coding and threat models" },
  { name: "Database Management (SQL/MySQL)", icon: Database, progress: 77, focus: "Schema design, joins, indexing" }
];

export const dashboardStats = [
  { label: "Learning Progress", value: "68%", detail: "+14% this month" },
  { label: "Weekly Hours", value: "18.5", detail: "4.2h above target" },
  { label: "Upcoming Tests", value: "3", detail: "Next Saturday AI test" },
  { label: "Certificates", value: "5", detail: "2 verified by QR" }
];

export const activities = ["Python milestone test unlocked at 75%", "AI trainer recommended SQL indexing revision", "Earned 450 XP from JavaScript challenges", "Submitted backend practical assignment"];

export const courseModules = ["Beginner Level", "Intermediate Level", "Advanced Level", "Video Lessons", "AI Tutor Chat", "Notes & Resources", "Coding Playground", "Practice Assignments", "Progress Tracking"];
