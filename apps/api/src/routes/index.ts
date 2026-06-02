import { Router } from "express";
import jwt from "jsonwebtoken";
import { env } from "../config/env.js";
import { requireAuth } from "../middleware/auth.js";
import { AssessmentController } from "../controllers/assessment.controller.js";
import { CertificateController } from "../controllers/certificate.controller.js";
import { TrainerController } from "../controllers/trainer.controller.js";

export const router = Router();

router.get("/health", (_req, res) => res.json({ ok: true, service: "codementor-ai-api" }));
router.post("/auth/demo-login", (req, res) => {
  const user = { id: "demo-user", email: req.body.email ?? "student@codementor.ai", role: req.body.role ?? "STUDENT" };
  res.json({ user, token: jwt.sign(user, env.JWT_SECRET, { expiresIn: "7d" }) });
});

router.get("/trainers", TrainerController.list);
router.post("/trainers/:slug/chat", requireAuth(["STUDENT", "TRAINER", "ADMIN"]), TrainerController.chat);
router.post("/trainers/:slug/learning-path", requireAuth(["STUDENT", "TRAINER", "ADMIN"]), TrainerController.path);

router.get("/tests/milestone", requireAuth(["STUDENT", "TRAINER", "ADMIN"]), AssessmentController.milestone);
router.post("/tests/weekly-saturday", requireAuth(["TRAINER", "ADMIN"]), AssessmentController.weekly);
router.post("/assessments/analyze", requireAuth(["STUDENT", "TRAINER", "ADMIN"]), AssessmentController.analyze);
router.post("/certificates", requireAuth(["ADMIN"]), CertificateController.generate);

router.get("/admin/analytics", requireAuth(["ADMIN"]), (_req, res) => res.json({ students: 12840, activeCourses: 10, weeklyTests: 43, certificatePassRate: 91 }));
