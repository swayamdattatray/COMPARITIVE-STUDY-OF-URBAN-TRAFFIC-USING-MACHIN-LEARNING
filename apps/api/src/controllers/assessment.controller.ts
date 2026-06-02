import type { Request, Response } from "express";
import { z } from "zod";
import { AssessmentService } from "../services/assessment.service.js";

const service = new AssessmentService();
const analyzeSchema = z.object({ scores: z.array(z.number()).default([]), weakSignals: z.array(z.string()).default([]) });

export const AssessmentController = {
  milestone: (req: Request, res: Response) => res.json({ unlocked: service.milestoneUnlocks(Number(req.query.progress ?? 0)) }),
  weekly: (req: Request, res: Response) => res.json(service.buildWeeklySaturdayTest(req.body.topics ?? [])),
  analyze: (req: Request, res: Response) => {
    const body = analyzeSchema.parse(req.body);
    res.json(service.analyze(body.scores, body.weakSignals));
  }
};
