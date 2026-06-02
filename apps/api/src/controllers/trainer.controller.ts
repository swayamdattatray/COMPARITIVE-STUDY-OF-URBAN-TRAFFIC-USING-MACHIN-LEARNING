import type { Request, Response } from "express";
import { z } from "zod";
import { AiTrainerService } from "../services/ai-trainer.service.js";
import type { TrainerSlug } from "../types/domain.js";

const service = new AiTrainerService();
const chatSchema = z.object({ question: z.string().min(2), progress: z.number().min(0).max(100).default(0) });

export const TrainerController = {
  list: (_req: Request, res: Response) => res.json({ trainers: service.listTrainers() }),
  chat: async (req: Request, res: Response) => {
    const body = chatSchema.parse(req.body);
    res.json(await service.chat(req.params.slug as TrainerSlug, body.question, body.progress));
  },
  path: (req: Request, res: Response) => res.json(service.generateLearningPath(req.params.slug as TrainerSlug, req.body.goals ?? []))
};
