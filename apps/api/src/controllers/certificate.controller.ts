import type { Request, Response } from "express";
import { CertificateService } from "../services/certificate.service.js";

const service = new CertificateService();

export const CertificateController = {
  generate: async (req: Request, res: Response) => res.json(await service.generate(req.body))
};
