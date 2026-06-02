import cors from "cors";
import express from "express";
import helmet from "helmet";
import { router } from "./routes/index.js";

export function createApp() {
  const app = express();
  app.use(helmet());
  app.use(cors({ origin: true, credentials: true }));
  app.use(express.json({ limit: "1mb" }));
  app.use("/api/v1", router);
  app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => res.status(400).json({ message: err.message }));
  return app;
}
