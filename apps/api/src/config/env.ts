import dotenv from "dotenv";
import { z } from "zod";

dotenv.config();

const schema = z.object({
  NODE_ENV: z.enum(["development", "test", "production"]).default("development"),
  PORT: z.coerce.number().default(4000),
  DATABASE_URL: z.string().default("postgresql://postgres:postgres@localhost:5432/codementor_ai"),
  JWT_SECRET: z.string().default("replace-with-a-secure-secret"),
  OPENAI_API_KEY: z.string().optional()
});

export const env = schema.parse(process.env);
