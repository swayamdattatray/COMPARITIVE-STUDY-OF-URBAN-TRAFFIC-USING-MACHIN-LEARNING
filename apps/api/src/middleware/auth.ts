import type { NextFunction, Request, Response } from "express";
import jwt from "jsonwebtoken";
import { env } from "../config/env.js";
import type { AuthUser, Role } from "../types/domain.js";

declare global {
  namespace Express { interface Request { user?: AuthUser } }
}

export function requireAuth(roles: Role[] = []) {
  return (req: Request, res: Response, next: NextFunction) => {
    const token = req.headers.authorization?.replace("Bearer ", "");
    if (!token) return res.status(401).json({ message: "Missing bearer token" });
    try {
      const user = jwt.verify(token, env.JWT_SECRET) as AuthUser;
      if (roles.length && !roles.includes(user.role)) return res.status(403).json({ message: "Insufficient role" });
      req.user = user;
      return next();
    } catch {
      return res.status(401).json({ message: "Invalid or expired token" });
    }
  };
}
