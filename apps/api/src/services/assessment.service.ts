import type { AssessmentResult } from "../types/domain.js";

export class AssessmentService {
  milestoneUnlocks(progress: number) {
    return [25, 50, 75, 100].filter((milestone) => progress >= milestone);
  }

  buildWeeklySaturdayTest(topics: string[]) {
    return {
      schedule: "Every Saturday",
      coverage: topics,
      sections: ["AI-generated MCQs", "Coding challenge", "Practical assignment"],
      grading: "Auto evaluation with rubric-based performance analysis"
    };
  }

  analyze(scores: number[], weakSignals: string[]): AssessmentResult {
    const score = Math.round(scores.reduce((sum, item) => sum + item, 0) / Math.max(scores.length, 1));
    const weakAreas = weakSignals.length ? weakSignals : score < 70 ? ["Core concepts", "Debugging", "Problem decomposition"] : ["Advanced optimization"];
    return {
      score,
      weakAreas,
      revisionPlan: weakAreas.map((area) => `Revise ${area}, complete 2 examples, then pass a 10-question quiz.`),
      nextTopics: score >= 80 ? ["Advanced project", "Peer code review"] : ["Fundamentals recap", "Guided practice"],
      completionPrediction: score >= 80 ? "On track to complete within 3 weeks" : "Likely completion in 5 weeks with revision plan",
      careerPaths: ["Software Developer", "Backend Engineer", "AI/ML Engineer"]
    };
  }
}
