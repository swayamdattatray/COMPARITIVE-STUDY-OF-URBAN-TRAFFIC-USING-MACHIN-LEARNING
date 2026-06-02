import assert from "node:assert/strict";
import test from "node:test";
import { AssessmentService } from "./services/assessment.service.js";

test("unlocks milestone tests at every 25 percent", () => {
  const service = new AssessmentService();
  assert.deepEqual(service.milestoneUnlocks(76), [25, 50, 75]);
});

test("returns performance analysis with revision plans", () => {
  const service = new AssessmentService();
  const result = service.analyze([60, 80], ["recursion"]);
  assert.equal(result.score, 70);
  assert.equal(result.weakAreas[0], "recursion");
  assert.match(result.revisionPlan[0], /recursion/);
});
