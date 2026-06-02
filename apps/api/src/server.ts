import { env } from "./config/env.js";
import { createApp } from "./app.js";

createApp().listen(env.PORT, () => {
  console.log(`CodeMentor AI API listening on ${env.PORT}`);
});
