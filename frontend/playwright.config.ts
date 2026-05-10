import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./src/test",
  timeout: 60_000,
  use: {
    baseURL: "http://localhost:8080",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
