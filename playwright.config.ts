import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'tests/browser',
  workers: 1,
  timeout: 180_000,
  expect: { timeout: 60_000 },
  reporter: [['list'], ['json', { outputFile: 'evidence/browser-results.json' }]],
  use: {
    baseURL: 'http://127.0.0.1:5173',
    viewport: { width: 1440, height: 1000 },
    headless: true,
    launchOptions: {
      executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      args: ['--mute-audio'],
    },
    trace: 'retain-on-failure',
  },
});
