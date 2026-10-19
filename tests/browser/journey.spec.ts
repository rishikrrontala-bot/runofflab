import { readFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';

const status = (page: Page) => page.locator('.run-status');

/** A short storm keeps each simulation well under a second on the 96 x 96 domain. */
async function shortStorm(page: Page, rain = '3', after = '1') {
  await page.getByLabel('Rain duration · min', { exact: true }).fill(rain);
  await page.getByLabel('After rain · min', { exact: true }).fill(after);
}

async function runBaseline(page: Page) {
  await page.getByRole('button', { name: /^(Run|Rerun) baseline/ }).click();
  await expect(status(page)).toContainText('Complete');
}

function trackErrors(page: Page) {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  return errors;
}

test('baseline, keyboard garden, comparison, scrub, provider failure, scenario round trip, stale and cancel', async ({ page }) => {
  const errors = trackErrors(page);
  let plannerBody: unknown;
  await page.route('**/api/planner', async route => {
    plannerBody = route.request().postDataJSON();
    await route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ error: 'Provider unavailable. Start the configured local model server; greedy planning remains available.' }) });
  });

  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await expect(page.getByText('384 m computed domain · 256 m study site · 4 m cells · 2014–2019 LiDAR composite')).toBeVisible();
  await expect(page.locator('.model-label')).toContainText('li-1.0.0 · local inertial');
  await expect(page.getByRole('group', { name: 'Rain-garden candidates' }).getByRole('button')).toHaveCount(25);

  await page.getByRole('button', { name: 'Flat map', exact: true }).click();
  await shortStorm(page);
  await runBaseline(page);
  await expect(page.locator('.result-feature')).toContainText('Peak ponded water in target');

  const garden = page.getByRole('button', { name: 'Garden G7', exact: true });
  await garden.focus();
  await page.keyboard.press('Enter');
  await expect(garden).toHaveAttribute('aria-pressed', 'true');

  await page.getByRole('button', { name: 'Compare same storm', exact: true }).click();
  await expect(status(page)).toContainText('Complete');
  await expect(page.locator('.map-title').getByText('With gardens', { exact: true })).toBeVisible();
  await expect(page.locator('.map-title').nth(1)).toContainText(/gardens hold [\d.]+ of 38 m³/);
  const accounting = page.locator('table.accounting');
  for (const row of ['Rain input', 'Initial storage', 'Surface water', 'Soil', 'Held in gardens', 'Boundary outflow', 'Into lake', 'Residual']) {
    await expect(accounting.getByRole('rowheader', { name: row, exact: true })).toBeVisible();
  }
  await expect(page.locator('.metrics-table')).toContainText('Held in gardens at end');

  await page.getByLabel('Simulated time').fill('60');
  await expect(page.locator('.playback output')).toContainText('1.0 /');

  await page.getByRole('button', { name: 'Request model proposal', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Provider unavailable');
  await expect(status(page)).toContainText('No other method was substituted');
  expect(plannerBody).toMatchObject({ patches: [], config: { solver: 'li-1.0.0', rainMinutes: 3 } });

  await page.getByText('Save & reproduce', { exact: true }).click();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export scenario JSON', exact: true }).click();
  await (await download).saveAs('evidence/browser-scenario.json');
  const saved = JSON.parse(readFileSync('evidence/browser-scenario.json', 'utf8'));
  expect(saved).toMatchObject({ version: 2, solver: 'li-1.0.0', patches: [{ id: 'G7', pondingDepthM: 0.15 }] });

  await page.getByRole('button', { name: 'Remove all', exact: true }).click();
  await expect(garden).toHaveAttribute('aria-pressed', 'false');
  await page.locator('input[type=file]').setInputFiles('evidence/browser-scenario.json');
  await expect(status(page)).toContainText('restored');
  await expect(garden).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Compare same storm', exact: true }).click();
  await expect(status(page)).toContainText('Complete');
  await expect(page.locator('.map-shell').first()).toHaveAttribute('data-aerial', 'loaded');
  await page.screenshot({ path: '.impeccable/review/desktop.png', fullPage: true });

  await page.getByLabel('Intensity · mm/hour', { exact: true }).fill('70');
  await expect(page.getByText('Inputs changed.', { exact: false })).toBeVisible();
  await expect(page.locator('.map-title').first()).toContainText('Stale settings');
  await page.getByRole('button', { name: 'Rerun baseline', exact: true }).click();
  await page.getByRole('button', { name: 'Cancel run', exact: true }).click();
  await expect(status(page)).toContainText('Canceled');
  expect(errors).toEqual([]);
});

test('model proposals: invalid IDs are reported, a valid one runs under the model name', async ({ page }) => {
  const errors = trackErrors(page);
  let reply: object = { proposal: { selected_patch_ids: ['G99'] }, model: 'test-model', method: 'trained-local' };
  await page.route('**/api/planner', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(reply) }));
  await page.goto('/');
  await page.getByRole('button', { name: 'Flat map', exact: true }).click();
  await shortStorm(page);
  await runBaseline(page);

  await page.getByRole('button', { name: 'Request model proposal', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Model proposal rejected: Unknown candidate: G99');
  await expect(page.locator('.method')).toHaveText('No proposal evaluated');

  reply = { proposal: { selected_patch_ids: ['G7'] }, model: 'test-model', method: 'trained-local' };
  await page.getByRole('button', { name: 'Request model proposal', exact: true }).click();
  await expect(status(page)).toContainText('Complete');
  await expect(page.locator('.method')).toHaveText('Trained local model · test-model');
  await expect(page.getByRole('button', { name: 'Garden G7', exact: true })).toHaveAttribute('aria-pressed', 'true');
  expect(errors).toEqual([]);
});

test('custom target clicks are validated before they apply; target changes drop non-candidates', async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto('/');
  await page.getByRole('button', { name: 'Flat map', exact: true }).click();
  const target = page.getByLabel('Inspect target');
  const map = page.locator('canvas.flat-map').first();
  const clickCell = async (x: number, y: number) => {
    const box = (await map.boundingBox())!;
    await map.click({ position: { x: (x + 0.5) / 96 * box.width, y: (y + 0.5) / 96 * box.height } });
  };

  await page.getByLabel('Flat-map click').selectOption('target');
  await clickCell(68, 42);   // picnic-loop building
  await expect(page.getByRole('alert')).toContainText('masked building');
  await expect(page.getByRole('alert')).toContainText('still South loop target');
  await clickCell(85, 5);    // lake
  await expect(page.getByRole('alert')).toContainText('lake cells');
  await expect(target.locator('option:checked')).toHaveText('South loop target');

  // A valid custom target still lets every run and export work.
  await clickCell(30, 80);
  await expect(page.locator('.notice')).toContainText('Target set to Custom target');
  await expect(target.locator('option:checked')).toHaveText('Custom target');
  await page.getByText('Save & reproduce', { exact: true }).click();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export scenario JSON', exact: true }).click();
  const exported = await download;
  expect(exported.suggestedFilename()).toBe('runofflab-scenario-v2.json');
  expect(JSON.parse(readFileSync(await exported.path(), 'utf8')).config.roi.name).toBe('Custom target');
  await exported.delete();

  await target.selectOption({ label: 'South loop target' });
  await page.getByLabel('Flat-map click').selectOption('garden');
  await page.getByRole('button', { name: 'Garden G2', exact: true }).click();
  await target.selectOption({ label: 'North parking target' });
  await expect(page.locator('.notice')).toContainText('Removed G2');
  await expect(page.getByRole('button', { name: /^Garden G/, pressed: true })).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('mobile flat view, reduced motion and constraint feedback', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Flat map', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('canvas.flat-map')).toBeVisible();
  for (const id of ['G1', 'G2', 'G3', 'G4']) await page.getByRole('button', { name: `Garden ${id}`, exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Maximum patch count');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await expect(page.locator('.map-shell').first()).toHaveAttribute('data-aerial', 'loaded');
  await page.screenshot({ path: '.impeccable/review/mobile.png', fullPage: true });

  await shortStorm(page, '2', '0');
  await runBaseline(page);
  await page.getByRole('button', { name: 'Play history', exact: true }).click();
  await expect(status(page)).toContainText('Reduced motion is on');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test('greedy in the worker with one garden; cancel; WebGL initializes on first view', async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto('/');
  await expect(page.locator('.three-map canvas').first()).toBeVisible();
  await shortStorm(page, '2', '0');
  await page.getByLabel('Maximum gardens', { exact: true }).fill('1');
  await runBaseline(page);
  await expect(page.locator('.results')).toContainText('up to 26 runs');

  await page.getByRole('button', { name: 'Evaluate greedy proposal', exact: true }).click();
  await expect(status(page)).toContainText(/evaluated \d+ of up to 26 designs|Replaying|Complete/);
  await expect(status(page)).toContainText('Complete', { timeout: 120_000 });
  await expect(status(page)).toContainText('greedy evaluated');
  await expect(page.locator('.method')).toContainText('Greedy');
  await expect(page.locator('.method')).toContainText(/\d+ simulation calls/);

  await page.getByRole('button', { name: 'Evaluate greedy proposal', exact: true }).click();
  await page.getByRole('button', { name: 'Cancel run', exact: true }).click();
  await expect(status(page)).toContainText('Canceled');
  await expect(page.getByRole('button', { name: 'Evaluate greedy proposal', exact: true })).toBeEnabled();
  expect(errors).toEqual([]);
});

test('saved model + downhill: two live runs, the downhill design is kept and labelled, export carries both', async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto('/');
  await page.getByRole('button', { name: 'Flat map', exact: true }).click();
  await runBaseline(page);   // default South-loop scenario: one of the recorded saved answers
  await page.getByRole('button', { name: 'Compare saved model + downhill', exact: true }).click();
  await expect(status(page)).toContainText('Complete · downhill rule design kept · 2 live simulations', { timeout: 60_000 });
  const table = page.getByRole('table', { name: 'Saved model and downhill comparison' });
  await expect(table.getByRole('row', { name: /Trained model \(saved answer\)/ })).toContainText('G7, G8, G12');
  await expect(table.getByRole('row', { name: /Downhill rule \(computed now\)/ })).toContainText('G7, G11, G12');
  await expect(page.getByText(/Saved answer \(not live\) from Trained model \(Adaption LoRA, run locally\) · Qwen\/Qwen3\.5-4B · checkpoint merged-v2-qwen35-4b-e6/)).toBeVisible();
  await expect(page.getByText(/Result: downhill rule design kept · 2 live simulations/)).toBeVisible();
  // The design now in the run is the downhill rule's, so it is the one shown as pressed.
  for (const id of ['G7', 'G11', 'G12']) await expect(page.getByRole('button', { name: `Garden ${id}`, exact: true })).toHaveAttribute('aria-pressed', 'true');

  await page.getByText('Save & reproduce', { exact: true }).click();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export result summary', exact: true }).click();
  const exported = await download;
  const both = JSON.parse(readFileSync(await exported.path(), 'utf8')).comparison.savedModelAndDownhill;
  expect(both).toMatchObject({winner: 'downhill', tie: false, sameDesign: false, executedSimulations: 2,
    source: {kind: 'saved-output-not-live-inference', method: 'trained-adaption', checkpoint: 'models/checkpoints/merged-v2-qwen35-4b-e6'}});
  expect(both.components.savedModel.objectiveM3).toBeGreaterThan(both.components.downhill.objectiveM3);
  await exported.delete();

  // Changing the storm away from a recorded scenario hides the saved buttons and says why.
  await page.getByLabel('Intensity · mm/hour', { exact: true }).fill('61');
  await expect(page.getByRole('button', { name: 'Compare saved model + downhill', exact: true })).toHaveCount(0);
  await expect(page.getByText(/No saved model answer for these exact settings/)).toBeVisible();
  expect(errors).toEqual([]);
});
