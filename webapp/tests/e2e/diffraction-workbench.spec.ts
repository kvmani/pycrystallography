import { expect, test } from '@playwright/test';
import type { Locator, Page, TestInfo } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import type {
  PowderPatternResponse,
  StructureModel,
  StructureSummary,
  TemPatternResponse,
  UiConfig
} from '../../src/types/structure';

interface PhaseFixture {
  phase: string;
  structure: StructureModel;
  summary: StructureSummary & { name?: string | null };
  xrd: PowderPatternResponse;
  tem: Record<string, TemPatternResponse>;
  default_tem_axis: string;
  ui_config: UiConfig;
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FIXTURES_DIR = path.join(__dirname, '..', 'fixtures');
const ARTIFACTS_DIR = path.join(__dirname, '..', '..', 'artfacts');
const REPORT_FILE = path.join(ARTIFACTS_DIR, 'screenshot-report.md');

function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function loadFixture(fileName: string): PhaseFixture {
  const fixturePath = path.join(FIXTURES_DIR, fileName);
  return JSON.parse(fs.readFileSync(fixturePath, 'utf-8')) as PhaseFixture;
}

async function setupPhaseRoutes(page: Page, fixture: PhaseFixture) {
  await page.route('**/ui/config', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(fixture.ui_config)
    });
  });

  await page.route('**/structures/from-cif', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ structure: fixture.structure, summary: fixture.summary })
    });
  });

  await page.route('**/diffraction/generate', async (route) => {
    const request = route.request().postDataJSON?.() as { tem?: { zone_axis: number[] } } | undefined;
    const axis = request?.tem?.zone_axis ?? fixture.default_tem_axis.split(' ').map(Number);
    const key = Array.isArray(axis) ? axis.join(' ') : fixture.default_tem_axis;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        structure: fixture.structure,
        summary: fixture.summary,
        xrd: fixture.xrd,
        tem: fixture.tem[key] ?? fixture.tem[fixture.default_tem_axis]
      })
    });
  });

  await page.route('**/diffraction/xrd', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(fixture.xrd)
    });
  });

  await page.route('**/diffraction/tem', async (route) => {
    const request = route.request().postDataJSON?.() as { settings?: { zone_axis: number[] } } | undefined;
    const axis = request?.settings?.zone_axis ?? fixture.default_tem_axis.split(' ').map(Number);
    const key = Array.isArray(axis) ? axis.join(' ') : fixture.default_tem_axis;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(fixture.tem[key] ?? fixture.tem[fixture.default_tem_axis])
    });
  });
}

async function captureSnapshot(
  locator: Locator,
  fixture: PhaseFixture,
  testTitle: string,
  name: string,
  description: string,
  testInfo: TestInfo
) {
  const phaseSlug = slugify(fixture.phase);
  const snapshotSlug = slugify(`${testTitle}-${name}`);
  const snapshotDir = path.join(ARTIFACTS_DIR, phaseSlug);
  fs.mkdirSync(snapshotDir, { recursive: true });
  const snapshotPath = path.join(snapshotDir, `${snapshotSlug}.png`);
  await locator.waitFor();
  await locator.screenshot({ path: snapshotPath });

  const relativePath = path.relative(ARTIFACTS_DIR, snapshotPath).replace(/\\/g, '/');
  fs.appendFileSync(REPORT_FILE, `- **${description}**: ![${snapshotSlug}](${relativePath})\n`);
  await testInfo.attach(`${name}.png`, { path: snapshotPath, contentType: 'image/png' });
}

test.describe.configure({ mode: 'serial' });

const feFixture = loadFixture('fe-alpha.json');
const zrFixture = loadFixture('zr-alpha.json');

async function uploadCif(page: Page, fixture: PhaseFixture, fileName: string) {
  const fileChooserPromise = page.waitForEvent('filechooser');
  await page.getByText('Drag & drop a CIF file here, or click to browse').click();
  const fileChooser = await fileChooserPromise;
  const filePath = path.resolve(__dirname, '..', '..', '..', 'data', 'structureData', fileName);
  await fileChooser.setFiles(filePath);
  await expect(page.getByLabel('Phase name')).toHaveValue(fixture.structure.name ?? fixture.phase);
}

async function rotateUnitCell(page: Page) {
  const canvas = page.locator('[data-testid="unit-cell-viewer"] canvas').first();
  const box = await canvas.boundingBox();
  if (!box) return;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 160, box.y + box.height / 2 + 60, { steps: 20 });
  await page.mouse.up();
  await page.waitForTimeout(400);
}

async function styleUnitCell(page: Page, fixture: PhaseFixture) {
  await page.getByLabel('Atom scale').fill('1.25');
  await page.keyboard.press('Enter');
  await page.waitForTimeout(200);
  await page.getByTestId('viewer-background-hex').fill('#0b1120');
  await page.waitForTimeout(200);
  await page
    .getByTestId(`element-color-input-${fixture.structure.atom_sites[0].element.toLowerCase()}`)
    .fill('#22d3ee');
  await page.waitForTimeout(300);
}

async function generatePatterns(page: Page) {
  await page.getByRole('button', { name: 'Generate all' }).click();
  await expect(page.getByText('Formula')).toBeVisible();
  await page.waitForTimeout(800);
}

async function setZoneAxis(page: Page, axis: [number, number, number]) {
  const [h, k, l] = axis;
  await page.getByLabel('Zone axis h').fill(h.toString());
  await page.getByLabel('Zone axis k').fill(k.toString());
  await page.getByLabel('Zone axis l').fill(l.toString());
  await page.getByRole('button', { name: 'Regenerate TEM' }).click();
  await page.waitForTimeout(700);
}

test('Fe alpha workflow snapshots', async ({ page }, testInfo) => {
  await setupPhaseRoutes(page, feFixture);
  await page.goto('/');

  await uploadCif(page, feFixture, 'Fe.cif');

  const viewer = page.locator('[data-testid="unit-cell-viewer"]');
  await captureSnapshot(viewer, feFixture, testInfo.title, 'unit-cell-default', 'Fe α unit cell — default view', testInfo);

  await rotateUnitCell(page);
  await captureSnapshot(viewer, feFixture, testInfo.title, 'unit-cell-rotated', 'Fe α unit cell — rotated perspective', testInfo);

  await styleUnitCell(page, feFixture);
  await captureSnapshot(viewer, feFixture, testInfo.title, 'unit-cell-styled', 'Fe α unit cell — styled atoms and background', testInfo);

  await generatePatterns(page);

  const xrd = page.locator('[data-testid="xrd-pattern"] .js-plotly-plot').first();
  await captureSnapshot(xrd, feFixture, testInfo.title, 'xrd-pattern', 'Fe α powder XRD pattern', testInfo);

  const tem = page.locator('[data-testid="tem-pattern"] .js-plotly-plot').first();
  await captureSnapshot(tem, feFixture, testInfo.title, 'tem-pattern-001', 'Fe α TEM SAED [0 0 1]', testInfo);

  await setZoneAxis(page, [1, 1, 0]);
  await captureSnapshot(tem, feFixture, testInfo.title, 'tem-pattern-110', 'Fe α TEM SAED [1 1 0]', testInfo);
});

test('Zr alpha workflow snapshots', async ({ page }, testInfo) => {
  await setupPhaseRoutes(page, zrFixture);
  await page.goto('/');

  await uploadCif(page, zrFixture, 'Zr-Alpha.cif');

  const viewer = page.locator('[data-testid="unit-cell-viewer"]');
  await captureSnapshot(viewer, zrFixture, testInfo.title, 'unit-cell-default', 'Zr α unit cell — default view', testInfo);

  await rotateUnitCell(page);
  await captureSnapshot(viewer, zrFixture, testInfo.title, 'unit-cell-rotated', 'Zr α unit cell — rotated perspective', testInfo);

  await styleUnitCell(page, zrFixture);
  await captureSnapshot(viewer, zrFixture, testInfo.title, 'unit-cell-styled', 'Zr α unit cell — styled atoms and background', testInfo);

  await generatePatterns(page);

  const xrd = page.locator('[data-testid="xrd-pattern"] .js-plotly-plot').first();
  await captureSnapshot(xrd, zrFixture, testInfo.title, 'xrd-pattern', 'Zr α powder XRD pattern', testInfo);

  const tem = page.locator('[data-testid="tem-pattern"] .js-plotly-plot').first();
  await captureSnapshot(tem, zrFixture, testInfo.title, 'tem-pattern-001', 'Zr α TEM SAED [0 0 1]', testInfo);

  await setZoneAxis(page, [1, 0, 0]);
  await captureSnapshot(tem, zrFixture, testInfo.title, 'tem-pattern-100', 'Zr α TEM SAED [1 0 0]', testInfo);
});
