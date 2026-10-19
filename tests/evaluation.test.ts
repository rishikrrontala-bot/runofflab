import {describe, it, expect} from 'vitest';
import {capturedImprovement, clusterBootstrapMeans, familyDraws, isTruncated, marginalSummary, meanOrNull, pairedComparison, percentileCI, siteScenarioFingerprint, timedProposal,
  type CaseOutcome, type MethodRow} from '../scripts/evaluation-metrics';
import {preparedSite, candidates, targets} from '../src/core/site';
import {defaultConfig} from '../src/core/types';

const cases: CaseOutcome[] = ['a', 'b', 'c'].flatMap(family => [0, 1].map(variation => ({family, variation, source: 'synthetic', noneM3: 10, referenceM3: 8})));
const row = (c: CaseOutcome, objective: number | null, valid = objective !== null): MethodRow =>
  ({family: c.family, variation: c.variation, source: c.source, status: 'completed', valid, firstPassValid: valid, objective});

describe('evaluation metrics (Codex EVAL-CX items)', () => {
  it('scores an invalid answer as no gardens: zero capture when there is room (EVAL-CX-001)', () => {
    expect(capturedImprovement(10, 8, null)).toBe(0);
    expect(capturedImprovement(10, 8, 9)).toBeCloseTo(.5);
    expect(capturedImprovement(10, 10, 9)).toBeNull();   // no room: undefined, not 0
    expect(capturedImprovement(10, 8, 7)).toBeCloseTo(1.5);   // can beat the bounded-search reference
  });

  it('reports empty groups as null, not 0', () => {
    expect(meanOrNull([])).toBeNull();
    const s = marginalSummary(cases, cases.map(c => row(c, 9)), ['a', 'b', 'c'], familyDraws(3, 50));
    expect((s.bySource.real as {meanCapturedImprovement: number | null}).meanCapturedImprovement).toBeNull();
    expect(s.meanCapturedImprovement).toBeCloseTo(.5);
  });

  it('excludes infrastructure failures from validity and capture but reports coverage', () => {
    const rows = cases.map((c, i) => i === 0 ? {...row(c, null), status: 'infra-failure' as const, valid: null, firstPassValid: null} : row(c, 9));
    const s = marginalSummary(cases, rows, ['a', 'b', 'c'], familyDraws(3, 50));
    expect(s.infraFailures).toBe(1); expect(s.coverage).toBeCloseTo(5 / 6); expect(s.validity).toBe(1); expect(s.meanCapturedImprovement).toBeCloseTo(.5);
  });

  it('paired comparison recovers a known constant difference with a zero-width interval (EVAL-CX-005)', () => {
    const draws = familyDraws(3, 200);
    const p = pairedComparison(cases, cases.map(c => row(c, 9)), cases.map(c => row(c, 9.5)), ['a', 'b', 'c'], draws);
    expect(p.pairedCases).toBe(6); expect(p.families).toBe(3);
    expect(p.meanCapturedDifference).toBeCloseTo(.25);
    expect(p.capturedDifferenceCI95![0]).toBeCloseTo(.25); expect(p.capturedDifferenceCI95![1]).toBeCloseTo(.25);
    expect(p.comparisonStatus).toBe('complete'); expect(p.modelBetter).toBe(true); expect(p.modelWorse).toBe(false);
    expect(p.meanObjectiveDifferenceM3).toBeCloseTo(-.5);
  });

  it('counts coverage against expected cases and rejects duplicate or mismatched rows (EVAL-CX-006)', () => {
    const draws = familyDraws(3, 50), rows = cases.slice(1).map(c => row(c, 9));   // one expected case has no row
    const s = marginalSummary(cases, rows, ['a', 'b', 'c'], draws);
    expect(s.cases).toBe(6); expect(s.infraFailures).toBe(1); expect(s.coverage).toBeCloseTo(5 / 6);
    expect(() => marginalSummary(cases, [...cases.map(c => row(c, 9)), row(cases[0], 7)], ['a', 'b', 'c'], draws)).toThrow(/Duplicate row/);
    expect(() => marginalSummary(cases, cases.map((c, i) => i ? row(c, 9) : {...row(c, 9), source: 'real'}), ['a', 'b', 'c'], draws)).toThrow(/Source mismatch/);
  });

  it('marks a comparison with missing pairs as incomplete and makes no directional claim (EVAL-CX-007)', () => {
    const p = pairedComparison(cases, cases.slice(1).map(c => row(c, 9)), cases.map(c => row(c, 9.5)), ['a', 'b', 'c'], familyDraws(3, 100));
    expect(p.comparisonStatus).toBe('incomplete'); expect(p.incompletePairs).toBe(1);
    expect(p.modelBetter).toBeNull(); expect(p.modelWorse).toBeNull();
  });

  it('family bootstrap resamples whole families with the same draws for every method', () => {
    const draws = familyDraws(2, 2000, 7);
    expect(familyDraws(2, 2000, 7)).toEqual(draws);
    const means = clusterBootstrapMeans([[0, 0, 0], [1]], draws);   // a family with 3 cases at 0, one with 1 case at 1
    const ci = percentileCI(means)!;
    expect(ci[0]).toBe(0); expect(ci[1]).toBe(1);   // draws of only one family give 0 or 1: whole families, not cases
  });

  it('timing closes after scoring finishes (EVAL-CX-003 order bug)', async () => {
    const t = await timedProposal(() => 'design', () => { const s = performance.now(); while (performance.now() - s < 15) { /* scoring */ } return 1; });
    expect(t.scoringMs).toBeGreaterThanOrEqual(14);
    expect(t.endToEndMs).toBeGreaterThanOrEqual(t.selectionMs + t.scoringMs - 1);
  });

  it('detects a truncated local answer from the server stop reason (EVAL-CX-009)', () => {
    const cut = {raw: '{"selected_patch_ids":["G1",', outputTokens: 96};
    expect(isTruncated({...cut, finishReason: 'length'}, 96)).toBe(true);   // non-empty but cut off: only the stop reason shows it
    expect(isTruncated({raw: '{"selected_patch_ids":["G1"]}', outputTokens: 12, finishReason: 'stop'}, 96)).toBe(false);
    expect(isTruncated({raw: '', outputTokens: 96}, 96)).toBe(true);   // hosted reasoning model that spent the cap thinking
    expect(isTruncated({raw: '', outputTokens: 96}, null)).toBe(false);   // unknown cap and no stop reason: not claimed
  });

  it('site fingerprint changes when a candidate, the configuration or the prompt changes (EVAL-CX-004)', () => {
    const g = preparedSite(), config = {...defaultConfig(), roi: targets[0]}, pool = candidates(g, targets[0]);
    const base = {grid: g, config, candidates: pool, prompt: 'p', solverVersion: 'li-1.0.0', promptVersion: 'v', searchPolicy: 's'};
    const f = siteScenarioFingerprint(base);
    expect(siteScenarioFingerprint({...base})).toBe(f);
    expect(siteScenarioFingerprint({...base, candidates: pool.map((p, i) => i === 0 ? {...p, x: p.x + 1} : p)})).not.toBe(f);
    expect(siteScenarioFingerprint({...base, config: {...config, rainMmH: 61}})).not.toBe(f);
    expect(siteScenarioFingerprint({...base, prompt: 'q'})).not.toBe(f);
  });
});
