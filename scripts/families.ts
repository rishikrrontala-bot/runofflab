import {readFileSync} from 'node:fs';
import {syntheticFamilies, type Family, type Split} from '../src/core/trainingCases';
import type {Grid} from '../src/core/types';

/** All training families: real NJ crops (splits fixed in data/families-manifest.json) and synthetic ones. */
export function loadFamilies(): Family[] {
  const manifest = JSON.parse(readFileSync('data/families-manifest.json', 'utf8'));
  const real = manifest.families.map((f: {id: string; split: Split}) => ({
    id: `nj-${f.id}`, source: 'real' as const, split: f.split, seed: 0,
    grid: JSON.parse(readFileSync(`models/data/families/${f.id}.json`, 'utf8')) as Grid,
  }));
  return [...real, ...syntheticFamilies()];
}
