"""Prepare real-terrain training families from public NJ LiDAR crops.

Input: data/raw/families/<id>/dem-512m-4m.tiff (512 m square, 4 m, US survey feet, fetched by
scripts/fetch_families.py from the NJOGIS/NJDEP 10 ft DEM ImageServer) plus the 2015 aerial reference.
Output: models/data/families/<id>.json (64 x 64 cells at 8 m: 2 x 2 block means, metres NAVD88, water-body
sink mask) and data/families-manifest.json (provenance, checksums, split, exclusions).

Families are whole crops. Splits are fixed here, before any case is generated or simulated, by a hash of the
family ID, so a family never contributes to more than one split. The Mercer County Park showcase crop is
excluded from every split. Land-cover notes come from looking at the 2015 aerial and are approximate; crops
are named by coordinates, not by park boundaries (several approximate park coordinates missed the park).
Run: python3 scripts/prepare_families.py
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_data import water_bodies, FT_US_M  # noqa: E402  (same conversion and water rule as the site)

RAW = ROOT / 'data/raw/families'
OUT = ROOT / 'models/data/families'
EXCLUDED = {
    'cooper-river-park': 'two thirds of the inner area is a hydro-flattened river',
    'johnson-park-piscataway': 'more than 5% hydro-flattened water inside the core area',
    'parvin-sp': 'more than 5% hydro-flattened water inside the core area',
    'mercer-county-park': 'showcase site; never used for training or tuning',
}
LAND_COVER = {
    'branch-brook-park': 'urban edge, parking and lawn', 'cape-may-county-park': 'highway, buildings and lawn',
    'central-park-morris': 'woods with clearings', 'colonial-park-somerset': 'fields and woods',
    'donaldson-park-highland-park': 'residential and park lawn', 'duke-island-park': 'suburban edge and woods',
    'estell-manor-park': 'pine forest', 'garret-mountain': 'rail yard and commercial lots',
    'holmdel-park': 'fields and woods', 'liberty-state-park': 'woods and open ground',
    'lord-stirling-park': 'flat woods and wetland', 'monmouth-battlefield-sp': 'residential streets',
    'ocean-county-park': 'pine forest, lake corner outside core', 'overpeck-park': 'residential and commercial',
    'roosevelt-park-edison': 'residential and institutional', 'rosedale-park': 'open field and woods',
    'thompson-park-lincroft': 'suburban and woods', 'turkey-swamp-park': 'woods and sand',
    'van-saun-park': 'residential streets', 'verona-park': 'residential streets',
    'veterans-park-hamilton': 'open field and woods', 'warinanco-park': 'residential streets',
    'washington-crossing-sp': 'residential and woods',
}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.name for p in RAW.iterdir() if p.is_dir() and p.name not in EXCLUDED)
    ranked = sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    splits = {fid: ('validation' if k < 4 else 'heldout' if k < 8 else 'train') for k, fid in enumerate(ranked)}
    manifest = {'version': 1, 'source': 'NJOGIS/NJDEP NJ_10ft_DEM ImageServer (2014-2019 LiDAR composite); NJ 2015 orthoimagery',
                'license': 'NJ public data; attribution NJOGIS / NJDEP; not survey data', 'resolutionM': 8, 'extentM': 512,
                'processing': '4 m bilinear export -> metres (x 1200/3937) -> 2 x 2 block mean to 8 m; water-body sink: same rule as site at 4 m, a block is water if any of its 4 cells is',
                'splitRule': 'rank by sha256(id): first 4 validation, next 4 heldout, rest train; fixed before generation',
                'excluded': EXCLUDED, 'families': []}
    for fid in ids:
        d = RAW / fid
        meta = json.loads((d / 'meta.json').read_text())
        a = np.asarray(Image.open(d / 'dem-512m-4m.tiff'), dtype=np.float64)
        valid = np.isfinite(a) & (a > -1000) & (a < 15000)
        z = np.where(valid, a * FT_US_M, 0.0)
        sink4 = water_bodies(z, valid)
        z8 = z.reshape(64, 2, 64, 2).mean((1, 3))
        mask8 = valid.reshape(64, 2, 64, 2).all((1, 3))
        sink8 = (sink4.reshape(64, 2, 64, 2).max((1, 3)) > 0) & mask8
        grid = {'id': f'nj-{fid}', 'name': f'Real NJ terrain near {meta["latApprox"]:.4f}, {meta["lonApprox"]:.4f}', 'source': 'real',
                'nx': 64, 'ny': 64, 'dx': 8, 'dy': 8, 'crs': 'EPSG:32618', 'bounds': meta['boundsM'],
                'z': np.round(z8, 4).ravel().tolist(), 'mask': mask8.astype(int).ravel().tolist(), 'sink': sink8.astype(int).ravel().tolist()}
        path = OUT / f'{fid}.json'
        path.write_text(json.dumps(grid, separators=(',', ':')) + '\n')
        manifest['families'].append({'id': fid, 'gridId': grid['id'], 'split': splits[fid], 'landCoverFromAerial': LAND_COVER.get(fid, 'not noted'),
            'centerUtm18': meta['utm18'], 'latLonApprox': [meta['latApprox'], meta['lonApprox']], 'demUrl': meta['demUrl'], 'imageryUrl': meta['imageryUrl'],
            'rawSha256': meta['sha256'], 'gridSha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'reliefM': round(float(z8[mask8].max() - z8[mask8].min()), 2), 'waterCells8m': int(sink8.sum())})
    (ROOT / 'data/families-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    counts = {s: sum(f['split'] == s for f in manifest['families']) for s in ('train', 'validation', 'heldout')}
    print(json.dumps({'families': len(ids), **counts}))

if __name__ == '__main__':
    main()
