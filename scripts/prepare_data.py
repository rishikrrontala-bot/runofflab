"""Reproduce a small NJ DEM crop. Python 3 + numpy + Pillow; network uses system curl TLS.
No private coordinates: this is a public park. Run --fetch once, then offline by default.
"""
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
OUT = ROOT / 'public/data'
DEM = 'https://maps.nj.gov/arcgis/rest/services/Elevation/NJ_10ft_DEM/ImageServer'
IMAGERY = 'https://maps.nj.gov/arcgis/rest/services/Basemap/Orthos_Natural_2015_NJ/MapServer'
CX, CY = 529972.6749304753, 4457247.4057999011
EXTENT = [CX-256, CY-256, CX+256, CY+256]
FT_US_M = 1200 / 3937

def fetch(url, target):
    subprocess.run(['curl','-fsSL','--retry','2','--max-time','90',url,'-o',str(target)],check=True)

def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def convert(values, expected_shape):
    a = np.asarray(values, dtype=np.float64)
    if a.shape != expected_shape:
        raise ValueError(f'Expected {expected_shape}; got {a.shape}')
    valid = np.isfinite(a) & (a > -1000) & (a < 15000)
    if not valid.any():
        raise ValueError('Crop has no valid elevations')
    return np.where(valid, a * FT_US_M, 0), valid

def water_bodies(z, valid, tolerance_m=1e-3, min_cells=20):
    """Hydro-flattened water: 4-connected runs of valid cells whose neighbours share the same
    elevation (within 1 mm) and that cover at least `min_cells` cells. Returns a 0/1 sink mask."""
    ny, nx = z.shape
    flat = np.zeros_like(valid)
    same_x = (np.abs(z[:, 1:] - z[:, :-1]) < tolerance_m) & valid[:, 1:] & valid[:, :-1]
    same_y = (np.abs(z[1:, :] - z[:-1, :]) < tolerance_m) & valid[1:, :] & valid[:-1, :]
    flat[:, 1:] |= same_x; flat[:, :-1] |= same_x; flat[1:, :] |= same_y; flat[:-1, :] |= same_y
    sink = np.zeros((ny, nx), dtype=int); seen = np.zeros((ny, nx), dtype=bool)
    for y in range(ny):
        for x in range(nx):
            if not flat[y, x] or seen[y, x]:
                continue
            stack, component = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop(); component.append((cy, cx))
                for yy, xx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= yy < ny and 0 <= xx < nx and flat[yy, xx] and not seen[yy, xx] and abs(z[yy, xx] - z[cy, cx]) < tolerance_m:
                        seen[yy, xx] = True; stack.append((yy, xx))
            if len(component) >= min_cells:
                for cy, cx in component:
                    sink[cy, cx] = 1
    return sink

def surface_classes(image, n, pixels_per_cell=8):
    """Per-cell colour classification of the April 2015 aerial (leaf-off: lawns are green, bare trees brown).
    paved = >= 50% bright low-chroma pixels, restricted to the largest connected network (roads + parking),
    with single-cell gaps (parked cars) filled; lawn = >= 50% green pixels and not paved; else other.
    Reference classification only: not surveyed ground truth. Returns (classes, gray fraction, green fraction)."""
    a = np.asarray(image.convert('RGB'), dtype=np.float64)
    gray = ((a.max(2) - a.min(2) < 28) & (a.max(2) > 150)).reshape(n, pixels_per_cell, n, pixels_per_cell).mean((1, 3))
    green = ((a[:, :, 1] > a[:, :, 0] + 8) & (a[:, :, 1] > a[:, :, 2] + 8)).reshape(n, pixels_per_cell, n, pixels_per_cell).mean((1, 3))
    paved = gray >= 0.5
    label = np.zeros((n, n), dtype=int); sizes = {}
    for y in range(n):
        for x in range(n):
            if paved[y, x] and not label[y, x]:
                k = len(sizes) + 1; label[y, x] = k; stack = [(y, x)]; sizes[k] = 0
                while stack:
                    cy, cx = stack.pop(); sizes[k] += 1
                    for yy in range(cy - 1, cy + 2):
                        for xx in range(cx - 1, cx + 2):
                            if 0 <= yy < n and 0 <= xx < n and paved[yy, xx] and not label[yy, xx]:
                                label[yy, xx] = k; stack.append((yy, xx))
    network = label == max(sizes, key=sizes.get) if sizes else np.zeros((n, n), dtype=bool)
    filled = network.copy()
    for y in range(1, n - 1):
        for x in range(1, n - 1):
            if not network[y, x] and network[y - 1, x] + network[y + 1, x] + network[y, x - 1] + network[y, x + 1] >= 3:
                filled[y, x] = True
    classes = np.where(filled, 1, np.where(green >= 0.5, 2, 0))
    return classes, gray, green

def prepare(fetch_data=False, fetch_fine=False):
    RAW.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    common = {'bbox':','.join(map(str,EXTENT)), 'bboxSR':32618,'imageSR':32618,'f':'image'}
    dem_url = DEM+'/exportImage?'+urlencode({**common,'size':'128,128','format':'tiff','pixelType':'F32','interpolation':'RSP_BilinearInterpolation','noData':-9999})
    image_url = IMAGERY+'/export?'+urlencode({**common,'size':'1024,1024','format':'jpg','transparent':'false'})
    if fetch_data:
        fetch(dem_url,RAW/'park-dem-512m.tiff')
        fetch(image_url,RAW/'park-2015-512m.jpg')
        fetch(DEM+'?f=pjson',RAW/'nj-dem-service.json')
        fetch('https://newjersey.maps.arcgis.com/sharing/rest/content/items/c29ab646471d440f83843b704e811207?f=pjson',RAW/'nj-dem-item.json')
        fetch(IMAGERY+'?f=pjson',RAW/'nj-imagery-service.json')
    z, valid = convert(Image.open(RAW/'park-dem-512m.tiff'), (128,128))
    # No pit filling, smoothing, hydrologic conditioning, building edits or invented terrain.
    for name, lo, hi in [('park',32,96),('park-expanded',0,128)]:
        a = z[lo:hi,lo:hi]; m = valid[lo:hi,lo:hi]; n = hi-lo
        extent = [EXTENT[0]+lo*4,EXTENT[3]-hi*4,EXTENT[0]+hi*4,EXTENT[3]-lo*4]
        payload = {'version':1,'id':name+'-v1','name':'Mercer County Park · recreation grounds',
                   'nx':n,'ny':n,'dx':4,'dy':4,'crs':'EPSG:32618','bounds':extent,
                   'sourceYears':'2014–2019 LiDAR composite','catalogEdition':'2021-01-04',
                   'sourceResolutionM':10*FT_US_M,'elevationUnit':'m','verticalDatum':'NAVD88',
                   'z':np.round(a,6).ravel().tolist(),'mask':m.astype(int).ravel().tolist()}
        (OUT/(name+'.json')).write_text(json.dumps(payload,separators=(',',':'))+'\n')
    # Computational domain: the 256 m study site plus a 64 m buffer on every side (384 m, 96 x 96 at 4 m).
    # Domain tests showed the parking-loop target receives runoff from up to 32 m beyond the west edge of the
    # 256 m crop, and that a 320 m domain already matches the 512 m result; 384 m keeps a 2x margin.
    lo, hi = 16, 112
    a = z[lo:hi, lo:hi]; m = valid[lo:hi, lo:hi]; n = hi - lo
    sink = water_bodies(z, valid)[lo:hi, lo:hi]
    extent = [EXTENT[0]+lo*4, EXTENT[3]-hi*4, EXTENT[0]+hi*4, EXTENT[3]-lo*4]
    domain = {'version': 1, 'id': 'park-domain-v1', 'name': 'Mercer County Park · west picnic loop and surroundings',
              'nx': n, 'ny': n, 'dx': 4, 'dy': 4, 'crs': 'EPSG:32618', 'bounds': extent,
              'sourceYears': '2014–2019 LiDAR composite', 'catalogEdition': '2021-01-04',
              'sourceResolutionM': 10*FT_US_M, 'elevationUnit': 'm', 'verticalDatum': 'NAVD88',
              'site': {'x': 32 - lo, 'y': 32 - lo, 'width': 64, 'height': 64, 'name': '256 m study site'},
              'z': np.round(a, 6).ravel().tolist(), 'mask': m.astype(int).ravel().tolist(), 'sink': sink.ravel().tolist(),
              'sinkRule': 'hydro-flattened water: 4-connected cells with equal elevation (1 mm) covering >= 20 cells'}
    image = Image.open(RAW/'park-2015-512m.jpg')   # 1024 px for 512 m: 2 px per metre
    domain_image = image.crop((lo*8, lo*8, hi*8, hi*8))
    classes, gray, green = surface_classes(domain_image, n)
    domain['surface'] = classes.ravel().tolist()
    domain['surfaceClasses'] = {'0': 'other (woods, bare ground, structures)', '1': 'paved network', '2': 'lawn'}
    domain['surfaceRule'] = surface_classes.__doc__.split('Returns')[0].strip()
    (OUT/'park-domain.json').write_text(json.dumps(domain, separators=(',', ':'))+'\n')
    image.crop((256,256,768,768)).save(OUT/'park-aerial.jpg',quality=90)
    domain_image.save(OUT/'park-domain-aerial.jpg', quality=90)
    # Evidence-only grids (not shipped to the browser): the full 512 m crop with its water bodies for
    # domain-size tests, and a 2 m export of the 384 m domain for spatial refinement.
    DERIVED = ROOT/'data/derived'; DERIVED.mkdir(exist_ok=True)
    full_sink = water_bodies(z, valid)
    (DERIVED/'park-512m-4m.json').write_text(json.dumps({'id': 'park-512m-4m', 'nx': 128, 'ny': 128, 'dx': 4, 'dy': 4, 'offsetToDomain': lo,
        'z': np.round(z, 6).ravel().tolist(), 'mask': valid.astype(int).ravel().tolist(), 'sink': full_sink.ravel().tolist()}, separators=(',', ':'))+'\n')
    fine = RAW/'park-dem-384m-2m.tiff'
    fine_url = DEM+'/exportImage?'+urlencode({**common, 'bbox': ','.join(map(str, [EXTENT[0]+lo*4, EXTENT[3]-hi*4, EXTENT[0]+hi*4, EXTENT[3]-lo*4])),
                                             'size': '192,192', 'format': 'tiff', 'pixelType': 'F32', 'interpolation': 'RSP_BilinearInterpolation', 'noData': -9999})
    if fetch_data or (fetch_fine and not fine.exists()):
        fetch(fine_url, fine)
    if fine.exists():
        z2, v2 = convert(Image.open(fine), (192, 192))
        (DERIVED/'park-domain-2m.json').write_text(json.dumps({'id': 'park-domain-2m', 'nx': 192, 'ny': 192, 'dx': 2, 'dy': 2, 'url': fine_url,
            'z': np.round(z2, 6).ravel().tolist(), 'mask': v2.astype(int).ravel().tolist(), 'sink': water_bodies(z2, v2, min_cells=80).ravel().tolist()}, separators=(',', ':'))+'\n')
    manifest = {'version':1,'retrievedAt':'2026-10-16','site':'Mercer County Park public recreation grounds',
        'centerWGS84Approx':[-74.6475,40.26515],'projectedCRS':'EPSG:32618','boundsM':EXTENT,
        'projectionService':'https://utility.arcgisonline.com/ArcGIS/rest/services/Geometry/GeometryServer/project',
        'terrain':{'url':dem_url,'service':DEM,'item':'c29ab646471d440f83843b704e811207',
        'sourceHorizontalCRS':'EPSG:6527 NAD83(2011) / New Jersey (ftUS)',
        'sourceVerticalCRS':'EPSG:6360 NAVD88 height (ftUS)','conversionMPerFtUS':FT_US_M,
        'collection':'statewide composite of 2014–2019 collections; Mercer regional collection DVRPC 2015',
        'acquisitionDates':'Exact per-pixel collection date not returned by composite service',
        'nativePixelSizeFtUS':10,'outputPixelSizeM':4,'resampling':'ArcGIS bilinear to UTM 18N; no vertical transformation; explicit elevation scaling',
        'nodata':'nonfinite / outside [-1000,15000] ft -> mask=0, z=0; masked faces closed',
        'validCells':int(valid.sum()),'conditioning':'none'},
        'imagery':{'url':image_url,'service':IMAGERY,'year':2015,'use':'reference only; not surface classification ground truth'},
        'license':{'attribution':'NJOGIS / NJDEP; NJ 2015 imagery partners',
        'source':'https://www.nj.gov/njgin/edata/elevation/',
        'terms':'https://www.nj.gov/nj/legal.html','limitations':'Not survey data; supplied as is; attribution requested. Full source terms retained in nj-dem-item.json.'},
        'files':{str(p.relative_to(ROOT)):checksum(p) for p in list(RAW.glob('*'))+list(OUT.glob('*')) if p.is_file()},
        'limitations':['No verified drains, curbs, culverts, roof routing, or field infiltration.',
        'Uniform hypothetical surface parameters; no pavement-level classification claimed.',
        'The 384 m computational domain is chosen from measured domain-size tests, not a surveyed catchment boundary.',
        'Hydro-flattened lake cells are treated as a fixed-level open sink; lake level change is not modelled.',
        'UTM uses WGS84; source NAD83(2011) transformation selected by ArcGIS default, not survey grade.']}
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'shape':z.shape,'elevationM':[float(z[valid].min()),float(z[valid].max())],
                      'valid':int(valid.sum()),'demSha256':checksum(RAW/'park-dem-512m.tiff')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fetch',action='store_true');parser.add_argument('--fetch-fine',action='store_true',help='fetch only the 2 m evidence grid if missing');args=parser.parse_args()
    prepare(args.fetch, args.fetch_fine)
