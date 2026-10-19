"""Fetch candidate real-terrain training crops (512 m, 4 m) + 2015 imagery from public NJ services.
Coordinates are approximate park locations from general knowledge; each crop is verified by imagery
before use and labelled by its coordinates, not by an unverified park boundary."""
import json, math, subprocess, sys, hashlib
from pathlib import Path
from urllib.parse import urlencode
OUT = Path(sys.argv[1])
DEM = 'https://maps.nj.gov/arcgis/rest/services/Elevation/NJ_10ft_DEM/ImageServer'
IMG = 'https://maps.nj.gov/arcgis/rest/services/Basemap/Orthos_Natural_2015_NJ/MapServer'
def utm18(lat, lon):
    a=6378137.0; f=1/298.257223563; e2=f*(2-f); ep2=e2/(1-e2); k0=0.9996
    p=math.radians(lat); l=math.radians(lon+75)
    N=a/math.sqrt(1-e2*math.sin(p)**2); T=math.tan(p)**2; C=ep2*math.cos(p)**2; A=math.cos(p)*l
    e4,e6=e2*e2,e2**3
    M=a*((1-e2/4-3*e4/64-5*e6/256)*p-(3*e2/8+3*e4/32+45*e6/1024)*math.sin(2*p)+(15*e4/256+45*e6/1024)*math.sin(4*p)-(35*e6/3072)*math.sin(6*p))
    x=k0*N*(A+(1-T+C)*A**3/6+(5-18*T+T*T+72*C-58*ep2)*A**5/120)+500000
    y=k0*(M+N*math.tan(p)*(A*A/2+(5-T+9*C+4*C*C)*A**4/24+(61-58*T+T*T+600*C-330*ep2)*A**6/720))
    return x,y
SITES = [
 ('johnson-park-piscataway',40.5125,-74.4790),('duke-island-park',40.5700,-74.6740),('colonial-park-somerset',40.5040,-74.5720),
 ('thompson-park-lincroft',40.3370,-74.1170),('holmdel-park',40.3600,-74.1780),('warinanco-park',40.6560,-74.2520),
 ('branch-brook-park',40.7700,-74.1770),('verona-park',40.8270,-74.2420),('liberty-state-park',40.7020,-74.0560),
 ('cooper-river-park',39.9260,-75.0590),('washington-crossing-sp',40.3050,-74.8730),('monmouth-battlefield-sp',40.2570,-74.3270),
 ('turkey-swamp-park',40.1890,-74.3060),('ocean-county-park',40.0960,-74.1880),('van-saun-park',40.9300,-74.0430),
 ('overpeck-park',40.8650,-73.9900),('central-park-morris',40.8640,-74.5150),('lord-stirling-park',40.7080,-74.5300),
 ('cape-may-county-park',39.1050,-74.8000),('estell-manor-park',39.4100,-74.7400),('parvin-sp',39.5100,-75.1300),
 ('veterans-park-hamilton',40.2050,-74.6690),('rosedale-park',40.3130,-74.7200),('garret-mountain',40.9050,-74.1650),
 ('donaldson-park-highland-park',40.5080,-74.4320),('roosevelt-park-edison',40.5610,-74.3510),('mercer-county-park-SHOWCASE-CHECK',40.26515,-74.6475),
]
OUT.mkdir(parents=True, exist_ok=True); report=[]
for sid,lat,lon in SITES:
    x,y=utm18(lat,lon); ext=[x-256,y-256,x+256,y+256]; d=OUT/sid; d.mkdir(exist_ok=True)
    common={'bbox':','.join(f'{v:.3f}' for v in ext),'bboxSR':32618,'imageSR':32618,'f':'image'}
    dem=DEM+'/exportImage?'+urlencode({**common,'size':'128,128','format':'tiff','pixelType':'F32','interpolation':'RSP_BilinearInterpolation','noData':-9999})
    img=IMG+'/export?'+urlencode({**common,'size':'512,512','format':'jpg','transparent':'false'})
    ok=True
    for url,name in [(dem,'dem-512m-4m.tiff'),(img,'aerial-2015-512m.jpg')]:
        r=subprocess.run(['curl','-fsSL','--retry','2','--max-time','90',url,'-o',str(d/name)])
        ok&=r.returncode==0
    meta={'id':sid,'latApprox':lat,'lonApprox':lon,'utm18':[x,y],'boundsM':ext,'demUrl':dem,'imageryUrl':img,'ok':ok,
          'sha256':{n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in ['dem-512m-4m.tiff','aerial-2015-512m.jpg'] if (d/n).exists()}}
    (d/'meta.json').write_text(json.dumps(meta,indent=2)); report.append(meta); print(sid, 'ok' if ok else 'FAILED', f'{x:.1f},{y:.1f}', flush=True)
(OUT/'fetch-report.json').write_text(json.dumps(report,indent=2))
