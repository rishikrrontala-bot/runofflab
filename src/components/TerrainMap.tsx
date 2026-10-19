import { useEffect, useLayoutEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Grid } from '../core/types';
import { depthLUT, lutIndex, paintBase, paintOverlay, VERTICAL_EXAGGERATION, type Overlay } from './mapLayers';

/**
 * 3D terrain view, loaded on demand so the flat map (the phone default) never downloads Three.js.
 * Renders on demand only: camera moves, new water states and overlay changes.
 */

export interface TerrainProps {
  grid: Grid;
  aerial?: HTMLImageElement;
  depth?: Float32Array;
  overlay: Overlay;
  title: string;
  reset: number;
  onFail: () => void;
}

interface Scene {
  paintTexture: (aerial: HTMLImageElement | undefined, overlay: Overlay) => void;
  setDepth: (depth?: Float32Array) => void;
  resetView: () => void;
}

const TEXTURE_SIZE = 1536;

export default function TerrainMap({ grid, aerial, depth, overlay, title, reset, onFail }: TerrainProps) {
  const host = useRef<HTMLDivElement>(null);
  const scene = useRef<Scene | undefined>(undefined);
  const latest = useRef({ aerial, depth, overlay, onFail });
  latest.current = { aerial, depth, overlay, onFail };

  useLayoutEffect(() => {
    const el = host.current;
    if (!el) return;
    const g = grid, nx = g.nx, ny = g.ny, exag = VERTICAL_EXAGGERATION;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch {
      latest.current.onFail();
      return;
    }
    el.appendChild(renderer.domElement);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor('#E7E4DA');
    renderer.domElement.setAttribute('role', 'img');
    renderer.domElement.setAttribute('aria-label',
      `${title}: 3D terrain of the 384 m computed domain, vertical exaggeration ${exag} times. Same outlines and labels as the flat map.`);

    let low = Infinity;
    for (const z of g.z) low = Math.min(low, z);

    // Terrain: a smooth heightfield through cell corners (mean of the adjoining cells), draped with the
    // aerial and the map overlay. Display only; the solver uses cell-centre elevations.
    const cornerY = new Float32Array((nx + 1) * (ny + 1));
    for (let cy = 0; cy <= ny; cy++) for (let cx = 0; cx <= nx; cx++) {
      let sum = 0, count = 0;
      for (const [x, y] of [[cx - 1, cy - 1], [cx, cy - 1], [cx - 1, cy], [cx, cy]]) {
        if (x < 0 || y < 0 || x >= nx || y >= ny) continue;
        sum += g.z[y * nx + x]; count++;
      }
      cornerY[cy * (nx + 1) + cx] = (sum / count - low) * exag;
    }
    const terrainPos = new Float32Array(cornerY.length * 3), terrainUv = new Float32Array(cornerY.length * 2);
    for (let cy = 0; cy <= ny; cy++) for (let cx = 0; cx <= nx; cx++) {
      const k = cy * (nx + 1) + cx;
      terrainPos.set([(cx - nx / 2) * g.dx, cornerY[k], (cy - ny / 2) * g.dy], k * 3);
      terrainUv.set([cx / nx, 1 - cy / ny], k * 2);
    }
    const terrainIndex: number[] = [];
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const a = y * (nx + 1) + x, b = a + 1, c = a + nx + 1, d = c + 1;
      terrainIndex.push(a, c, b, b, c, d);
    }
    const terrain = new THREE.BufferGeometry();
    terrain.setAttribute('position', new THREE.BufferAttribute(terrainPos, 3));
    terrain.setAttribute('uv', new THREE.BufferAttribute(terrainUv, 2));
    terrain.setIndex(terrainIndex);
    terrain.computeVertexNormals();

    const textureCanvas = document.createElement('canvas');
    textureCanvas.width = TEXTURE_SIZE; textureCanvas.height = TEXTURE_SIZE;
    const textureCtx = textureCanvas.getContext('2d');
    const overCanvas = document.createElement('canvas');
    overCanvas.width = TEXTURE_SIZE; overCanvas.height = TEXTURE_SIZE;
    const overCtx = overCanvas.getContext('2d');
    const texture = new THREE.CanvasTexture(textureCanvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    const terrainMaterial = new THREE.MeshLambertMaterial({ map: texture });
    const world = new THREE.Scene();
    world.add(new THREE.Mesh(terrain, terrainMaterial));

    // Water: one quad per cell, lifted by its computed depth; colour and opacity from the same scale as the
    // flat map. Buffers are allocated once and rewritten in place for every saved state.
    const cellsCount = nx * ny;
    const waterPos = new Float32Array(cellsCount * 4 * 3), waterBaseY = new Float32Array(cellsCount * 4);
    const waterColor = new Float32Array(cellsCount * 4 * 4), waterIndex = new Uint32Array(cellsCount * 6);
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const i = y * nx + x;
      const corners = [[x, y], [x + 1, y], [x, y + 1], [x + 1, y + 1]];
      corners.forEach(([cx, cy], j) => {
        const v = i * 4 + j;
        waterBaseY[v] = cornerY[cy * (nx + 1) + cx];
        waterPos.set([(cx - nx / 2) * g.dx, waterBaseY[v], (cy - ny / 2) * g.dy], v * 3);
      });
      waterIndex.set([i * 4, i * 4 + 2, i * 4 + 1, i * 4 + 1, i * 4 + 2, i * 4 + 3], i * 6);
    }
    const water = new THREE.BufferGeometry();
    const waterPosAttr = new THREE.BufferAttribute(waterPos, 3).setUsage(THREE.DynamicDrawUsage);
    const waterColorAttr = new THREE.BufferAttribute(waterColor, 4).setUsage(THREE.DynamicDrawUsage);
    water.setAttribute('position', waterPosAttr);
    water.setAttribute('color', waterColorAttr);
    water.setIndex(new THREE.BufferAttribute(waterIndex, 1));
    const waterMaterial = new THREE.MeshBasicMaterial({
      vertexColors: true, transparent: true, depthWrite: false, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    });
    const waterMesh = new THREE.Mesh(water, waterMaterial);
    waterMesh.renderOrder = 2;
    waterMesh.frustumCulled = false;
    world.add(waterMesh);

    world.add(new THREE.AmbientLight(0xffffff, 1.25));
    const sun = new THREE.DirectionalLight(0xffffff, 1.7);
    sun.position.set(-160, 260, 140);
    world.add(sun);

    const camera = new THREE.PerspectiveCamera(34, 1, 1, 5000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    controls.maxPolarAngle = Math.PI * 0.47;
    controls.minDistance = 120;
    controls.maxDistance = 1300;
    // Default view from the south-east; pulled back on narrow canvases so the whole domain stays in frame.
    let userMoved = false;
    controls.addEventListener('start', () => { userMoved = true; });
    const resetView = () => {
      const fit = Math.max(1, 1.45 / Math.max(0.3, camera.aspect));
      controls.target.set(0, 6, 12);
      camera.position.set(240 * fit, 6 + 320 * fit, 12 + 400 * fit);
      camera.lookAt(controls.target);
      controls.update();
      userMoved = false;
    };
    resetView();

    const render = () => renderer.render(world, camera);

    const paintTexture = (image: HTMLImageElement | undefined, layer: Overlay) => {
      if (!textureCtx || !overCtx) return;
      paintBase(textureCtx, TEXTURE_SIZE, g, image);
      paintOverlay(overCtx, TEXTURE_SIZE, g, layer, { unit: 3.2, ids: true, scaleBar: false });
      textureCtx.drawImage(overCanvas, 0, 0);
      texture.needsUpdate = true;
      render();
    };

    const setDepth = (h?: Float32Array) => {
      const valid = !!h && h.length === cellsCount;
      for (let i = 0; i < cellsCount; i++) {
        const d = valid && g.mask[i] && !g.sink?.[i] ? h![i] : 0;
        const k = lutIndex(d), alpha = depthLUT[k + 3] / 255;
        const lift = alpha > 0 ? d * exag + 0.2 : 0;
        const r = depthLUT[k] / 255, gr = depthLUT[k + 1] / 255, b = depthLUT[k + 2] / 255;
        for (let j = 0; j < 4; j++) {
          const v = i * 4 + j;
          waterPos[v * 3 + 1] = waterBaseY[v] + lift;
          waterColor[v * 4] = r; waterColor[v * 4 + 1] = gr; waterColor[v * 4 + 2] = b; waterColor[v * 4 + 3] = alpha;
        }
      }
      waterPosAttr.needsUpdate = true;
      waterColorAttr.needsUpdate = true;
      render();
    };

    const resize = () => {
      const { width, height } = el.getBoundingClientRect();
      if (width < 1 || height < 1) return;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      if (!userMoved) resetView();
      render();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    controls.addEventListener('change', render);

    scene.current = { paintTexture, setDepth, resetView: () => { resetView(); render(); } };
    resize();
    paintTexture(latest.current.aerial, latest.current.overlay);
    setDepth(latest.current.depth);

    return () => {
      scene.current = undefined;
      observer.disconnect();
      controls.removeEventListener('change', render);
      controls.dispose();
      terrain.dispose(); water.dispose();
      terrainMaterial.dispose(); waterMaterial.dispose(); texture.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    };
  }, [grid, title]);

  useEffect(() => scene.current?.paintTexture(aerial, overlay), [aerial, overlay]);
  useEffect(() => scene.current?.setDepth(depth), [depth]);
  useEffect(() => scene.current?.resetView(), [reset]);

  return <div className="three-map" ref={host} />;
}
