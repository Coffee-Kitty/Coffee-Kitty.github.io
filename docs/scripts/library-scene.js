// 3D 浮空图书馆首页场景（程序化几何，数据来自 assets/site-graph.json）
// 计划：plans/2026-10-21-3d-library-homepage.md
import * as THREE from 'three';

const DATA_URL = 'assets/site-graph.json';
const HOME_ROUTES = ['', '/', '/README.md'];
const BOOKS_PER_ROW = 12;
const MAX_BOOKS_PER_SHELF = BOOKS_PER_ROW * 2;

function isHomeRoute() {
  const raw = location.hash.replace(/^#/, '').split('?')[0];
  const path = raw.replace(/\/+$/, '');
  return HOME_ROUTES.includes(path) || HOME_ROUTES.includes(path.toLowerCase());
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function bookColor(hex, rnd) {
  const color = new THREE.Color(hex);
  const hsl = { h: 0, s: 0, l: 0 };
  color.getHSL(hsl);
  color.setHSL(
    (hsl.h + (rnd() - 0.5) * 0.035 + 1) % 1,
    clamp(hsl.s + (rnd() - 0.5) * 0.2, 0.3, 0.62),
    clamp(hsl.l + (rnd() - 0.5) * 0.2, 0.46, 0.74),
  );
  return color;
}

function textTexture(text, options = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = options.background || '#f3e7cf';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = options.border || '#8a6f52';
  ctx.lineWidth = 8;
  ctx.strokeRect(6, 6, canvas.width - 12, canvas.height - 12);
  ctx.fillStyle = options.color || '#5c4630';
  ctx.font = `bold ${options.font || 52}px "PingFang SC", "Noto Sans SC", sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2 + 4);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

function spineTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 64, 128);
  ctx.fillStyle = 'rgba(255, 244, 214, 0.9)';
  ctx.fillRect(6, 18, 52, 5);
  ctx.fillRect(6, 104, 52, 5);
  ctx.fillStyle = 'rgba(60, 40, 20, 0.28)';
  ctx.fillRect(6, 26, 52, 2);
  ctx.fillRect(6, 100, 52, 2);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
  ctx.fillRect(10, 44, 44, 40);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function deckTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ecd9b8';
  ctx.fillRect(0, 0, 256, 256);
  for (let x = 0; x < 256; x += 32) {
    ctx.fillStyle = x % 64 === 0 ? '#e5cfa9' : '#f1e0c2';
    ctx.fillRect(x, 0, 30, 256);
    ctx.fillStyle = 'rgba(150, 118, 80, 0.18)';
    ctx.fillRect(x + 30, 0, 2, 256);
  }
  for (let i = 0; i < 240; i += 1) {
    ctx.fillStyle = `rgba(${80 + Math.random() * 60 | 0}, ${50 + Math.random() * 40 | 0}, 30, 0.08)`;
    ctx.fillRect(Math.random() * 256, Math.random() * 256, 3, 3);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(3, 3);
  return texture;
}

function rugTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#f0e2c8';
  ctx.fillRect(0, 0, 256, 256);
  ctx.strokeStyle = '#c97b5f';
  ctx.lineWidth = 14;
  ctx.beginPath();
  ctx.arc(128, 128, 116, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = '#d9a08b';
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.arc(128, 128, 98, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = 'rgba(154, 176, 122, 0.45)';
  for (let i = 0; i < 36; i += 1) {
    const a = Math.random() * Math.PI * 2;
    const r = Math.random() * 78;
    ctx.beginPath();
    ctx.arc(128 + Math.cos(a) * r, 128 + Math.sin(a) * r, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.strokeStyle = 'rgba(201, 123, 95, 0.35)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(128, 128, 60, 0, Math.PI * 2);
  ctx.stroke();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

let toonGradient = null;
function toon(color, extra = {}) {
  if (!toonGradient) {
    toonGradient = new THREE.DataTexture(new Uint8Array([110, 170, 225, 255]), 4, 1, THREE.RedFormat);
    toonGradient.minFilter = THREE.NearestFilter;
    toonGradient.magFilter = THREE.NearestFilter;
    toonGradient.needsUpdate = true;
  }
  return new THREE.MeshToonMaterial({ color, gradientMap: toonGradient, ...extra });
}

function fluffify(geometry, amount, rnd) {
  const pos = geometry.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i += 1) {
    v.fromBufferAttribute(pos, i);
    v.multiplyScalar(1 + (rnd() - 0.5) * amount);
    pos.setXYZ(i, v.x, v.y, v.z);
  }
  geometry.computeVertexNormals();
  return geometry;
}

function furPoints(radius, count, color, rnd) {
  const arr = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    const theta = rnd() * Math.PI * 2;
    const phi = Math.acos(2 * rnd() - 1);
    const r = radius * (1 + rnd() * 0.16);
    arr[i * 3] = Math.sin(phi) * Math.cos(theta) * r;
    arr[i * 3 + 1] = Math.cos(phi) * r;
    arr[i * 3 + 2] = Math.sin(phi) * Math.sin(theta) * r;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3));
  return new THREE.Points(
    geo,
    new THREE.PointsMaterial({ color, size: 0.05, transparent: true, opacity: 0.8, depthWrite: false }),
  );
}

async function loadData() {
  const res = await fetch(DATA_URL, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`site-graph.json HTTP ${res.status}`);
  return res.json();
}

function buildMeta(data) {
  const stats = document.querySelector('#library-stats');
  if (stats) {
    stats.textContent = `${data.categories.length} 个主题 · ${data.totalPosts} 篇公开笔记 · 索引生成于 ${data.generated}`;
  }
  const legend = document.querySelector('#library-legend');
  if (legend) {
    legend.textContent = '';
    for (const cat of data.categories) {
      const chip = document.createElement('a');
      chip.className = 'legend-chip';
      chip.href = `#${cat.url}`;
      chip.style.setProperty('--chip', cat.color);
      chip.title = `进入 ${cat.name}`;
      const dot = document.createElement('i');
      const label = document.createElement('span');
      label.textContent = cat.name;
      const count = document.createElement('em');
      count.textContent = String(cat.posts.length);
      chip.append(dot, label, count);
      legend.appendChild(chip);
    }
  }
}

function initScene(data, container) {
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const small = container.clientWidth < 640;
  const rnd = mulberry32(20261021);
  const portrait = container.clientWidth / Math.max(container.clientHeight, 1) < 0.95;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, small ? 1.5 : 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);
  renderer.domElement.setAttribute('aria-label', '浮空图书馆 3D 场景：拖动旋转，滚轮缩放，点击书架或书脊跳转');

  const tooltip = document.createElement('div');
  tooltip.className = 'scene-tooltip';
  tooltip.hidden = true;
  container.appendChild(tooltip);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 120);
  const target = new THREE.Vector3(0, portrait ? 2.6 : 0.2, 0);

  scene.add(new THREE.HemisphereLight(0xfff3e0, 0xb08a63, 0.95));
  const key = new THREE.DirectionalLight(0xffdfae, 2.1);
  key.position.set(9, 15, 7);
  key.castShadow = true;
  key.shadow.mapSize.set(small ? 512 : 1024, small ? 512 : 1024);
  key.shadow.camera.left = -15;
  key.shadow.camera.right = 15;
  key.shadow.camera.top = 15;
  key.shadow.camera.bottom = -15;
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 45;
  key.shadow.bias = -0.0006;
  key.shadow.normalBias = 0.02;
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xa9c4de, 0.7);
  rim.position.set(-9, 7, -9);
  scene.add(rim);

  // 暖阳晕 + 距离雾，营造空气感
  const glowCanvas = document.createElement('canvas');
  glowCanvas.width = glowCanvas.height = 128;
  const gctx = glowCanvas.getContext('2d');
  const grad = gctx.createRadialGradient(64, 64, 4, 64, 64, 62);
  grad.addColorStop(0, 'rgba(255, 240, 210, 0.9)');
  grad.addColorStop(0.5, 'rgba(255, 224, 190, 0.35)');
  grad.addColorStop(1, 'rgba(255, 224, 190, 0)');
  gctx.fillStyle = grad;
  gctx.fillRect(0, 0, 128, 128);
  const glowTex = new THREE.CanvasTexture(glowCanvas);
  glowTex.colorSpace = THREE.SRGBColorSpace;
  const sun = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: glowTex, transparent: true, opacity: 0.85, depthWrite: false, fog: false }),
  );
  sun.position.set(12, 13, -16);
  sun.scale.set(26, 26, 1);
  scene.add(sun);
  scene.fog = new THREE.Fog(0xf7e0d2, 36, 68);

  const island = new THREE.Group();
  scene.add(island);

  const wood = (color) => toon(color);

  const deckTop = 0.55;
  const deck = new THREE.Mesh(
    new THREE.CylinderGeometry(10.6, 9.8, 1.1, 10),
    toon(0xffffff, { map: deckTexture() }),
  );
  deck.receiveShadow = true;
  island.add(deck);
  const rock = new THREE.Mesh(new THREE.CylinderGeometry(9.4, 1.1, 6.0, 9), wood(0xa08468));
  rock.position.y = -3.5;
  island.add(rock);
  const strata1 = new THREE.Mesh(new THREE.CylinderGeometry(8.2, 6.4, 1.2, 9), wood(0x8d7256));
  strata1.position.y = -2.2;
  island.add(strata1);
  const strata2 = new THREE.Mesh(new THREE.CylinderGeometry(5.6, 3.6, 1.4, 9), wood(0x77604a));
  strata2.position.y = -4.6;
  island.add(strata2);
  for (let i = 0; i < 4; i += 1) {
    const root = new THREE.Mesh(new THREE.ConeGeometry(0.5 + rnd() * 0.4, 2.4 + rnd() * 1.6, 5), wood(0x6e5844));
    const ra = rnd() * Math.PI * 2;
    root.position.set(Math.sin(ra) * (2 + rnd() * 3), -6.6 - rnd() * 0.8, Math.cos(ra) * (2 + rnd() * 3));
    root.rotation.x = Math.PI;
    island.add(root);
  }
  const grassRim = new THREE.Mesh(new THREE.TorusGeometry(10.3, 0.4, 8, 28), toon(0xb5c99a));
  grassRim.rotation.x = Math.PI / 2;
  grassRim.scale.z = 0.5;
  grassRim.position.y = deckTop + 0.12;
  island.add(grassRim);
  for (let i = 0; i < 3; i += 1) {
    const bump = new THREE.Mesh(new THREE.DodecahedronGeometry(1.1 + rnd() * 0.8, 0), wood(0x7d6450));
    const a = rnd() * Math.PI * 2;
    bump.position.set(Math.sin(a) * 7.4, -1.6 - rnd() * 1.8, Math.cos(a) * 7.4);
    island.add(bump);
  }
  const rug = new THREE.Mesh(
    new THREE.CylinderGeometry(3.5, 3.5, 0.1, 24),
    wood(0xa3543f),
  );
  rug.position.y = deckTop + 0.05;
  rug.receiveShadow = true;
  island.add(rug);
  const rugTop = new THREE.Mesh(
    new THREE.CircleGeometry(3.48, 32),
    toon(0xffffff, { map: rugTexture() }),
  );
  rugTop.rotation.x = -Math.PI / 2;
  rugTop.position.y = deckTop + 0.11;
  rugTop.receiveShadow = true;
  island.add(rugTop);

  const pickables = [];
  const shelves = [];
  const shelfRadius = 7.4;
  const categories = data.categories;
  const spineTex = spineTexture();

  categories.forEach((cat, index) => {
    const angle = (index / categories.length) * Math.PI * 2 + 0.32;
    const group = new THREE.Group();
    group.position.set(Math.sin(angle) * shelfRadius, deckTop, Math.cos(angle) * shelfRadius);
    group.rotation.y = angle + Math.PI;
    group.userData = { type: 'shelf', name: cat.name, url: cat.url, count: cat.posts.length };

    const shown = cat.posts.slice(0, MAX_BOOKS_PER_SHELF);
    const rows = shown.length > BOOKS_PER_ROW ? 2 : 1;
    const perRow = Math.max(1, Math.ceil(shown.length / rows));
    const width = clamp(perRow * 0.4 + 0.8, 2.6, 5.6);
    const height = rows === 2 ? 3.15 : 1.95;
    const depth = 1.0;
    const frame = wood(0x8a6a48);
    const shell = toon(0xe9dcc3);
    const parts = [
      [-width / 2, height / 2, 0, 0.18, height, depth, shell],
      [width / 2, height / 2, 0, 0.18, height, depth, shell],
      [0, 0.07, 0, width, 0.14, depth, frame],
      [0, height - 0.07, 0, width, 0.14, depth, shell],
    ];
    if (rows === 2) parts.push([0, height / 2, 0, width, 0.12, depth, frame]);
    for (const [x, y, z, w, h, d, mat] of parts) {
      const board = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
      board.position.set(x, y, z);
      board.castShadow = true;
      board.receiveShadow = true;
      board.userData = { type: 'shelf', name: cat.name, url: cat.url, count: cat.posts.length };
      group.add(board);
      pickables.push(board);
    }
    const back = new THREE.Mesh(new THREE.BoxGeometry(width, height, 0.1), shell);
    back.position.set(0, height / 2, -depth / 2 + 0.05);
    back.userData = { type: 'shelf', name: cat.name, url: cat.url, count: cat.posts.length };
    group.add(back);
    pickables.push(back);
    const innerBack = new THREE.Mesh(new THREE.PlaneGeometry(width - 0.2, height - 0.3), frame);
    innerBack.position.set(0, height / 2, -depth / 2 + 0.12);
    group.add(innerBack);
    const glow = new THREE.Mesh(
      new THREE.BoxGeometry(width - 0.36, 0.05, 0.05),
      toon(0x3a2a1a, { emissive: 0xffc98a, emissiveIntensity: 1.1 }),
    );
    glow.position.set(0, height - 0.18, 0.3);
    group.add(glow);
    const crown = new THREE.Mesh(new THREE.BoxGeometry(width + 0.34, 0.16, depth + 0.2), toon(0xe8d9bd));
    crown.position.set(0, height + 0.1, 0);
    crown.castShadow = true;
    group.add(crown);
    const crest = new THREE.Mesh(new THREE.CircleGeometry(0.17, 16), toon(cat.color));
    crest.position.set(0, height + 0.2, depth / 2 + 0.11);
    group.add(crest);
    group.userData.anchorY = height + 0.9;
    group.userData.angle = angle;

    const placeBooks = (list, boardY) => {
      let cursor = -width / 2 + 0.28;
      for (const post of list) {
        const bw = 0.22 + rnd() * 0.12;
        const bh = 1.02 + rnd() * 0.32;
        const book = new THREE.Mesh(
          new THREE.BoxGeometry(bw, bh, 0.72 + rnd() * 0.14),
          toon(bookColor(cat.color, rnd), { map: spineTex }),
        );
        const tilt = rnd() < 0.14 ? (rnd() - 0.5) * 0.22 : 0;
        book.rotation.z = tilt;
        book.position.set(cursor + bw / 2, boardY + bh / 2 + 0.07, 0.06);
        book.userData = { type: 'book', title: post.title, url: post.url, baseZ: 0.06, cat: cat.name };
        book.castShadow = true;
        group.add(book);
        pickables.push(book);
        cursor += bw + 0.05 + Math.abs(tilt) * 0.6;
      }
      return cursor;
    };

    if (shown.length === 0) {
      const sign = new THREE.Mesh(
        new THREE.PlaneGeometry(1.5, 0.75),
        toon(0xffffff, { map: textTexture('待写') }),
      );
      sign.position.set(0, 0.62, 0.2);
      sign.rotation.x = -0.18;
      group.add(sign);
      pickables.push(sign);
    } else {
      placeBooks(shown.slice(0, perRow), 0.14);
      if (rows === 2) placeBooks(shown.slice(perRow), height / 2 + 0.06);
      if (cat.posts.length > MAX_BOOKS_PER_SHELF) {
        const more = new THREE.Mesh(
          new THREE.BoxGeometry(0.42, 1.3, 0.86),
          toon(0xffffff, { map: textTexture(`+${cat.posts.length - MAX_BOOKS_PER_SHELF}`, { font: 60 }) }),
        );
        more.position.set(width / 2 - 0.5, 0.14 + 0.72, 0.06);
        more.userData = { type: 'shelf', name: cat.name, url: cat.url, count: cat.posts.length };
        more.castShadow = true;
        group.add(more);
        pickables.push(more);
      }
    }

    island.add(group);
    shelves.push(group);
  });

  // 书架分类标签（HTML 覆盖层，始终清晰可读）
  const labelLayer = document.createElement('div');
  labelLayer.className = 'shelf-labels';
  container.appendChild(labelLayer);
  const labels = categories.map((cat, i) => {
    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'shelf-label';
    el.style.setProperty('--chip', cat.color);
    el.title = `进入 ${cat.name}`;
    const dot = document.createElement('i');
    const name = document.createElement('span');
    name.textContent = cat.name;
    const count = document.createElement('em');
    count.textContent = String(cat.posts.length);
    el.append(dot, name, count);
    el.addEventListener('click', () => {
      openShelf(cat.name);
    });
    labelLayer.appendChild(el);
    return { el, shelf: shelves[i] };
  });

  // 分类书目面板：点书架/名牌/合集书脊后弹出
  const panel = document.createElement('aside');
  panel.className = 'shelf-panel';
  panel.hidden = true;
  container.appendChild(panel);

  function openShelf(name) {
    const cat = categories.find((item) => item.name === name);
    const shelf = shelves.find((item) => item.userData.name === name);
    if (!cat || !shelf) return;
    panelOpen = true;
    panel.textContent = '';
    const head = document.createElement('header');
    const dot = document.createElement('i');
    dot.style.background = cat.color;
    const title = document.createElement('h3');
    title.textContent = cat.name;
    const count = document.createElement('em');
    count.textContent = `${cat.posts.length} 篇`;
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'shelf-panel-close';
    close.setAttribute('aria-label', '关闭书目');
    close.textContent = '×';
    close.addEventListener('click', closeShelf);
    head.append(dot, title, count, close);
    const enter = document.createElement('a');
    enter.className = 'shelf-panel-enter';
    enter.href = `#${cat.url}`;
    enter.textContent = '进入分类总览 →';
    const list = document.createElement('ul');
    for (const post of cat.posts) {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = `#${post.url}`;
      a.textContent = post.title;
      li.appendChild(a);
      if (post.date) {
        const time = document.createElement('time');
        time.textContent = post.date;
        li.appendChild(time);
      }
      list.appendChild(li);
    }
    if (!cat.posts.length) {
      const li = document.createElement('li');
      li.className = 'empty';
      li.textContent = '还没有公开笔记，书架待写…';
      list.appendChild(li);
    }
    panel.append(head, enter, list);
    panel.hidden = false;
    const want = (shelf.userData.angle || 0) + Math.PI;
    const diff = ((((want - orbit.tTheta) % (Math.PI * 2)) + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
    orbit.tTheta += diff;
    orbit.tPhi = 0.95;
    orbit.tRadius = 19;
    lastInteract = performance.now();
  }

  function closeShelf() {
    panelOpen = false;
    panel.hidden = true;
    orbit.tRadius = startRadius;
    orbit.tPhi = startPhi;
    lastInteract = performance.now();
  }

  const labelPos = new THREE.Vector3();
  const labelDir = new THREE.Vector3();
  const labelToCam = new THREE.Vector3();
  function updateLabels() {
    for (const label of labels) {
      const shelf = label.shelf;
      labelPos.copy(shelf.position);
      labelPos.y += shelf.userData.anchorY || 3;
      labelPos.project(camera);
      shelf.getWorldPosition(labelToCam);
      labelDir.copy(camera.position).sub(labelToCam).normalize();
      shelf.getWorldDirection(labelToCam);
      const facing = labelToCam.dot(labelDir);
      const visible = labelPos.z < 1 && facing > 0.12;
      label.el.hidden = !visible;
      if (!visible) continue;
      const lx = labelBox.x + (labelPos.x * 0.5 + 0.5) * labelBox.w;
      const ly = labelBox.y + (-labelPos.y * 0.5 + 0.5) * labelBox.h;
      const inCopy =
        lx > copyBox.x - 40 &&
        lx < copyBox.x + copyBox.w + 40 &&
        ly > copyBox.y - 44 &&
        ly < copyBox.y + copyBox.h + 12;
      label.el.hidden = inCopy;
      if (inCopy) continue;
      label.el.style.opacity = clamp((facing - 0.12) * 2.4, 0, 1).toFixed(2);
      label.el.style.left = `${lx}px`;
      label.el.style.top = `${ly}px`;
    }
  }

  // 中心书桌 + 猫
  const desk = new THREE.Group();
  desk.position.y = deckTop;
  const deskTop = new THREE.Mesh(new THREE.BoxGeometry(2.7, 0.16, 1.5), wood(0x7a5230));
  deskTop.position.y = 0.95;
  deskTop.castShadow = true;
  desk.add(deskTop);
  for (const [lx, lz] of [[-1.2, -0.6], [1.2, -0.6], [-1.2, 0.6], [1.2, 0.6]]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.9, 0.12), wood(0x6b4626));
    leg.position.set(lx, 0.45, lz);
    leg.castShadow = true;
    desk.add(leg);
  }
  for (const side of [-1, 1]) {
    const page = new THREE.Mesh(
      new THREE.BoxGeometry(0.62, 0.035, 0.8),
      toon(0xf6efdf),
    );
    page.position.set(side * 0.3, 1.06, 0.1);
    page.rotation.z = side * -0.22;
    desk.add(page);
  }
  const lampPole = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.06, 0.85, 8), wood(0x4c3a28));
  lampPole.position.set(-0.95, 1.45, -0.4);
  desk.add(lampPole);
  const lampShade = new THREE.Mesh(
    new THREE.ConeGeometry(0.36, 0.42, 12, 1, true),
    toon(0xcaa26a, {
      emissive: 0xffd9a0,
      emissiveIntensity: 0.85,
      side: THREE.DoubleSide,
    }),
  );
  lampShade.position.set(-0.95, 1.92, -0.4);
  desk.add(lampShade);
  const lampLight = new THREE.PointLight(0xffcf8a, 9, 9, 2);
  lampLight.position.set(-0.95, 1.8, -0.4);
  desk.add(lampLight);

  const cat = new THREE.Group();
  const fur = toon(0x4a4441);
  const body = new THREE.Mesh(fluffify(new THREE.SphereGeometry(0.34, 12, 10), 0.1, rnd), fur);
  body.scale.set(1, 1.18, 0.92);
  body.position.y = 0.36;
  body.castShadow = true;
  cat.add(body);
  const bodyFur = furPoints(0.35, 110, 0x6b6560, rnd);
  bodyFur.scale.copy(body.scale);
  bodyFur.position.y = 0.36;
  cat.add(bodyFur);
  const head = new THREE.Mesh(fluffify(new THREE.SphereGeometry(0.3, 12, 10), 0.12, rnd), fur);
  head.position.set(0, 0.88, 0.06);
  head.castShadow = true;
  cat.add(head);
  const headFur = furPoints(0.31, 90, 0x6b6560, rnd);
  headFur.position.set(0, 0.88, 0.06);
  cat.add(headFur);
  for (const side of [-1, 1]) {
    const ear = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.24, 4), fur);
    ear.position.set(side * 0.15, 1.12, 0.05);
    ear.rotation.z = side * -0.18;
    cat.add(ear);
    const inner = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.12, 4), toon(0xf2b8c6));
    inner.position.set(side * 0.15, 1.1, 0.12);
    inner.rotation.z = side * -0.18;
    cat.add(inner);
    const eye = new THREE.Mesh(
      new THREE.SphereGeometry(0.055, 10, 8),
      toon(0x2b2b28, { emissive: 0xffe9b8, emissiveIntensity: 0.35 }),
    );
    eye.position.set(side * 0.11, 0.9, 0.29);
    cat.add(eye);
    const shine = new THREE.Mesh(
      new THREE.SphereGeometry(0.018, 6, 6),
      toon(0xffffff, { emissive: 0xffffff, emissiveIntensity: 0.8 }),
    );
    shine.position.set(side * 0.09, 0.93, 0.33);
    cat.add(shine);
    const blush = new THREE.Mesh(new THREE.SphereGeometry(0.05, 8, 6), toon(0xf2b8c6));
    blush.scale.set(1, 0.6, 0.4);
    blush.position.set(side * 0.2, 0.8, 0.24);
    cat.add(blush);
  }
  const chest = new THREE.Mesh(
    fluffify(new THREE.SphereGeometry(0.17, 10, 8), 0.12, rnd),
    toon(0xf3ead9),
  );
  chest.position.set(0, 0.34, 0.24);
  chest.scale.set(1, 1.2, 0.7);
  cat.add(chest);
  const patch = new THREE.Mesh(
    fluffify(new THREE.SphereGeometry(0.13, 8, 6), 0.15, rnd),
    toon(0xb85c38),
  );
  patch.position.set(0.16, 0.9, -0.02);
  cat.add(patch);
  const collar = new THREE.Mesh(
    new THREE.TorusGeometry(0.15, 0.035, 6, 14),
    toon(0x2f6f68),
  );
  collar.position.set(0, 0.62, 0.02);
  collar.rotation.x = Math.PI / 2.2;
  cat.add(collar);
  const tailCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0.12, 0.28, -0.3),
    new THREE.Vector3(0.05, 0.62, -0.42),
    new THREE.Vector3(-0.08, 0.8, -0.3),
  ]);
  const tailPivot = new THREE.Group();
  tailPivot.position.set(0, 0.16, -0.28);
  const tail = new THREE.Mesh(new THREE.TubeGeometry(tailCurve, 14, 0.06, 6), fur);
  tailPivot.add(tail);
  cat.add(tailPivot);
  cat.position.set(0.85, 1.03, -0.25);
  cat.rotation.y = 0.95;
  desk.add(cat);
  island.add(desk);
  cat.traverse((obj) => {
    if (obj.isMesh) {
      obj.userData = { type: 'cat' };
      pickables.push(obj);
    }
  });

  // 地毯上蝙睡的小奶猫
  const kitten = new THREE.Group();
  const kfur = toon(0xf0dcc3);
  const kbody = new THREE.Mesh(fluffify(new THREE.SphereGeometry(0.42, 12, 10), 0.1, rnd), kfur);
  kbody.scale.set(1.15, 0.72, 1);
  kbody.castShadow = true;
  kitten.add(kbody);
  const khead = new THREE.Mesh(fluffify(new THREE.SphereGeometry(0.24, 12, 10), 0.12, rnd), kfur);
  khead.position.set(0.3, 0.16, 0.18);
  kitten.add(khead);
  for (const side of [-1, 1]) {
    const kear = new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.16, 4), kfur);
    kear.position.set(0.3 + side * 0.12, 0.36, 0.16);
    kear.rotation.z = side * -0.2;
    kitten.add(kear);
  }
  const kpatch = new THREE.Mesh(fluffify(new THREE.SphereGeometry(0.14, 8, 6), 0.15, rnd), toon(0xd98e5f));
  kpatch.position.set(-0.18, 0.2, -0.1);
  kitten.add(kpatch);
  const ktail = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.07, 6, 14, Math.PI * 1.15), kfur);
  ktail.rotation.x = Math.PI / 2;
  ktail.rotation.z = 0.5;
  ktail.position.set(-0.1, -0.12, 0.12);
  kitten.add(ktail);
  const kfurCloud = furPoints(0.44, 90, 0xf7ead8, rnd);
  kfurCloud.scale.set(1.15, 0.75, 1);
  kitten.add(kfurCloud);
  kitten.position.set(-1.5, deckTop + 0.42, 1.3);
  kitten.rotation.y = 0.9;
  island.add(kitten);
  kitten.traverse((obj) => {
    if (obj.isMesh) {
      obj.userData = { type: 'cat' };
      pickables.push(obj);
    }
  });

  // 飞书 / 浮尘 / 碎石
  const flyers = [];
  for (let i = 0; i < 3; i += 1) {
    const flyer = new THREE.Group();
    const cover = new THREE.Mesh(
      new THREE.BoxGeometry(0.68, 0.06, 0.88),
      toon(bookColor(categories[i % categories.length].color, rnd)),
    );
    const pages = new THREE.Mesh(
      new THREE.BoxGeometry(0.6, 0.045, 0.78),
      toon(0xf8f2e4),
    );
    pages.position.y = 0.045;
    flyer.add(cover, pages);
    flyer.userData = { radius: 5.2 + i * 1.0, speed: 0.1 + i * 0.035, height: 6.6 + i * 0.8, phase: i * 2.1 };
    flyer.scale.setScalar(0.85);
    scene.add(flyer);
    flyers.push(flyer);
  }
  const dustCount = small ? 60 : 120;
  const dustPos = new Float32Array(dustCount * 3);
  for (let i = 0; i < dustCount; i += 1) {
    const a = rnd() * Math.PI * 2;
    const r = 2 + rnd() * 8;
    dustPos[i * 3] = Math.sin(a) * r;
    dustPos[i * 3 + 1] = 0.8 + rnd() * 7.5;
    dustPos[i * 3 + 2] = Math.cos(a) * r;
  }
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute('position', new THREE.Float32BufferAttribute(dustPos, 3));
  const dust = new THREE.Points(
    dustGeo,
    new THREE.PointsMaterial({ color: 0xe8d9b8, size: 0.07, transparent: true, opacity: 0.55, depthWrite: false }),
  );
  scene.add(dust);
  const chunks = [];
  for (let i = 0; i < 3; i += 1) {
    const chunk = new THREE.Mesh(new THREE.DodecahedronGeometry(0.55 + rnd() * 0.5, 0), wood(0x8a6f5c));
    const a = rnd() * Math.PI * 2;
    chunk.userData = { angle: a, radius: 3.5 + rnd() * 3, height: -5.2 - rnd() * 1.6, speed: 0.05 + rnd() * 0.05 };
    scene.add(chunk);
    chunks.push(chunk);
  }

  // 毛绒云
  const clouds = [];
  for (let i = 0; i < 4; i += 1) {
    const cloud = new THREE.Group();
    const puffs = 3 + Math.floor(rnd() * 2);
    for (let p = 0; p < puffs; p += 1) {
      const puff = new THREE.Mesh(
        fluffify(new THREE.SphereGeometry(0.9 + rnd() * 0.7, 10, 8), 0.18, rnd),
        toon(0xfff8ef, { transparent: true, opacity: 0.85 }),
      );
      puff.position.set((p - puffs / 2) * 1.1, (rnd() - 0.5) * 0.5, (rnd() - 0.5) * 0.8);
      cloud.add(puff);
    }
    cloud.scale.set(1.5, 0.6, 1.2);
    cloud.userData = {
      angle: rnd() * Math.PI * 2,
      radius: 10 + rnd() * 4,
      height: -5 - rnd() * 2,
      speed: 0.03 + rnd() * 0.03,
    };
    scene.add(cloud);
    clouds.push(cloud);
  }

  // 书架间的小灯串
  const bulbs = [];
  for (let i = 0; i < shelves.length; i += 1) {
    const a = shelves[i];
    const b = shelves[(i + 1) % shelves.length];
    const ha = (a.userData.anchorY || 3) - 0.9;
    const hb = (b.userData.anchorY || 3) - 0.9;
    const start = new THREE.Vector3(a.position.x, deckTop + ha + 0.15, a.position.z);
    const end = new THREE.Vector3(b.position.x, deckTop + hb + 0.15, b.position.z);
    const mid = start.clone().add(end).multiplyScalar(0.5);
    mid.y -= 0.9;
    const wire = new THREE.Mesh(
      new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(start, mid, end), 12, 0.015, 4),
      toon(0x6b5644),
    );
    island.add(wire);
    for (let t = 1; t <= 5; t += 1) {
      const u = t / 6;
      const p = new THREE.Vector3()
        .addScaledVector(start, (1 - u) * (1 - u))
        .addScaledVector(mid, 2 * (1 - u) * u)
        .addScaledVector(end, u * u);
      const bulb = new THREE.Mesh(
        new THREE.SphereGeometry(0.07, 6, 6),
        toon(0xfff1c9, { emissive: 0xffc98a, emissiveIntensity: 1.4 }),
      );
      bulb.position.copy(p);
      island.add(bulb);
      bulbs.push(bulb);
    }
  }

  // 地毯上的坐垫
  for (const [cx, cz, cc] of [
    [1.9, 1.6, 0xf2b8c6],
    [-2.1, -1.4, 0xbcd4e6],
  ]) {
    const cushion = new THREE.Mesh(fluffify(new THREE.SphereGeometry(0.55, 10, 8), 0.12, rnd), toon(cc));
    cushion.scale.set(1, 0.42, 1);
    cushion.position.set(cx, deckTop + 0.3, cz);
    cushion.castShadow = true;
    island.add(cushion);
  }

  // 小道具：书堆、盆栽、石径
  for (const [px, pz] of [
    [3.4, 0.9],
    [-3.2, 2.6],
  ]) {
    const pile = new THREE.Group();
    for (let b = 0; b < 3; b += 1) {
      const tome = new THREE.Mesh(
        new THREE.BoxGeometry(0.9 - b * 0.12, 0.16, 0.66 - b * 0.08),
        toon(bookColor(categories[b % categories.length].color, rnd)),
      );
      tome.position.y = 0.08 + b * 0.17;
      tome.rotation.y = (rnd() - 0.5) * 0.5;
      tome.castShadow = true;
      pile.add(tome);
    }
    pile.position.set(px, deckTop + 0.02, pz);
    island.add(pile);
  }
  for (const [px, pz] of [
    [5.2, -4.6],
    [-5.4, 4.2],
  ]) {
    const plant = new THREE.Group();
    const pot = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.26, 0.5, 10), toon(0xc97b5f));
    pot.position.y = 0.25;
    plant.add(pot);
    const leaf = new THREE.Mesh(fluffify(new THREE.SphereGeometry(0.5, 10, 8), 0.2, rnd), toon(0x9db889));
    leaf.position.y = 0.85;
    leaf.castShadow = true;
    plant.add(leaf);
    plant.position.set(px, deckTop, pz);
    island.add(plant);
  }
  for (let s = 0; s < 4; s += 1) {
    const stone = new THREE.Mesh(
      new THREE.CylinderGeometry(0.5 - s * 0.04, 0.55 - s * 0.04, 0.1, 8),
      toon(0xe6d3b3),
    );
    const t = s / 3;
    stone.position.set(4.6 - t * 1.6, deckTop + 0.06, 5.4 - t * 2.2);
    stone.rotation.y = rnd() * Math.PI;
    stone.receiveShadow = true;
    island.add(stone);
  }

  // 轨道控制（手写球坐标）
  const startRadius = portrait ? 46 : 34;
  const startPhi = portrait ? 0.9 : 0.98;
  const orbit = {
    theta: 0.65,
    phi: startPhi,
    radius: startRadius + 9,
    tTheta: 0.65,
    tPhi: startPhi,
    tRadius: startRadius,
  };
  let lastInteract = performance.now();
  const pointers = new Map();
  let pinchBase = 0;
  let downAt = 0;
  let moved = 0;
  const pointerNdc = new THREE.Vector2(-2, -2);
  let pointerInside = false;

  const canvas = renderer.domElement;
  canvas.style.touchAction = 'none';

  const onPointerDown = (event) => {
    canvas.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.size === 1) {
      downAt = performance.now();
      moved = 0;
    } else if (pointers.size === 2) {
      const [p1, p2] = [...pointers.values()];
      pinchBase = Math.hypot(p1.x - p2.x, p1.y - p2.y);
    }
    lastInteract = performance.now();
  };
  const onPointerMove = (event) => {
    const rect = canvas.getBoundingClientRect();
    lastInteract = performance.now();
    pointerNdc.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    pointerInside = true;
    const contRect = container.getBoundingClientRect();
    tooltip.style.left = `${event.clientX - contRect.left}px`;
    tooltip.style.top = `${event.clientY - contRect.top}px`;
    const prev = pointers.get(event.pointerId);
    if (!prev) return;
    const dx = event.clientX - prev.x;
    const dy = event.clientY - prev.y;
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    moved += Math.abs(dx) + Math.abs(dy);
    if (pointers.size === 1) {
      orbit.tTheta -= dx * 0.0052;
      orbit.tPhi = clamp(orbit.tPhi - dy * 0.0042, 0.62, 1.35);
    } else if (pointers.size === 2 && pinchBase > 0) {
      const [p1, p2] = [...pointers.values()];
      const dist = Math.hypot(p1.x - p2.x, p1.y - p2.y);
      orbit.tRadius = clamp(orbit.tRadius * (pinchBase / Math.max(dist, 1)), 12, 55);
      pinchBase = dist;
    }
    lastInteract = performance.now();
  };
  const onPointerUp = (event) => {
    pointers.delete(event.pointerId);
    if (pointers.size === 0) pinchBase = 0;
    lastInteract = performance.now();
    const isClick = moved < 6 && performance.now() - downAt < 450;
    if (!isClick) return;
    const hit = pickAt();
    if (!hit) {
      if (panelOpen) closeShelf();
      return;
    }
    const u = hit.userData;
    if (u.type === 'book') {
      location.hash = `#${u.url}`;
      return;
    }
    if (u.type === 'cat') {
      catWiggle = 1;
      tooltip.textContent = '喵～';
      tooltip.hidden = false;
      window.clearTimeout(meowTimer);
      meowTimer = window.setTimeout(() => {
        tooltip.hidden = true;
      }, 1200);
      return;
    }
    if (u.type === 'shelf') openShelf(u.name);
  };
  const onWheel = (event) => {
    event.preventDefault();
    orbit.tRadius = clamp(orbit.tRadius * (1 + event.deltaY * 0.0011), 12, 55);
    lastInteract = performance.now();
  };
  const onLeave = () => {
    pointerInside = false;
    tooltip.hidden = true;
  };
  const onKey = (event) => {
    if (event.key === 'Escape' && panelOpen) closeShelf();
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointerleave', onLeave);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  const onAim = () => {
    lastInteract = performance.now();
  };
  container.addEventListener('pointermove', onAim);
  window.addEventListener('keydown', onKey);

  const raycaster = new THREE.Raycaster();
  let hovered = null;
  const NDC_OFFSETS = [0, -0.02, 0.02];
  function pickAt() {
    for (const ox of NDC_OFFSETS) {
      for (const oy of NDC_OFFSETS) {
        probeNdc.set(pointerNdc.x + ox, pointerNdc.y + oy);
        raycaster.setFromCamera(probeNdc, camera);
        const hits = raycaster.intersectObjects(pickables, false);
        if (hits.length) return hits[0].object;
      }
    }
    return null;
  }
  const probeNdc = new THREE.Vector2();

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const w = Math.round(rect.width);
    const h = Math.round(rect.height);
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    const crect = container.getBoundingClientRect();
    labelBox.x = rect.left - crect.left;
    labelBox.y = rect.top - crect.top;
    labelBox.w = w;
    labelBox.h = h;
    const copy = container.parentElement?.querySelector('.hero-copy');
    if (copy) {
      const kr = copy.getBoundingClientRect();
      copyBox.x = kr.left - crect.left;
      copyBox.y = kr.top - crect.top;
      copyBox.w = kr.width;
      copyBox.h = kr.height;
    }
  }
  const labelBox = { x: 0, y: 0, w: 0, h: 0 };
  const copyBox = { x: 0, y: 0, w: 0, h: 0 };
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(container);

  let paused = false;
  let frozen = false;
  let panelOpen = false;
  let catWiggle = 0;
  let meowTimer = 0;
  const io = new IntersectionObserver((entries) => {
    paused = !entries[0].isIntersecting;
  });
  io.observe(container);

  const clock = { last: performance.now() };
  let elapsed = 0;
  let frame = 0;

  function tick() {
    frame = requestAnimationFrame(tick);
    const now = performance.now();
    const delta = Math.min((now - clock.last) / 1000, 0.05);
    clock.last = now;
    if (paused || document.hidden) return;
    if (!reduced && !frozen) elapsed += delta;

    if (!reduced && !frozen && !panelOpen && performance.now() - lastInteract > 3500) orbit.tTheta += delta * 0.07;
    if (frozen) {
      orbit.theta = orbit.tTheta;
      orbit.phi = orbit.tPhi;
      orbit.radius = orbit.tRadius;
    } else {
      orbit.theta += (orbit.tTheta - orbit.theta) * 0.09;
      orbit.phi += (orbit.tPhi - orbit.phi) * 0.09;
      orbit.radius += (orbit.tRadius - orbit.radius) * 0.09;
    }
    camera.position.set(
      target.x + orbit.radius * Math.sin(orbit.phi) * Math.sin(orbit.theta),
      target.y + orbit.radius * Math.cos(orbit.phi),
      target.z + orbit.radius * Math.sin(orbit.phi) * Math.cos(orbit.theta),
    );
    camera.lookAt(target);
    updateLabels();

    island.position.y = reduced || frozen ? 0 : Math.sin(elapsed * 0.5) * 0.12;
    tailPivot.rotation.z = reduced
      ? 0
      : Math.sin(elapsed * 1.4) * 0.16 + Math.sin(elapsed * 12) * 0.3 * catWiggle;
    catWiggle = Math.max(0, catWiggle - delta * 0.7);
    body.scale.y = 1.18 + (reduced ? 0 : Math.sin(elapsed * 2.1) * 0.02);
    for (const flyer of flyers) {
      const u = flyer.userData;
      const a = u.phase + elapsed * u.speed;
      flyer.position.set(Math.sin(a) * u.radius, u.height + Math.sin(elapsed * 0.8 + u.phase) * 0.25, Math.cos(a) * u.radius);
      flyer.rotation.set(Math.sin(elapsed * 0.6 + u.phase) * 0.3, a + Math.PI / 2, Math.cos(elapsed * 0.5 + u.phase) * 0.25);
    }
    for (const chunk of chunks) {
      const u = chunk.userData;
      const a = u.angle + elapsed * u.speed;
      chunk.position.set(Math.sin(a) * u.radius, u.height + Math.sin(elapsed * 0.6 + u.angle) * 0.3, Math.cos(a) * u.radius);
      chunk.rotation.y = elapsed * 0.2 + u.angle;
    }
    for (const cloud of clouds) {
      const u = cloud.userData;
      const a = u.angle + elapsed * u.speed;
      cloud.position.set(
        Math.sin(a) * u.radius,
        u.height + Math.sin(elapsed * 0.4 + u.angle) * 0.2,
        Math.cos(a) * u.radius,
      );
    }
    bulbs.forEach((bulb, i) => {
      bulb.material.emissiveIntensity = reduced ? 1.2 : 1.1 + Math.sin(elapsed * 2 + i) * 0.5;
    });
    kbody.scale.y = 0.72 + (reduced ? 0 : Math.sin(elapsed * 1.6) * 0.025);
    if (!reduced) {
      const pos = dustGeo.attributes.position;
      for (let i = 0; i < pos.count; i += 1) {
        let y = pos.getY(i) + delta * 0.22;
        if (y > 8.6) y = 0.8;
        pos.setY(i, y);
      }
      pos.needsUpdate = true;
    }

    // hover 检测与高亮
    const hit = pointers.size === 0 && pointerInside ? pickAt() : null;
    if (window.__libraryScene) {
      window.__libraryScene.state = {
        hit: hit?.userData?.title || hit?.userData?.name || null,
        ndc: [Number(pointerNdc.x.toFixed(3)), Number(pointerNdc.y.toFixed(3))],
        inside: pointerInside,
        pointers: pointers.size,
      };
    }
    if (hit !== hovered) {
      hovered = hit;
      canvas.style.cursor = hit ? 'pointer' : 'grab';
      if (hit) {
        const u = hit.userData;
        tooltip.textContent =
          u.type === 'book'
            ? `${u.title} · ${u.cat}`
            : u.type === 'cat'
              ? '喵～（点一下试试）'
              : `${u.name} · ${u.count} 篇 · 看书目`;
        tooltip.hidden = false;
      } else {
        tooltip.hidden = true;
      }
    }
    for (const book of pickables) {
      const u = book.userData;
      if (u.type !== 'book') continue;
      const want = book === hovered ? u.baseZ + 0.24 : u.baseZ;
      book.position.z += (want - book.position.z) * 0.2;
    }
    for (const shelf of shelves) {
      const scaleWant = hovered && shelf.children.includes(hovered) ? 1.02 : 1;
      shelf.scale.setScalar(shelf.scale.x + (scaleWant - shelf.scale.x) * 0.15);
    }

    renderer.render(scene, camera);
  }
  tick();

  function dispose() {
    cancelAnimationFrame(frame);
    ro.disconnect();
    io.disconnect();
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointerleave', onLeave);
    canvas.removeEventListener('wheel', onWheel);
    container.removeEventListener('pointermove', onAim);
    window.removeEventListener('keydown', onKey);
    window.clearTimeout(meowTimer);
    scene.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        for (const mat of Array.isArray(obj.material) ? obj.material : [obj.material]) {
          mat.map?.dispose();
          mat.dispose();
        }
      }
    });
    renderer.dispose();
    canvas.remove();
    tooltip.remove();
    labelLayer.remove();
    panel.remove();
    delete window.__libraryScene;
  }

  const toScreen = (obj) => {
    const v = new THREE.Vector3();
    obj.getWorldPosition(v);
    v.project(camera);
    const rect = container.getBoundingClientRect();
    return { x: (v.x * 0.5 + 0.5) * rect.width, y: (-v.y * 0.5 + 0.5) * rect.height };
  };
  window.__libraryScene = {
    freeze() {
      frozen = true;
    },
    project(query) {
      const obj =
        pickables.find((item) => item.userData.title === query || item.userData.name === query) ||
        shelves.find((item) => item.userData.name === query);
      return obj ? toScreen(obj) : null;
    },
    screenBooks(limit = 8) {
      const rect = container.getBoundingClientRect();
      return pickables
        .filter((item) => item.userData.type === 'book')
        .map((item) => ({ ...toScreen(item), title: item.userData.title, url: item.userData.url }))
        .filter(
          (p) =>
            p.x > rect.width * 0.08 &&
            p.x < rect.width * 0.92 &&
            p.y > rect.height * 0.08 &&
            p.y < rect.height * 0.92,
        )
        .slice(0, limit);
    },
  };

  return { container, dispose };
}

let handle = null;
let dataPromise = null;
let syncing = false;

async function sync() {
  if (syncing) return;
  syncing = true;
  try {
    const container = document.querySelector('#library-scene');
    if (!isHomeRoute()) {
      if (handle) {
        handle.dispose();
        handle = null;
      }
      return;
    }
    if (!container) return;
    if (handle && handle.container === container) return;
    if (handle) {
      handle.dispose();
      handle = null;
    }
    dataPromise = dataPromise || loadData();
    const data = await dataPromise;
    buildMeta(data);
    if (!container.isConnected || !isHomeRoute()) return;
    if (handle && handle.container === container) return;
    container.querySelectorAll('canvas, .scene-tooltip').forEach((el) => el.remove());
    handle = initScene(data, container);
  } catch (err) {
    console.warn('[library-scene] 场景初始化失败，回退到静态导航：', err);
    const container = document.querySelector('#library-scene');
    if (container && !container.querySelector('.scene-fallback')) {
      container.classList.add('scene-failed');
      const note = document.createElement('p');
      note.className = 'scene-fallback';
      note.textContent = '3D 场景暂不可用（浏览器不支持 WebGL 或索引加载失败），请用下方图例与卡片导航。';
      container.appendChild(note);
    }
  } finally {
    syncing = false;
  }
}

const mo = new MutationObserver(() => {
  void sync();
});
mo.observe(document.body, { childList: true, subtree: true });
window.addEventListener('hashchange', () => {
  void sync();
});
void sync();
