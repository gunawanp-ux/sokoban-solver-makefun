// Sokoban solver for make.fun sokoban-solver-01-run-20261121.
// Strategy: the generated levels are a hallway with independent rooms off it,
// and every box stays in its own room, so each room is solved on its own with
// a push-based A* search (minimum-cost box/goal matching heuristic, dead
// square pruning, 2x2 block deadlock pruning) and the room plans are joined
// by plain walks along the hallway. If a level does not split into rooms the
// same A* runs on the whole board as a fallback. All budgets are counted in
// expanded nodes, never in wall time, so two runs write identical bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Direction order is up, down, left, right, so d ^ 1 is the opposite.
const DY = [-1, 1, 0, 0];
const DX = [0, 0, -1, 1];
const LO = ['u', 'd', 'l', 'r'];
const UP = ['U', 'D', 'L', 'R'];
const INF = 1e9;

function parseLevel(text) {
  const rows = text.replace(/\r/g, '').split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Set();
  const goal = new Set();
  const boxes = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const ch = x < rows[y].length ? rows[y][x] : '#';
      const id = y * w + x;
      if (ch === '#') { wall.add(id); continue; }
      if (ch === '.' || ch === '*' || ch === '+') goal.add(id);
      if (ch === '$' || ch === '*') boxes.push(id);
      if (ch === '@' || ch === '+') player = id;
    }
  }
  return { w, h, wall, goal, boxes, player };
}

function buildNeigh(w, h) {
  const neigh = new Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const row = new Int32Array(4);
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d];
        const ny = y + DY[d];
        row[d] = nx >= 0 && nx < w && ny >= 0 && ny < h ? ny * w + nx : -1;
      }
      neigh[y * w + x] = row;
    }
  }
  return neigh;
}

// Flood fill of cells the player can stand on, avoiding walls and boxes.
// allowed is the set of cells the player may occupy at all in this search.
function flood(start, allowed, boxSet, neigh) {
  const seen = new Set([start]);
  const stack = [start];
  while (stack.length) {
    const c = stack.pop();
    for (let d = 0; d < 4; d++) {
      const n = neigh[c][d];
      if (n >= 0 && allowed.has(n) && !boxSet.has(n) && !seen.has(n)) {
        seen.add(n);
        stack.push(n);
      }
    }
  }
  return seen;
}

// Shortest walk (lowercase LURD) from start to target over allowed cells
// avoiding boxes, or null when unreachable.
function walkPath(start, target, allowed, boxSet, neigh) {
  if (start === target) return '';
  const prev = new Map([[start, -1]]);
  const prevDir = new Map();
  const queue = [start];
  let head = 0;
  while (head < queue.length) {
    const c = queue[head++];
    for (let d = 0; d < 4; d++) {
      const n = neigh[c][d];
      if (n < 0 || !allowed.has(n) || boxSet.has(n) || prev.has(n)) continue;
      prev.set(n, c);
      prevDir.set(n, d);
      if (n === target) {
        let out = '';
        let cur = n;
        while (prev.get(cur) !== -1) {
          out = LO[prevDir.get(cur)] + out;
          cur = prev.get(cur);
        }
        return out;
      }
      queue.push(n);
    }
  }
  return null;
}

// Squares a box can sit on and still reach some goal, found by pulling boxes
// backwards off the goals. A box anywhere else can never be solved.
function liveSquares(boxable, standable, goals, neigh) {
  const live = new Set(goals);
  const queue = [...goals];
  let head = 0;
  while (head < queue.length) {
    const c = queue[head++];
    for (let d = 0; d < 4; d++) {
      const from = neigh[c][d ^ 1];
      if (from < 0 || !boxable.has(from) || live.has(from)) continue;
      const stand = neigh[from][d ^ 1];
      if (stand < 0 || !standable.has(stand)) continue;
      live.add(from);
      queue.push(from);
    }
  }
  return live;
}

// BFS distances over boxable cells from each goal: how far a box travels.
function goalDistances(boxable, goals, neigh) {
  const maps = new Map();
  for (const g of goals) {
    const dist = new Map([[g, 0]]);
    const queue = [g];
    let head = 0;
    while (head < queue.length) {
      const c = queue[head++];
      for (let d = 0; d < 4; d++) {
        const n = neigh[c][d];
        if (n >= 0 && boxable.has(n) && !dist.has(n)) {
          dist.set(n, dist.get(c) + 1);
          queue.push(n);
        }
      }
    }
    maps.set(g, dist);
  }
  return maps;
}

// Minimum-cost matching of boxes to goals by box-travel distance, by DP over
// goal subsets. Admissible: every push moves one box one square, so solving
// needs at least this many pushes. Returns INF when no assignment works.
function matchCost(boxes, goalsArr, distMaps) {
  const n = boxes.length;
  const m = goalsArr.length;
  if (n === 0) return 0;
  if (n !== m) {
    let s = 0;
    for (const b of boxes) {
      let best = INF;
      for (const g of goalsArr) {
        const v = distMaps.get(g).get(b);
        if (v !== undefined && v < best) best = v;
      }
      if (best >= INF) return INF;
      s += best;
    }
    return s;
  }
  const cost = boxes.map((b) => goalsArr.map((g) => {
    const v = distMaps.get(g).get(b);
    return v === undefined ? INF : v;
  }));
  const size = 1 << m;
  const dp = new Array(size).fill(INF);
  dp[0] = 0;
  for (let mask = 0; mask < size; mask++) {
    let i = 0;
    let t = mask;
    while (t) { i += t & 1; t >>= 1; }
    if (i >= n || dp[mask] >= INF) continue;
    for (let j = 0; j < m; j++) {
      if (mask & (1 << j)) continue;
      const c = cost[i][j];
      if (c >= INF) continue;
      const nm = mask | (1 << j);
      if (dp[mask] + c < dp[nm]) dp[nm] = dp[mask] + c;
    }
  }
  return dp[size - 1];
}

// A 2x2 block of walls and boxes with any box off a goal can never move again.
function blockDeadlock(boxSet, wall, goals, w, h, neigh) {
  for (const b of boxSet) {
    if (goals.has(b)) continue;
    const x = b % w;
    const y = (b / w) | 0;
    for (let oy = -1; oy <= 0; oy++) {
      for (let ox = -1; ox <= 0; ox++) {
        const cx = x + ox;
        const cy = y + oy;
        if (cx < 0 || cy < 0 || cx + 1 >= w || cy + 1 >= h) continue;
        const cells = [cy * w + cx, cy * w + cx + 1, (cy + 1) * w + cx, (cy + 1) * w + cx + 1];
        let solid = true;
        let offGoal = false;
        for (const c of cells) {
          if (wall.has(c)) continue;
          if (boxSet.has(c)) { if (!goals.has(c)) offGoal = true; continue; }
          solid = false;
          break;
        }
        if (solid && offGoal) return true;
      }
    }
  }
  return false;
}

// Push-based A* over one region. standable: cells the player may occupy;
// boxable: cells a box may occupy (a subset). Returns { moves, boxes,
// player } with the full LURD string for the region, or null.
function solveRegion(w, h, wall, neigh, standable, boxable, goals, startBoxes, playerStart, cap) {
  const goalsArr = [...goals].sort((a, b) => a - b);
  const live = liveSquares(boxable, standable, goals, neigh);
  for (const b of startBoxes) if (!live.has(b)) return null;
  const distMaps = goalDistances(boxable, goalsArr, neigh);
  const h0 = matchCost(startBoxes, goalsArr, distMaps);
  if (h0 >= INF) return null;

  const nodes = [];
  const heap = [];
  const best = new Map();
  let counter = 0;
  const pushHeap = (idx) => {
    heap.push(idx);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const popHeap = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && less(heap[l], heap[m])) m = l;
        if (r < heap.length && less(heap[r], heap[m])) m = r;
        if (m === i) break;
        [heap[i], heap[m]] = [heap[m], heap[i]];
        i = m;
      }
    }
    return top;
  };
  const less = (a, b) => {
    const na = nodes[a];
    const nb = nodes[b];
    if (na.f !== nb.f) return na.f < nb.f;
    if (na.g !== nb.g) return na.g > nb.g;
    return na.seq < nb.seq;
  };

  const rootBoxes = [...startBoxes].sort((a, b) => a - b);
  nodes.push({ b: rootBoxes, p: playerStart, g: 0, f: h0, parent: -1, dir: -1, from: -1, seq: counter++ });
  pushHeap(0);
  const rootReach = flood(playerStart, standable, new Set(rootBoxes), neigh);
  best.set(rootBoxes.join(',') + '|' + Math.min(...rootReach), 0);

  let expansions = 0;
  while (heap.length) {
    const idx = popHeap();
    const node = nodes[idx];
    if (node.b.every((b) => goals.has(b))) {
      // Rebuild the move string by walking the parent chain.
      const chain = [];
      let cur = idx;
      while (nodes[cur].parent !== -1) {
        chain.push(nodes[cur]);
        cur = nodes[cur].parent;
      }
      chain.reverse();
      let moves = '';
      let player = playerStart;
      let boxes = nodes[0].b;
      for (const step of chain) {
        const stance = neigh[step.from][step.dir ^ 1];
        const walk = walkPath(player, stance, standable, new Set(boxes), neigh);
        if (walk === null) return null;
        moves += walk + UP[step.dir];
        player = step.from;
        boxes = step.b;
      }
      return { moves, boxes: [...boxes], player };
    }
    if (++expansions > cap) return null;
    const boxSet = new Set(node.b);
    const reach = flood(node.p, standable, boxSet, neigh);
    for (let bi = 0; bi < node.b.length; bi++) {
      const bc = node.b[bi];
      for (let d = 0; d < 4; d++) {
        const stance = neigh[bc][d ^ 1];
        const target = neigh[bc][d];
        if (stance < 0 || target < 0) continue;
        if (!reach.has(stance)) continue;
        if (!boxable.has(target) || boxSet.has(target)) continue;
        if (!live.has(target)) continue;
        const nb = [...node.b];
        nb[bi] = target;
        nb.sort((a, b) => a - b);
        const nSet = new Set(nb);
        if (blockDeadlock(nSet, wall, goals, w, h, neigh)) continue;
        const hv = matchCost(nb, goalsArr, distMaps);
        if (hv >= INF) continue;
        const g = node.g + 1;
        const nReach = flood(target === stance ? stance : bc, standable, nSet, neigh);
        const key = nb.join(',') + '|' + Math.min(...nReach);
        if (best.has(key) && best.get(key) <= g) continue;
        best.set(key, g);
        nodes.push({ b: nb, p: bc, g, f: g + hv, parent: idx, dir: d, from: bc, seq: counter++ });
        pushHeap(nodes.length - 1);
      }
    }
  }
  return null;
}

// Replay a move string on the level; true only when it is fully legal and
// ends with every box on a goal. Used as a safety net before answering.
function verify(level, neigh, moves) {
  const boxSet = new Set(level.boxes);
  let player = level.player;
  for (const ch of moves) {
    const d = LO.indexOf(ch.toLowerCase());
    if (d < 0) continue;
    const t = neigh[player][d];
    if (t < 0 || level.wall.has(t)) return false;
    if (boxSet.has(t)) {
      const beyond = neigh[t][d];
      if (beyond < 0 || level.wall.has(beyond) || boxSet.has(beyond)) return false;
      boxSet.delete(t);
      boxSet.add(beyond);
    }
    player = t;
  }
  for (const b of boxSet) if (!level.goal.has(b)) return false;
  return true;
}

function solveLevel(text) {
  const level = parseLevel(text);
  const { w, h, wall, goal, neigh } = { ...level, neigh: buildNeigh(level.w, level.h) };
  const floor = new Set();
  for (let id = 0; id < w * h; id++) if (!wall.has(id)) floor.add(id);
  if (level.player < 0 || level.boxes.length === 0) return '';

  // Try the room split first: hallway row out, the rest falls into rooms.
  const playerRow = (level.player / w) | 0;
  const hallway = new Set([...floor].filter((id) => ((id / w) | 0) === playerRow));
  const rest = [...floor].filter((id) => !hallway.has(id));
  const restSet = new Set(rest);
  const compOf = new Map();
  const comps = [];
  for (const c of rest) {
    if (compOf.has(c)) continue;
    const cells = [];
    const stack = [c];
    compOf.set(c, comps.length);
    while (stack.length) {
      const cur = stack.pop();
      cells.push(cur);
      for (let d = 0; d < 4; d++) {
        const n = neigh[cur][d];
        if (n >= 0 && restSet.has(n) && !compOf.has(n)) {
          compOf.set(n, comps.length);
          stack.push(n);
        }
      }
    }
    comps.push(new Set(cells));
  }

  let rooms = null;
  if (comps.length >= 1 && !level.boxes.some((b) => hallway.has(b))) {
    rooms = [];
    for (const cellSet of comps) {
      const doors = [...cellSet].filter((c) => {
        for (let d = 0; d < 4; d++) if (hallway.has(neigh[c][d])) return true;
        return false;
      });
      if (doors.length !== 1) { rooms = null; break; }
      const door = doors[0];
      let porch = -1;
      for (let d = 0; d < 4; d++) {
        const n = neigh[door][d];
        if (n >= 0 && hallway.has(n) && ((n / w) | 0) === playerRow) porch = n;
      }
      if (porch < 0) { rooms = null; break; }
      const cBoxes = level.boxes.filter((b) => cellSet.has(b));
      const cGoals = new Set([...goal].filter((g) => cellSet.has(g)));
      if (cBoxes.length === 0 || cBoxes.length !== cGoals.size) { rooms = null; break; }
      rooms.push({ cellSet, door, porch, boxes: cBoxes, goals: cGoals });
    }
  }

  if (rooms) {
    let out = '';
    let player = level.player;
    let playerComp = -1;
    const boxesNow = new Set(level.boxes);
    let failed = false;
    for (let ri = 0; ri < rooms.length; ri++) {
      const R = rooms[ri];
      const allowed = new Set(hallway);
      if (playerComp >= 0) for (const c of rooms[playerComp].cellSet) allowed.add(c);
      allowed.add(player);
      const walk = walkPath(player, R.porch, allowed, boxesNow, neigh);
      if (walk === null) { failed = true; break; }
      out += walk;
      const standable = new Set([...R.cellSet, R.porch]);
      const res = solveRegion(w, h, wall, neigh, standable, R.cellSet, R.goals, R.boxes, R.porch, 1500000);
      if (!res) { failed = true; break; }
      out += res.moves;
      for (const b of R.boxes) boxesNow.delete(b);
      for (const b of res.boxes) boxesNow.add(b);
      player = res.player;
      playerComp = ri;
    }
    if (!failed && verify(level, neigh, out)) return out;
  }

  // Fallback: one A* over the whole board.
  const res = solveRegion(w, h, wall, neigh, floor, floor, goal, level.boxes, level.player, 4000000);
  if (res && verify(level, neigh, res.moves)) return res.moves;
  return '';
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
