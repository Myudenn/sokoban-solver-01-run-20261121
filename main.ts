// Sokoban solver: push-optimal, room by room.
//
// Idea (from the bounty brief itself): every box stays in its room — a box
// pushed through a door into the hallway can never be pushed back — so a
// level is a set of small independent rooms joined only by the player's walk,
// and the level optimum is the sum of the rooms' optima. Each room (3-4
// boxes on a carved 6x6 slot) is solved with A* over pushes: a state is the
// boxes' squares plus the player's reachable region, each push costs 1, and
// the estimate is the cheapest assignment of boxes to goals by walking
// distance (admissible: one push moves one box one square). Dead squares
// (reverse-pull reachability from the goals) and frozen 2x2 blocks are
// pruned; both prune only positions no push can save. When the room split
// does not apply, the same search runs on the whole level, and a move-based
// A* is the last resort. Everything is budgeted by work counted (state
// expansions), never by the clock, so two runs write the same bytes.
//
// Runs as `node /bot/main.ts` on Node 22 (erasable TypeScript only).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// [dx, dy, walk letter, push letter], fixed order everywhere: deterministic.
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

// Whole-run work budget, counted in state expansions (not time): the run
// always finishes inside the wall budget because work, not the clock,
// decides when a search gives up.
const WORK_BUDGET = 1_500_000;
let workSpent = 0;

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outer frame is wall whatever it shows: nothing leaves the board,
  // and neighbour arithmetic never runs off the array.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares from which a lone box can be pulled back onto some goal: start
// at the goals and pull backwards over open floor. A box off a live square
// can never reach a goal, so any state holding one is dead.
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Walking distance from `from` over open floor, ignoring boxes.
function walkDist(level: Level, from: number): Int32Array {
  const { w, wall } = level;
  const dist = new Int32Array(wall.length).fill(-1);
  const queue = [from];
  dist[from] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (!wall[q] && dist[q] < 0) {
        dist[q] = dist[p]! + 1;
        queue.push(q);
      }
    }
  }
  return dist;
}

// The squares the player reaches from `from` with `occ` in the way; marks
// them in `seen` with `stamp` and returns the lowest square reached (the
// region's name: two states with the same boxes and the same lowest
// reachable square offer the same pushes).
function regionLow(
  level: Level,
  occ: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occ[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// Whether the box just pushed to `t` sits in a frozen 2x2 block of walls
// and boxes holding a box off its goal: no push can ever move it again.
function frozenBlock(level: Level, occ: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goal[a]) || (occ[b] && !goal[b]) || (occ[c] && !goal[c]) || (occ[d] && !goal[d])) return true;
  }
  return false;
}

// Cheapest assignment of boxes to goals by walking distance (brute-force
// permutations; rooms hold at most a handful of boxes), or the sum of each
// box's nearest goal when there are too many to permute. Infinity when no
// assignment exists: the state is dead. Admissible either way.
function estimate(boxes: number[], dists: Int32Array[]): number {
  const m = boxes.length;
  if (m === 0) return 0;
  if (m > 5) {
    let s = 0;
    for (const b of boxes) {
      let best = Infinity;
      for (const d of dists) if (d[b] >= 0 && d[b] < best) best = d[b];
      if (best === Infinity) return Infinity;
      s += best;
    }
    return s;
  }
  const perm = dists.map((_, i) => i);
  let best = Infinity;
  const rec = (k: number, sum: number): void => {
    if (sum >= best) return;
    if (k === m) {
      best = sum;
      return;
    }
    for (let i = k; i < m; i++) {
      const tmp = perm[k]!;
      perm[k] = perm[i]!;
      perm[i] = tmp;
      const d = dists[perm[k]!]![boxes[k]!]!;
      if (d >= 0) rec(k + 1, sum + d);
      const tmp2 = perm[k]!;
      perm[k] = perm[i]!;
      perm[i] = tmp2;
    }
  };
  rec(0, 0);
  return best;
}

// Shortest walk from `from` to `to` around walls and boxes, lower-case
// letters, first direction first on ties. Throws when unreachable (the
// caller only asks for reachable squares).
function walkTo(level: Level, occ: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how: string[] = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occ[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: target out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

interface Solution {
  pushes: number;
  moves: string;
}

// A* (or plain BFS when useHeuristic is false) over pushes. A state is the
// boxes' squares plus the player's region; expanding a state tries every
// push the player can reach. Returns the push-optimal solution within
// maxExpand expansions, or null when the budget runs out or no solution
// exists under it.
function pushSearch(level: Level, maxExpand: number, useHeuristic: boolean): Solution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  if (level.boxes.some((b) => !live[b])) return null;
  const dists = goals.map((g) => walkDist(level, g));
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);
  const keyOf = (bs: number[], low: number): string => bs.join(',') + '|' + low;

  const occ = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const sBoxes: number[][] = [level.boxes.slice()];
  const sPlayer: number[] = [level.player];
  const sG: number[] = [0];
  const parent: number[] = [-1];
  const pFrom: number[] = [-1];
  const pTo: number[] = [-1];
  const pDir: number[] = [-1];
  const best = new Map<string, number>();
  for (const b of level.boxes) occ[b] = 1;
  best.set(keyOf(level.boxes, regionLow(level, occ, level.player, seen, ++stamp, stack)), 0);
  for (const b of level.boxes) occ[b] = 0;

  // Binary heap on [f, -g, seq]: lowest f first, deepest g on ties, then
  // insertion order — fully deterministic.
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const heapLess = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  const h0 = useHeuristic ? estimate(level.boxes, dists) : 0;
  if (h0 === Infinity) return null;
  if (solved(level.boxes)) return { pushes: 0, moves: '' };
  heapPush([h0, 0, 0, seq++]);

  let expanded = 0;
  let found = -1;
  while (heap.length > 0) {
    const [, negG, s] = heapPop();
    const g = -negG;
    if (sG[s] !== g) continue; // superseded entry
    const boxes = sBoxes[s]!;
    if (solved(boxes)) {
      found = s;
      break;
    }
    if (expanded >= maxExpand || workSpent >= WORK_BUDGET) return null;
    expanded++;
    workSpent++;
    for (const b of boxes) occ[b] = 1;
    regionLow(level, occ, sPlayer[s]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di]!;
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occ[t] || !live[t]) continue;
        occ[b] = 0;
        occ[t] = 1;
        const frozen = frozenBlock(level, occ, goal, t);
        let next: number[] | null = null;
        let key = '';
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          key = keyOf(next, regionLow(level, occ, b, mark, ++stamp, stack));
        }
        occ[t] = 0;
        occ[b] = 1;
        if (frozen || next === null) continue;
        const prev = best.get(key);
        if (prev !== undefined && sG[prev]! <= g + 1) continue;
        const h = useHeuristic ? estimate(next, dists) : 0;
        if (h === Infinity) continue;
        const idx = sBoxes.length;
        best.set(key, idx);
        sBoxes.push(next);
        sPlayer.push(b);
        sG.push(g + 1);
        parent.push(s);
        pFrom.push(b);
        pTo.push(t);
        pDir.push(di);
        if (solved(next)) {
          found = idx;
          break;
        }
        heapPush([g + 1 + h, -(g + 1), idx, seq++]);
      }
    }
    for (const b of boxes) occ[b] = 0;
  }
  if (found < 0) return null;

  // Rebuild the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of path) {
    const nextBoxes = sBoxes[s]!;
    const from = pFrom[s]!;
    const to = pTo[s]!;
    const d = to - from;
    for (const b of boxes) occ[b] = 1;
    moves += walkTo(level, occ, player, from - d);
    for (const b of boxes) occ[b] = 0;
    moves += DIRS[pDir[s]!]![3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

// Move-based A* (every step costs 1) as a last resort: it finds fewest
// moves, not fewest pushes, but a valid solution beats an empty one.
function moveSearch(level: Level, maxExpand: number): Solution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const dists = goals.map((g) => walkDist(level, g));
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);
  if (solved(level.boxes)) return { pushes: 0, moves: '' };
  const sPlayer: number[] = [level.player];
  const sBoxes: number[][] = [level.boxes.slice()];
  const sG: number[] = [0];
  const parent: number[] = [-1];
  const step: string[] = [''];
  const best = new Map<string, number>();
  const keyOf = (p: number, bs: number[]): string => p + '|' + bs.join(',');
  best.set(keyOf(level.player, level.boxes), 0);
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const heapLess = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };
  const est = (bs: number[]): number => {
    let s = 0;
    for (const b of bs) {
      let m = Infinity;
      for (const d of dists) if (d[b] >= 0 && d[b] < m) m = d[b];
      if (m === Infinity) return Infinity;
      s += m;
    }
    return s;
  };
  const h0 = est(level.boxes);
  if (h0 === Infinity) return null;
  heapPush([h0, 0, 0, seq++]);
  let expanded = 0;
  let found = -1;
  while (heap.length > 0) {
    const [, negG, s] = heapPop();
    const g = -negG;
    if (sG[s] !== g) continue;
    const boxes = sBoxes[s]!;
    if (solved(boxes)) {
      found = s;
      break;
    }
    if (expanded >= maxExpand || workSpent >= WORK_BUDGET) return null;
    expanded++;
    workSpent++;
    const p = sPlayer[s]!;
    for (const [dx, dy, letter] of DIRS) {
      const d = dy * w + dx;
      const q = p + d;
      if (wall[q]) continue;
      let nb = boxes;
      let mv = letter;
      const at = boxes.indexOf(q);
      if (at >= 0) {
        if (wall[q + d] || boxes.includes(q + d)) continue;
        nb = boxes.slice();
        nb[at] = q + d;
        nb.sort((a, b) => a - b);
        mv = letter.toUpperCase();
      }
      const k = keyOf(q, nb);
      const prev = best.get(k);
      if (prev !== undefined && sG[prev]! <= g + 1) continue;
      const h = est(nb);
      if (h === Infinity) continue;
      const idx = sPlayer.length;
      best.set(k, idx);
      sPlayer.push(q);
      sBoxes.push(nb);
      sG.push(g + 1);
      parent.push(s);
      step.push(mv);
      heapPush([g + 1 + h, -(g + 1), idx, seq++]);
    }
  }
  if (found < 0) return null;
  let moves = '';
  let pushes = 0;
  for (let s = found; s > 0; s = parent[s]!) {
    moves = step[s]! + moves;
    if (step[s]! >= 'A' && step[s]! <= 'Z') pushes++;
  }
  return { pushes, moves };
}

interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}

// Split the level into rooms: cut the hallway out of the floor — the
// player's row holds only dead squares (a box there reaches no goal) — and
// each remaining part holding boxes is a room. Null when the level is no
// such set of rooms (a box starts in the hallway, a goal is cut off, or a
// part's boxes and goals do not match).
function splitRooms(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// The square the player stands on after playing `moves` from `from`.
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

function solveLevel(level: Level): string {
  try {
    const rooms = splitRooms(level);
    if (rooms !== null) {
      let player = level.player;
      let moves = '';
      let ok = true;
      for (const room of rooms) {
        // Other rooms' boxes stand as walls (on their goals once done).
        const wall = level.wall.slice();
        for (const other of rooms) {
          if (other === room) continue;
          for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
        }
        const sub: Level = {
          w: level.w,
          h: level.h,
          wall,
          goals: room.goals.slice().sort((a, b) => a - b),
          boxes: room.boxes.slice().sort((a, b) => a - b),
          player,
        };
        const solved = pushSearch(sub, 120_000, true) ?? pushSearch(sub, 250_000, false);
        if (solved === null) {
          ok = false;
          break;
        }
        moves += solved.moves;
        player = endOf(level.w, player, solved.moves);
        room.done = true;
      }
      if (ok) return moves;
    }
    // Fallback: the whole level at once, then fewest-moves search.
    const whole =
      pushSearch(level, 150_000, true) ?? pushSearch(level, 100_000, false) ?? moveSearch(level, 100_000);
    return whole === null ? '' : whole.moves;
  } catch {
    return '';
  }
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8')));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
