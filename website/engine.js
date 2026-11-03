/* NeuroType browser engine: simulation, classifiers, held-out evaluation, metrics. */
const NT = (() => {
  const CHANNELS = ['C3','C1','Cz','C2','C4','CP3','CPz','CP4'];
  const LANES = ['C3','Cz','C4'];
  const FS = 160, N = 480, T0 = 0.5, RUNS = [4,8,12], SUBJECTS = [1,2,3], PER_RUN = 15;
  const THRESHOLDS = [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95];
  const MODEL_NAMES = { csp:'CSP + logistic regression', power:'C3/C4 power + logistic regression', eegnet:'EEGNet' };

  function rng(seed){ return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const gauss = r => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());

  function resonator(r, f, n){ // narrow-band oscillation, used for the 10 Hz mu rhythm
    const rho = .985, w = 2 * Math.PI * f / FS, a = 2 * rho * Math.cos(w), b = -rho * rho, out = new Float64Array(n);
    let x1 = 0, x2 = 0, sum = 0;
    for (let i = -120; i < n; i++){ const x = a * x1 + b * x2 + gauss(r); x2 = x1; x1 = x; if (i >= 0){ out[i] = x; sum += x * x; } }
    const s = Math.sqrt(sum / n) || 1; for (let i = 0; i < n; i++) out[i] /= s; return out;
  }
  function drift(r, n){ // broad-band background
    const out = new Float64Array(n); let x = 0, sum = 0;
    for (let i = 0; i < n; i++){ x = .75 * x + gauss(r); out[i] = x; sum += x * x; }
    const s = Math.sqrt(sum / n) || 1; for (let i = 0; i < n; i++) out[i] /= s; return out;
  }

  /* Simulated contralateral mu suppression. Same seed, so moving clarity changes only the effect size. */
  function simulate(clarity){
    const r = rng(2026), ML = [1,.7,.3,.1,0,.8,.25,.05], MR = [0,.1,.3,.7,1,.05,.25,.8], trials = [];
    for (const s of SUBJECTS){
      const gains = CHANNELS.map(() => .75 + .5 * r()), f = 9.5 + 2.5 * r();
      for (const run of RUNS){
        const runGain = .85 + .3 * r();
        const labels = Array.from({length:PER_RUN}, (_,i) => i % 2);
        for (let i = labels.length - 1; i > 0; i--){ const j = Math.floor(r() * (i + 1)); [labels[i], labels[j]] = [labels[j], labels[i]]; }
        for (const y of labels){
          const L = resonator(r, f, N), R = resonator(r, f + .3, N), common = drift(r, N);
          const strength = Math.max(0, 1 + .75 * gauss(r)) * clarity; // attention wanders, so some trials carry little signal
          const env = t => Math.max(.05, 1 - .85 * strength * Math.exp(-.5 * ((t - 1.5) / .8) ** 2));
          const x = CHANNELS.map((_, c) => {
            const noise = drift(r, N), a = new Float32Array(N);
            for (let t = 0; t < N; t++){ const e = env(t / FS);
              a[t] = gains[c] * runGain * (ML[c] * L[t] * (y === 1 ? e : 1) + MR[c] * R[t] * (y === 0 ? e : 1) + .62 * noise[t] + .4 * common[t]) * 8; }
            return a; });
          trials.push({ s, r: run, y, x, lanes: LANES.map(n => x[CHANNELS.indexOf(n)]), cov: covariance(x) });
        }
      }
    }
    return { source:'synthetic', channels: CHANNELS, laneNames: LANES, lanesFs: FS, trials, models: ['csp','power'] };
  }

  function covariance(x){ const n = x.length, C = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = i; j < n; j++){ let s = 0; const a = x[i], b = x[j]; for (let t = 0; t < a.length; t++) s += a[t] * b[t]; C[i*n+j] = C[j*n+i] = s / a.length; }
    return C; }

  function eigSym(Ain, n){ // cyclic Jacobi
    const A = Float64Array.from(Ain), V = new Float64Array(n * n); for (let i = 0; i < n; i++) V[i*n+i] = 1;
    for (let sweep = 0; sweep < 60; sweep++){
      let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i*n+j] ** 2;
      if (off < 1e-20) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++){
        const apq = A[p*n+q]; if (Math.abs(apq) < 1e-15) continue;
        const th = (A[q*n+q] - A[p*n+p]) / (2 * apq), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++){ const akp = A[k*n+p], akq = A[k*n+q]; A[k*n+p] = c * akp - s * akq; A[k*n+q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++){ const apk = A[p*n+k], aqk = A[q*n+k]; A[p*n+k] = c * apk - s * aqk; A[q*n+k] = s * apk + c * aqk; }
        for (let k = 0; k < n; k++){ const vkp = V[k*n+p], vkq = V[k*n+q]; V[k*n+p] = c * vkp - s * vkq; V[k*n+q] = s * vkp + c * vkq; }
      }
    }
    return { vals: Array.from({length:n}, (_, i) => A[i*n+i]), vecs: V };
  }

  function cspFilters(train, n){
    const C = [new Float64Array(n*n), new Float64Array(n*n)], count = [0,0];
    for (const t of train){ let tr = 0; for (let i = 0; i < n; i++) tr += t.cov[i*n+i];
      for (let i = 0; i < n*n; i++) C[t.y][i] += t.cov[i] / tr; count[t.y]++; }
    for (const k of [0,1]) for (let i = 0; i < n*n; i++) C[k][i] /= count[k];
    const S = new Float64Array(n*n); for (let i = 0; i < n*n; i++) S[i] = C[0][i] + C[1][i];
    const mean = (S.reduce((a, _, i) => a + (i % (n + 1) === 0 ? S[i] : 0), 0)) / n;
    for (let i = 0; i < n; i++) S[i*n+i] += .02 * mean; // shrinkage, as few trials per fold
    const { vals, vecs } = eigSym(S, n);
    const P = new Float64Array(n*n); // whitening matrix
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) P[i*n+j] = vecs[j*n+i] / Math.sqrt(Math.max(vals[i], 1e-12));
    const mul = (A, B) => { const M = new Float64Array(n*n); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++){ let s = 0; for (let k = 0; k < n; k++) s += A[i*n+k] * B[k*n+j]; M[i*n+j] = s; } return M; };
    const tr = M => { const O = new Float64Array(n*n); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) O[i*n+j] = M[j*n+i]; return O; };
    const W = mul(mul(P, C[0]), tr(P)), eg = eigSym(W, n);
    const order = eg.vals.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]).map(p => p[1]);
    const picks = [order[0], order[1], order[n-1], order[n-2]];
    return picks.map(c => { const row = new Float64Array(n); for (let j = 0; j < n; j++){ let s = 0; for (let k = 0; k < n; k++) s += eg.vecs[k*n+c] * P[k*n+j]; row[j] = s; } return row; });
  }

  function features(F, cov, n){ return F.map(w => { let s = 0; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) s += w[i] * cov[i*n+j] * w[j]; return Math.log(Math.max(s, 1e-12)); }); }

  function fit(train, kind, channels){
    const n = channels.length;
    const F = kind === 'csp' ? cspFilters(train, n) : ['C3','C4'].map(c => { const w = new Float64Array(n); w[channels.indexOf(c)] = 1; return w; });
    const X = train.map(t => features(F, t.cov, n)), d = F.length, m = X.length;
    const mu = Array(d).fill(0), sd = Array(d).fill(0);
    for (const row of X) row.forEach((v, j) => mu[j] += v / m);
    for (const row of X) row.forEach((v, j) => sd[j] += (v - mu[j]) ** 2 / m);
    sd.forEach((v, j) => sd[j] = Math.sqrt(v) || 1);
    const Z = X.map(row => row.map((v, j) => (v - mu[j]) / sd[j]));
    const beta = Array(d).fill(0); let b = 0;
    for (let it = 0; it < 500; it++){ // L2-regularised logistic regression
      const g = Array(d).fill(0); let gb = 0;
      Z.forEach((z, i) => { const p = 1 / (1 + Math.exp(-(b + z.reduce((a, v, j) => a + v * beta[j], 0)))), e = p - train[i].y; gb += e; z.forEach((v, j) => g[j] += e * v); });
      for (let j = 0; j < d; j++) beta[j] -= .4 * (g[j] / m + beta[j] / m);
      b -= .4 * gb / m;
    }
    return { F, mu, sd, beta, b, n };
  }
  function prob(model, t){ const z = features(model.F, t.cov, model.n).map((v, j) => (v - model.mu[j]) / model.sd[j]);
    return 1 / (1 + Math.exp(-(model.b + z.reduce((a, v, j) => a + v * model.beta[j], 0)))); }

  /* Exact gradient of the score with respect to each channel and sample, in absolute value. */
  function attribution(model, trial, bins = 30){
    const n = model.n, T = trial.x[0].length, out = Array.from({length:n}, () => new Float64Array(bins));
    model.F.forEach((w, k) => {
      const src = new Float64Array(T); let pow = 0;
      for (let t = 0; t < T; t++){ let s = 0; for (let c = 0; c < n; c++) s += w[c] * trial.x[c][t]; src[t] = s; pow += s * s / T; }
      const weight = model.beta[k] / model.sd[k] / Math.max(pow, 1e-12) * 2 / T;
      for (let c = 0; c < n; c++) for (let t = 0; t < T; t++) out[c][Math.min(bins - 1, Math.floor(t / T * bins))] += weight * w[c] * src[t];
    });
    let max = 1e-12; const abs = out.map(row => Array.from(row, v => { const a = Math.abs(v); max = Math.max(max, a); return a; }));
    return abs.map(row => row.map(v => v / max));
  }

  /* Whole runs or whole participants are held out. Random trial splits never happen. */
  function foldsFor(trials){
    const subjects = [...new Set(trials.map(t => t.s))], within = [], loso = [];
    for (const s of subjects){
      const mine = trials.filter(t => t.s === s);
      for (const r of [...new Set(mine.map(t => t.r))]) within.push({ s, r, test: mine.filter(t => t.r === r), train: mine.filter(t => t.r !== r) });
      if (subjects.length > 1) loso.push({ s, test: trials.filter(t => t.s === s), train: trials.filter(t => t.s !== s) });
    }
    return { within, loso };
  }

  function evaluate(data){
    const { within, loso } = foldsFor(data.trials), index = new Map(data.trials.map((t, i) => [t, i]));
    const out = { p:{}, trainScore:{}, fitted:{} };
    for (const kind of data.models){
      out.p[kind] = {}; out.trainScore[kind] = {};
      for (const [name, folds] of [['within', within], ['loso', loso]]){
        const p = new Float64Array(data.trials.length).fill(NaN), scores = [];
        for (const f of folds){
          if (f.train.some(t => f.test.includes(t))) throw new Error('Train and test overlap');
          const model = fit(f.train, kind, data.channels);
          f.test.forEach(t => p[index.get(t)] = prob(model, t));
          scores.push(summary(f.train.map(t => t.y), f.train.map(t => prob(model, t))).balanced);
          if (name === 'within') out.fitted[`${kind}_${f.s}_${f.r}`] = model;
        }
        out.p[kind][name] = p; out.trainScore[kind][name] = scores.reduce((a, b) => a + b, 0) / (scores.length || 1);
      }
    }
    return out;
  }

  function wilson(k, n, z = 1.96){ if (!n) return [0,0]; const p = k / n, d = 1 + z*z/n, c = (p + z*z/(2*n)) / d, h = z * Math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d; return [c - h, c + h]; }

  function summary(y, p){
    const cm = [[0,0],[0,0]]; let right = 0;
    y.forEach((t, i) => { const q = p[i] >= .5 ? 1 : 0; cm[t][q]++; if (q === t) right++; });
    const cls = k => { const tp = cm[k][k], fp = cm[1-k][k], fn = cm[k][1-k], prec = tp + fp ? tp / (tp + fp) : 0, rec = tp + fn ? tp / (tp + fn) : 0;
      return { precision: prec, recall: rec, f1: prec + rec ? 2 * prec * rec / (prec + rec) : 0, support: tp + fn }; };
    const perClass = [cls(0), cls(1)], conf = p.map(v => Math.max(v, 1 - v)), ok = y.map((t, i) => (p[i] >= .5 ? 1 : 0) === t);
    let ece = 0; const edges = [.5,.6,.7,.8,.9,1];
    for (let b = 0; b < 5; b++){ const idx = conf.map((c, i) => (c >= edges[b] && (b === 4 ? c <= 1 : c < edges[b+1])) ? i : -1).filter(i => i >= 0);
      if (idx.length){ const mc = idx.reduce((a, i) => a + conf[i], 0) / idx.length, ma = idx.reduce((a, i) => a + ok[i], 0) / idx.length; ece += idx.length / y.length * Math.abs(mc - ma); } }
    const selective = THRESHOLDS.map(t => { const keep = conf.map((c, i) => c >= t ? i : -1).filter(i => i >= 0);
      return { threshold: t, coverage: keep.length / y.length, accuracy: keep.length ? keep.reduce((a, i) => a + ok[i], 0) / keep.length : null, n: keep.length }; });
    return { n: y.length, accuracy: right / y.length, ci: wilson(right, y.length), balanced: (perClass[0].recall + perClass[1].recall) / 2,
      macroF1: (perClass[0].f1 + perClass[1].f1) / 2, cm, perClass, ece, selective };
  }

  return { simulate, evaluate, summary, attribution, foldsFor, MODEL_NAMES, THRESHOLDS, CHANNELS, T0 };
})();
if (typeof module !== 'undefined') module.exports = NT;
