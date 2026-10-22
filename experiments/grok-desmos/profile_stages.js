// Stage profiler for a gen_grok.py graph, run from the browser console on desmos.com/calculator.
//
//   const STATE = <contents of out/grok32_train.json>;  then paste this file, then:
//   await psRun(STATE)          // returns one row per stage; window must be in front
//
// For each stage it loads the graph cut off after that expression, adds a probe that depends
// on the stage's output, and records the first evaluation time and two steady-state updates.
// Updates are driven by a slider s_w mixed into W_E (value-neutral scale 1 + 1e-9 s_w), so every
// stage recomputes from the weights each time, like a training step does.
// Timing is event driven (Calc.observe), and the frame rate is checked first: in a background
// tab Desmos delivers results only on animation frames (about 1 per second), which floors
// every measurement at about a second.

const psSleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function psFrameRate(ms = 600) {
  let frames = 0;
  const t0 = performance.now();
  await new Promise((resolve) => {
    const step = () => { frames += 1; if (performance.now() - t0 < ms) requestAnimationFrame(step); else resolve(); };
    requestAnimationFrame(step);
  });
  return frames / ((performance.now() - t0) / 1000);
}

function psWhenChanged(probeId, before, timeoutMs) {
  return new Promise((resolve) => {
    const timer = setTimeout(() => { Calc.unobserve('expressionAnalysis.ps'); resolve(-1); }, timeoutMs);
    Calc.observe('expressionAnalysis.ps', () => {
      const a = Calc.expressionAnalysis[probeId];
      const now = JSON.stringify(a?.evaluation?.value);
      if (a?.isError || (now !== undefined && now !== before)) {
        clearTimeout(timer);
        Calc.unobserve('expressionAnalysis.ps');
        resolve(a?.isError ? -2 : 1);
      }
    });
  });
}

// stage id -> probe latex (must change when the weights change, and depend on every block)
const PS_STAGES = [
  ['Bo', 'W_{E}\\left[2;2\\right]+B_{o}\\left[1;2\\right]'],
  ['A3', '\\operatorname{total}\\left(A_{1}\\right)\\left[2;2\\right]'],
  ['Za', '\\operatorname{total}\\left(Z_{a}\\right)\\left[2;2\\right]'],
  ['Rb', '\\operatorname{total}\\left(R_{b}\\right)\\left[2;2\\right]'],
  ['Gl', '\\operatorname{total}\\left(G_{l}\\right)\\left[2;2\\right]'],
  ['Pb', '\\operatorname{total}\\left(E_{x}\\right)\\left[2;2\\right]'],
  ['Cg', '\\operatorname{total}\\left(E_{x}\\right)\\left[2;2\\right]+C_{g}\\left[1;1\\right]'],
  ['Dl', '\\operatorname{total}\\left(D_{l}\\right)\\left[2;2\\right]'],
  ['Du', '\\operatorname{total}\\left(D_{u}\\right)\\left[2;2\\right]'],
  ['Dz', '\\operatorname{total}\\left(D_{z}\\right)\\left[2;2\\right]'],
  ['Y3', '\\operatorname{total}\\left(Y_{3}\\right)\\left[2;2\\right]'],
  ['Gp', 'G_{p}\\left[2;2\\right]+G_{e}\\left[2;2\\right]'],
  ['gu', '\\operatorname{total}\\left(g_{e}\\right)+\\operatorname{total}\\left(g_{u}\\right)'],
  ['As', '\\operatorname{total}\\left(w_{ne}\\right)+\\operatorname{total}\\left(w_{ny}\\right)'],
];

async function psOne(list, probe, loadTimeoutMs, updateTimeoutMs) {
  Calc.setBlank();
  await psSleep(400);
  const withProbe = list.concat([{ type: 'expression', id: 'pz', latex: 'p_{z}=' + probe }]);
  const t0 = performance.now();
  const loaded = psWhenChanged('pz', undefined, loadTimeoutMs);
  Calc.setState({ version: 11, graph: { viewport: { xmin: -10, xmax: 10, ymin: -10, ymax: 10 } }, expressions: { list: withProbe } });
  const ok = await loaded;
  const load = ok === 1 ? Math.round(performance.now() - t0) : (ok === -2 ? 'error' : 'timeout');
  const updates = [];
  if (ok === 1) {
    for (let k = 1; k <= 2; k++) {
      const before = JSON.stringify(Calc.expressionAnalysis.pz?.evaluation?.value);
      const t1 = performance.now();
      const changed = psWhenChanged('pz', before, updateTimeoutMs);
      Calc.setExpression({ id: 'sw', latex: 's_{w}=' + k });
      const r = await changed;
      updates.push(r === 1 ? Math.round(performance.now() - t1) : 'timeout');
    }
  }
  return { load, updates };
}

async function psRun(state, { stages = PS_STAGES, loadTimeoutMs = 240000, updateTimeoutMs = 120000, minFps = 30 } = {}) {
  const fps = await psFrameRate();
  if (fps < minFps) throw new Error(`profile_stages: ${fps.toFixed(1)} frames/s; bring the window to the front`);
  Calc.updateSettings({ expressionsCollapsed: true });
  const all = state.expressions.list;
  const ids = all.map((e) => e.id);
  const slider = { type: 'expression', id: 'sw', latex: 's_{w}=0' };
  const rows = [];
  for (const [last, probe] of stages) {
    const cut = ids.indexOf(last);
    if (cut < 0) throw new Error(`profile_stages: no expression with id ${last}`);
    const part = [slider].concat(all.slice(0, cut + 1).map((e) => (e.id === 'WE'
      ? { ...e, latex: 'W_{E}=m_{r}\\left(w_{e},32,32\\right)\\left(1+10^{-9}s_{w}\\right)' } : e)));
    const r = await psOne(part, probe, loadTimeoutMs, updateTimeoutMs);
    rows.push({ stage: last, ...r, fps: Math.round(await psFrameRate(300)) });
    if (r.load === 'timeout') break;
  }
  Calc.setBlank();
  return rows;
}
