// Generation-speed benchmark for a gen_desmos.py graph, run from the browser console on
// desmos.com/calculator with the window in front:
//
//   const STATE = <contents of a gen_desmos.py graph>;  then paste this file, then:
//   await gbRun(STATE)
//
// Returns the first-evaluation time, the latency of single forward passes (one character
// appended to T_x, timed until the logits y_l change), the generation rate with the ticker
// running, and the logits at load so two graphs can be compared value for value.
// Timing is event driven (HelperExpression.observe). In a background tab Desmos delivers
// results only on animation frames (about 1 per second), so the frame rate is checked first.

const gbSleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function gbFrameRate(ms = 600) {
  let frames = 0;
  const t0 = performance.now();
  await new Promise((resolve) => {
    const step = () => { frames += 1; if (performance.now() - t0 < ms) requestAnimationFrame(step); else resolve(); };
    requestAnimationFrame(step);
  });
  return frames / ((performance.now() - t0) / 1000);
}

// resolves with the new value once `prop` of a helper differs from `before`, or rejects on timeout
function gbWhenChanged(helper, prop, before, timeoutMs) {
  return new Promise((resolve, reject) => {
    const check = () => {
      const now = JSON.stringify(helper[prop]);
      if (helper[prop] !== undefined && now !== before && !(Array.isArray(helper[prop]) && helper[prop].length === 0)) {
        clearTimeout(timer);
        helper.unobserve(prop);
        resolve(helper[prop]);
      }
    };
    const timer = setTimeout(() => { helper.unobserve(prop); reject(new Error(`genbench: ${prop} unchanged after ${timeoutMs} ms`)); }, timeoutMs);
    helper.observe(prop, check);
    check();
  });
}

function gbTokens(state) {
  const tx = state.expressions.list.find((e) => e.id === 'Tx');
  if (!tx) throw new Error('genbench: graph has no T_x expression');
  return tx.latex.slice('T_{x}=\\left['.length, -'\\right]'.length).split(',').map(Number);
}

async function gbRun(state, { steps = 6, tickSeconds = 20, loadTimeoutMs = 240000, stepTimeoutMs = 60000, minFps = 30 } = {}) {
  const fps = await gbFrameRate();
  if (fps < minFps) throw new Error(`genbench: ${fps.toFixed(1)} frames/s; bring the window to the front`);
  Calc.setBlank();
  await gbSleep(400);
  Calc.updateSettings({ expressionsCollapsed: true });
  const t0 = performance.now();
  Calc.setState(state);
  const logits = Calc.HelperExpression({ latex: 'y_{l}' });   // after setState, or it is dropped
  const length = Calc.HelperExpression({ latex: 'n_{x}' });
  const atLoad = await gbWhenChanged(logits, 'listValue', undefined, loadTimeoutMs);
  const firstEval = Math.round(performance.now() - t0);

  const tokens = gbTokens(state);
  const latencies = [];
  for (let k = 0; k < steps; k++) {
    const before = JSON.stringify(logits.listValue);
    tokens.push(2 + (k % 26));                                 // letters a..z
    const t1 = performance.now();
    const changed = gbWhenChanged(logits, 'listValue', before, stepTimeoutMs);
    Calc.setExpression({ id: 'Tx', latex: `T_{x}=\\left[${tokens.join(',')}\\right]` });
    await changed;
    latencies.push(Math.round(performance.now() - t1));
  }

  const n0 = length.numericValue;
  const t2 = performance.now();
  Calc.setExpression({ id: 'go', latex: 'g_{o}=1' });
  await gbSleep(tickSeconds * 1000);
  Calc.setExpression({ id: 'go', latex: 'g_{o}=0' });
  const generated = length.numericValue - n0;
  const charsPerSecond = generated / ((performance.now() - t2) / 1000);

  const sorted = [...latencies].sort((a, b) => a - b);
  return {
    fps: Math.round(fps), firstEval, latencies, medianLatency: sorted[Math.floor(sorted.length / 2)],
    generated, charsPerSecond: Number(charsPerSecond.toFixed(3)), logitsAtLoad: atLoad,
  };
}
