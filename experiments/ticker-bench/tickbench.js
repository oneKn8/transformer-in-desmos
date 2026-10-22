// Ticker throughput vs the size of the list an action rewrites each tick, from the browser
// console on https://www.desmos.com/calculator.
//
//   paste this file, then: await tbRun()                       // list collapsed, default sizes
//                          await tbRun([1024], { collapsed: false })
//
// Each run loads a fresh graph with a counter c_t and a list l of n numbers, sets the ticker to
// rewrite both as fast as it can (minStep 0) for a few seconds, and reports ticks per second.
//
// The window must be in front: Chrome throttles requestAnimationFrame (and with it the Desmos
// ticker) for covered windows even while document.visibilityState still says "visible", so the
// run waits until the frame rate is high enough. The action target must be a plain literal or
// range; with a computed definition such as l=0.001[1...n] the action silently never fires.

const tbSleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const TB_VIEWPORT = { xmin: -10, xmax: 10, ymin: -10, ymax: 10 };

async function tbFrameRate(ms = 1000) {
  let frames = 0;
  const t0 = performance.now();
  await new Promise((resolve) => {
    const step = () => {
      frames += 1;
      if (performance.now() - t0 < ms) requestAnimationFrame(step);
      else resolve();
    };
    requestAnimationFrame(step);
  });
  return frames / ((performance.now() - t0) / 1000);
}

async function tbWaitForFront(minFps, timeoutMs) {
  const t0 = performance.now();
  for (;;) {
    const fps = await tbFrameRate(1000);
    if (fps >= minFps) return fps;
    if (performance.now() - t0 > timeoutMs) {
      throw new Error(`tickbench: still ${fps.toFixed(1)} frames/s after ${timeoutMs} ms, bring the Chrome window to the front`);
    }
  }
}

function tbSetTicker(playing) {
  const state = Calc.getState();
  state.expressions.ticker = { handlerLatex: 'A_{t}', minStepLatex: '0', open: true, playing };
  Calc.setState(state);
}

async function tbOne(n, runMs) {
  Calc.setState({
    version: 11,
    graph: { viewport: TB_VIEWPORT },
    expressions: {
      list: [
        { type: 'expression', id: 'ct', latex: 'c_{t}=0' },
        { type: 'expression', id: 'l', latex: `l=\\left[1...${n}\\right]` },
        { type: 'expression', id: 'at', latex: 'A_{t}=\\left(c_{t}\\to c_{t}+1,l\\to l\\cdot1.0001\\right)' },
      ],
    },
  });
  await tbSleep(800);
  const fps = await tbFrameRate(400);
  tbSetTicker(true);
  // created after setState, which drops helper expressions made before it
  const counter = Calc.HelperExpression({ latex: 'c_{t}' });
  const t0 = performance.now();
  await tbSleep(runMs);
  const ticks = counter.numericValue;
  const seconds = (performance.now() - t0) / 1000;
  tbSetTicker(false);
  if (!Number.isFinite(ticks)) throw new Error(`tickbench: could not read c_t for n=${n}`);
  await tbSleep(500);
  return { n, ticks, ticksPerSec: +(ticks / seconds).toFixed(1), fps: +fps.toFixed(0) };
}

async function tbRun(sizes = [32, 1024, 4096, 10000], options = {}) {
  const { collapsed = true, runMs = 5000, minFps = 30, waitMs = 600000 } = options;
  await tbWaitForFront(minFps, waitMs);
  Calc.updateSettings({ expressionsCollapsed: collapsed });
  const results = [];
  try {
    for (const n of sizes) results.push({ collapsed, ...(await tbOne(n, runMs)) });
  } finally {
    Calc.updateSettings({ expressionsCollapsed: false });
    Calc.setBlank();
  }
  return results;
}
