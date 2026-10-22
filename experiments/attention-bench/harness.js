// Check and time a head32 graph from the browser console on https://www.desmos.com/calculator.
//
//   1. Calc.setState(<contents of head32_readable.json or head32_fused.json>)
//   2. Paste this file, then: const REF = <contents of head32_ref.json>
//   3. await lmCheck(REF, 'y_{flat}')      // readable graph; use 'y_{f}' for the fused one
//   4. await lmTime('c', 9)                // fused graph; collapse the expression list first
//
// lmTime changes the slider s and measures until expression `id` shows a new value, so it
// includes the worker round trip and whatever the expression list re-renders.

const lmSleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function lmCheck(ref, listLatex, timeoutMs = 10000) {
  Calc.setExpression({ id: 's', latex: `s=${ref.seed}` });
  const helper = Calc.HelperExpression({ latex: listLatex });
  const start = performance.now();
  while (!(helper.listValue && helper.listValue.length === ref.Y.length)) {
    if (performance.now() - start > timeoutMs) {
      throw new Error(`lmCheck: ${listLatex} did not evaluate to ${ref.Y.length} numbers within ${timeoutMs} ms`);
    }
    await lmSleep(100);
  }
  const got = helper.listValue;
  let maxAbsErr = 0;
  let maxAbsRef = 0;
  for (let i = 0; i < ref.Y.length; i++) {
    maxAbsErr = Math.max(maxAbsErr, Math.abs(got[i] - ref.Y[i]));
    maxAbsRef = Math.max(maxAbsRef, Math.abs(ref.Y[i]));
  }
  return { n: got.length, maxAbsErr, relErr: maxAbsErr / maxAbsRef };
}

function lmWaitForChange(id, before, timeoutMs) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      Calc.unobserve('expressionAnalysis.lmTime');
      reject(new Error(`lmTime: ${id} did not update within ${timeoutMs} ms`));
    }, timeoutMs);
    Calc.observe('expressionAnalysis.lmTime', () => {
      const analysis = Calc.expressionAnalysis[id];
      const now = JSON.stringify(analysis?.evaluation?.value);
      if (analysis?.isError) {
        clearTimeout(timer);
        Calc.unobserve('expressionAnalysis.lmTime');
        reject(new Error(`lmTime: ${id} errored: ${analysis.errorMessage}`));
      } else if (now !== undefined && now !== before) {
        clearTimeout(timer);
        Calc.unobserve('expressionAnalysis.lmTime');
        resolve();
      }
    });
  });
}

async function lmTime(id, reps = 9, timeoutMs = 60000) {
  const times = [];
  for (let k = 0; k < reps; k++) {
    const before = JSON.stringify(Calc.expressionAnalysis[id]?.evaluation?.value);
    const t0 = performance.now();
    const changed = lmWaitForChange(id, before, timeoutMs);
    Calc.setExpression({ id: 's', latex: `s=${(0.1 + k * 0.0371).toFixed(4)}` });
    await changed;
    times.push(Math.round(performance.now() - t0));
    await lmSleep(150);
  }
  const sorted = [...times].sort((x, y) => x - y);
  return { medianMs: sorted[Math.floor(sorted.length / 2)], minMs: sorted[0], allMs: times };
}
