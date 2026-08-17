"use client";

/* The 48-expert grid.
 *
 * Generated rather than hand-authored: 48 identical cells differing only in
 * position is exactly the markup that should come from a loop, and it keeps
 * the fact this figure exists to show -- four are lit, forty-four are not --
 * in one readable place instead of scattered across 48 elements.
 */

const ROUTED = new Set([5, 17, 26, 39]);

export function ExpertGrid() {
  return (
    <div className="panel gridpanel">
      <span className="phlabel">One MoE layer, one token</span>

      <div className="egrid">
        {Array.from({ length: 48 }).map((_, i) => (
          <span
            key={i}
            className={`ecell ${ROUTED.has(i) ? "on" : ""}`}
            aria-hidden="true"
          />
        ))}
      </div>

      <div className="gridlegend">
        <span>
          <i className="swatch on" /> 4 routed
        </span>
        <span>
          <i className="swatch" /> 44 untouched
        </span>
        <span>
          <i className="swatch shared" /> 1 shared, always on
        </span>
      </div>

      <p className="panelnote">
        Capacity comes from memory, cost comes from compute. Adding experts
        raises the parameter count without raising the work per token — which is
        why a 1.32B model answers on a CPU at ten tokens a second.
      </p>
    </div>
  );
}
