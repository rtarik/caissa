/** Search's visit counts laid over the board, for action spaces that are squares. */
export function heatmap(counts: number[], squares: number, columns: number): string {
  const best = Math.max(...counts.slice(0, squares));
  const cells = counts
    .slice(0, squares)
    .map((v) => `<div class="heat" style="opacity:${
      best > 0 ? Math.max(0.04, v / best) : 0.04
    }" title="${v} visits"></div>`)
    .join("");
  return `<div class="heatmap" style="grid-template-columns: repeat(${columns}, 1fr)">${cells}</div>`;
}
