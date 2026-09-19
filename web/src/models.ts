/**
 * The networks the site ships, as `models/index.json` lists them.
 *
 * Most games have one. Chess keeps one per training stage, so the owner can
 * play each and feel the difference; the page offers the newest by default and
 * a choice whenever there is more than one.
 */
export interface ModelEntry {
  game: string;
  /** File name without extension; older indexes left it out, meaning the game's name. */
  file?: string;
  label?: string | null;
  generation?: number | null;
  parameters?: number | null;
}

export interface Network {
  game: string;
  file: string;
  label: string;
  generation: number;
}

/** Each game's networks, newest first. */
export function networksByGame(models: ModelEntry[]): Map<string, Network[]> {
  const byGame = new Map<string, Network[]>();
  for (const model of models) {
    const generation = model.generation ?? 0;
    const network: Network = {
      game: model.game,
      file: model.file ?? model.game,
      label: model.label ?? (generation === 0 ? "Untrained" : `Generation ${generation}`),
      generation,
    };
    byGame.set(model.game, [...(byGame.get(model.game) ?? []), network]);
  }
  for (const list of byGame.values()) list.sort((a, b) => b.generation - a.generation);
  return byGame;
}
