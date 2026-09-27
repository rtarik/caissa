/**
 * The network each game plays, as `models/index.json` lists them.
 *
 * One opponent per game: its four levels are four thinking budgets for the same
 * network. Should the index ever list more than one for a game - a newer stage
 * exported beside an older one - the newest is the one played.
 */
export interface ModelEntry {
  game: string;
  /** File name without extension; older indexes left it out, meaning the game's name. */
  file?: string;
  generation?: number | null;
  parameters?: number | null;
}

export interface Network {
  game: string;
  file: string;
  generation: number;
}

/** The newest network of each game in the index. */
export function networkByGame(models: ModelEntry[]): Map<string, Network> {
  const byGame = new Map<string, Network>();
  for (const model of models) {
    const network: Network = {
      game: model.game,
      file: model.file ?? model.game,
      generation: model.generation ?? 0,
    };
    const known = byGame.get(model.game);
    if (!known || network.generation > known.generation) byGame.set(model.game, network);
  }
  return byGame;
}
