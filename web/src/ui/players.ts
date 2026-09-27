/**
 * The two players, drawn above and below the board.
 *
 * A board with a status line underneath says whose turn it is in words. A card
 * for each side says it by where your eye already is: the side to move is lit,
 * the engine's card says when it is thinking, and each carries the piece or disc
 * it plays with, so there is never a question of which colour is yours. Before
 * a game each also carries its choice - the level on Caissa's, your side on yours.
 */
import { pieceSvg, tokenSvg } from "./pieces";

export type Side = "you" | "engine";

export interface PlayerCard {
  side: Side;
  name: string;
  /** A second line: the level and its rating, or the colour you play. */
  detail: string;
  /** Short, and only while it is true: "Your move", "Thinking…" - or the Start button. */
  status: string;
  /** Whether this side is to move. */
  active: boolean;
  /** Whether this side has won - the game is over and the card says so. */
  winner: boolean;
  /** A choice to make before the game, drawn on a row of its own under the rest. */
  controls?: string;
}

/**
 * The piece a side plays with, drawn the way the board draws it.
 *
 * Chess sides are colours rather than owners, so its token is a king in the
 * side's colour; Isolation's pieces are pawns; everything else is a disc in the
 * side's colour from the game's own palette.
 */
export function tokenFor(game: string, side: Side, white: boolean): string {
  if (game === "chess") return `<span class="player-token piece">${pieceSvg("k", white ? "w" : "b")}</span>`;
  if (game === "isola") return `<span class="player-token pawn ${side}">${tokenSvg()}</span>`;
  return `<span class="player-token disc ${side}"></span>`;
}

export function playerCardHtml(card: PlayerCard, token: string): string {
  const classes = ["player-card", card.side];
  if (card.active) classes.push("active");
  if (card.winner) classes.push("winner");
  if (card.controls) classes.push("choosing");
  return `<div class="${classes.join(" ")}">
    ${token}
    <span class="player-text">
      <span class="player-name">${card.name}</span>
      <span class="player-detail">${card.detail}</span>
    </span>
    ${card.status ? `<span class="player-status">${card.status}</span>` : ""}
    ${card.controls ? `<div class="player-controls">${card.controls}</div>` : ""}
  </div>`;
}
