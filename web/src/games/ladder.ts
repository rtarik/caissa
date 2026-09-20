/**
 * The games, in the order they are being built.
 *
 * Each one adds exactly one new difficulty to the framework, which is why the
 * order matters and why the menu shows all of them rather than only the ones
 * that are finished - the sequence is part of what the project is.
 *
 * Availability is not declared here. It comes from `models/index.json`, written
 * by the export script from the files that actually exist, so the menu cannot
 * offer a game whose network is missing or hide one that has just been trained.
 *
 * `title` is display text and `key` is the identifier used for model files,
 * checkpoints and registry lookups. They differ on purpose for two games:
 * "Connect 4" is a live Hasbro trademark and "Isola" is Ravensburger's, so the
 * page shows the generic names while the keys - and everything already trained
 * against them - stay put. Reversi, Gomoku and Dots & Boxes need no such care;
 * those are the unencumbered names already (it is *Othello* that is the
 * trademark, which is why nearly all software says Reversi).
 */
export interface LadderEntry {
  key: string;
  title: string;
  /** What the game is, in a few words. */
  blurb: string;
  /** How to win, for someone who has never played it. Shown while playing. */
  howToWin: string;
  /** What it forces the framework to handle that earlier games did not.
   *  Kept off the play screen: it is about the project, not about the game. */
  teaches: string;
  /** The same thing at length, for the page that explains the project. */
  detail: string;
  /** Board size, in the terms a player would use. */
  board: string;
  /** How many moves the engine picks between, which is what makes a game hard
   *  for a search rather than for a person. */
  moves: string;
  /** Where its skill came from. Chess is the only one that started with ours. */
  learned: string;
  /** Names for the two seats, where the game has its own: chess's White and Black. */
  seats?: [string, string];
}

export const LADDER: LadderEntry[] = [
  {
    key: "connect4",
    title: "Four in a Row",
    blurb: "Drop a disc; gravity does the rest.",
    howToWin: "Get four of your discs in a line: across, up or diagonally.",
    teaches: "The baseline. Solved, so play can be graded against perfect.",
    detail:
      "The simplest game that still needs real lookahead, and the only one here that has been solved outright, so its play can be graded against perfect rather than against itself.",
    board: "6 × 7",
    moves: "7",
    learned: "Self-play only, 40 rounds",
  },
  {
    key: "reversi",
    title: "Reversi",
    blurb: "Flank a line of discs to flip it.",
    howToWin: "Trap a line of your opponent's discs between two of yours to flip them. Most discs at the end wins.",
    teaches: "Passing: having no legal move does not end the game.",
    detail:
      "The first game where a player can be left with no legal move at all. Turns had to stop being something the framework could count and start being something the game states, which is a change that reaches into the search, the training labels and the board.",
    board: "8 × 8",
    moves: "up to 65",
    learned: "Self-play only, 30 rounds",
  },
  {
    key: "gomoku",
    title: "Gomoku",
    blurb: "Five in a row on an open board.",
    howToWin: "Place a stone anywhere. Five in a row wins, across, up or diagonally.",
    teaches: "A large action space, and very sparse policy targets.",
    detail:
      "Eighty-one places to put a stone, against seven columns in Four in a Row. The policy the network has to predict becomes almost entirely zeros, and the same search budget spread over ten times the moves buys far less certainty.",
    board: "9 × 9",
    moves: "81",
    learned: "Self-play only, 40 rounds",
  },
  {
    key: "isola",
    title: "Isolation",
    blurb: "Move your piece, then destroy a square.",
    howToWin: "Each turn, step one square and then destroy any empty square. Strand your opponent with nowhere to step.",
    teaches: "Compound actions, a rehearsal for chess's move encoding.",
    detail:
      "A turn here is two decisions: where to step, and which square to remove. They are encoded as one number rather than two half-moves, which is the same trick chess needs for its 4,672 moves, rehearsed on a smaller board.",
    board: "7 × 7",
    moves: "392",
    learned: "Self-play only, 30 rounds",
  },
  {
    key: "dotsandboxes",
    title: "Dots & Boxes",
    blurb: "Draw lines; close a box to claim it, then move again.",
    howToWin: "Draw one line per turn. Complete the fourth side of a box to claim it and go again. Most boxes wins.",
    teaches: "Turns that don't always pass, so the framework must be told whose move it is.",
    detail:
      "Closing a box earns another turn, so the players do not alternate. Every place in the project that assumed they did had to be found and fixed, including the sign of the training labels, which is the kind of bug that trains happily in the wrong direction and never raises an error.",
    board: "5 × 5 boxes",
    moves: "60",
    learned: "Self-play only, 55 rounds",
  },
  {
    key: "chess",
    title: "Chess",
    blurb: "The full game, pieces and all.",
    howToWin: "Checkmate the enemy king. All the usual rules, including castling, en passant and promotion.",
    teaches: "Everything at once, plus a bootstrap from human games.",
    detail:
      "Everything at once, and the only game here that did not start from nothing. Learning chess from scratch is a question of compute rather than method, and the compute is not available on one laptop, so this one began by copying people.",
    board: "8 × 8",
    moves: "4,672",
    learned: "530,000 human games, then self-play",
    seats: ["White", "Black"],
  },
];

export const TITLES: Record<string, string> = Object.fromEntries(
  LADDER.map((entry) => [entry.key, entry.title]),
);
