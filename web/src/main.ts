/**
 * The site: a gallery of games, and a screen for playing one.
 *
 * Two audiences, one page. Someone who wants a game of Reversi should meet a
 * board and nothing else; someone who wants to know what the engine is thinking
 * should be able to ask. So the play screen shows the board, whose turn it is
 * and the result, and everything about *how* it plays - search budgets, visit
 * heatmaps, what each game taught the framework - sits behind a quiet toggle and
 * a page of its own.
 *
 * Routing is the URL hash, which costs nothing and buys the back button, a
 * bookmarkable game, and a reload that stays where you were.
 */
import "./style.css";
import { createGame } from "./games/registry";
import { LADDER, type LadderEntry } from "./games/ladder";
import type { Game } from "./games/types";
import { VIEWS, type View } from "./ui/views";
import { iconFor } from "./ui/icons";
import { moveLabel } from "./ui/notation";
import { rowsOf, undoTarget, type Ply } from "./moves";
import { browserStore, clearGames, deleteGame, loadGames, newGameId, saveGame, type GameRecord } from "./archive";
import { chessOutcome, collectionPgn, engineName, movesText, pgnFileName, recordPgn } from "./pgn";
import { pieceSvg } from "./ui/pieces";
import type { ChessState } from "./games/chess";
import { STANDARD_FEN, fenOf, parseFen, problems, withCastling, withPiece, withSideToMove, type Castling, type Setup } from "./editor";
import { editorBoardHtml, paletteHtml, problemsHtml, rulesHtml, type Tool } from "./ui/editorview";
import { playerCardHtml, tokenFor, type PlayerCard } from "./ui/players";
import { approximately, readSettings, resolveSeat, sheetHtml, type Rating, type Settings } from "./ui/sheet";
import type { FromEngine, ToEngine } from "./engine/protocol";
import { networksByGame, type ModelEntry, type Network } from "./models";
import { LEVELS, DEFAULT_LEVEL } from "./levels";
import { parseRoute, type Route } from "./routes";

const asset = (path: string) => new URL(path, document.baseURI).href;

const DETAILS_KEY = "caissa:details";

interface ModelIndex {
  models: ModelEntry[];
}

let route: Route = { name: "gallery" };
let trained = new Set<string>();
/** Each game's networks, newest first, and the one being played. */
let networks = new Map<string, Network[]>();
let network: Network | null = null;
let entry: LadderEntry = LADDER[0];
let game: Game<unknown> = createGame(entry.key);
let view: View = VIEWS[entry.key];
let moves: number[] = [];
let humanFirst = true;
let simulations = LEVELS[DEFAULT_LEVEL].simulations;
let thinking = false;
let ready = false;
let engineInfo = "";
let report: Extract<FromEngine, { kind: "move" }> | null = null;
/** Half a turn, for games that ask for more than one decision. */
let pending: number | null = null;
let error: string | null = null;
let showDetails = false;
/** What the new-game sheet last chose, per visit; Random stays Random. */
let settings: Settings = { level: DEFAULT_LEVEL, seat: "first", network: null };
let sheetOpen = false;
let sheetCancellable = false;
/**
 * How far into the game the board is showing, while stepping back through it;
 * null shows the game as it stands. Looking back never changes the game.
 */
let viewing: number | null = null;
/** The number of the latest engine request; answers to any other are stale. */
let requestId = 0;
/**
 * The chess game in progress as the history knows it: an id once there is
 * something worth keeping, when it began, and whether you resigned it.
 */
let gameId: string | null = null;
let startedAt: string | null = null;
let resigned = false;
const store = browserStore();
/** Where the game in progress began, when not at the usual start: a FEN, for chess. */
let startFen: string | null = null;
/** Where the next game will begin: what the new-game sheet shows, until changed. */
let customStart: string | null = null;
/** The board editor's work in progress while it is open, and the piece it places. */
let editing: Setup | null = null;
let tool: Tool = "P";
/** Measured ratings, per network file and then per level, where there are any. */
let ratings = new Map<string, Map<string, Rating>>();
try {
  showDetails = localStorage.getItem(DETAILS_KEY) === "1";
} catch {
  // Private browsing, or storage blocked. The toggle still works for this visit.
}

const worker = new Worker(new URL("./engine/worker.ts", import.meta.url), {
  type: "module",
});

worker.onmessage = (event: MessageEvent<FromEngine>) => {
  const message = event.data;
  if (message.kind === "ready") {
    ready = true;
    error = null;
    const params = message.parameters
      ? `${(message.parameters / 1000).toFixed(0)}k parameters`
      : "";
    engineInfo = message.generation === 0
      ? `Untrained network · ${params}`
      : message.generation
        ? `Generation ${message.generation} · ${params}`
        : params;
    render();
    advance();
  } else if (message.kind === "move") {
    // An answer to a request since superseded - by an undo or a new game - is
    // for a position that no longer exists. Playing it would put a move from
    // one game into another.
    if (message.id !== requestId) return;
    thinking = false;
    report = message;
    moves = [...moves, message.action];
    saveCurrent();
    render();
    advance();
  } else {
    if (message.id !== undefined && message.id !== requestId) return;
    thinking = false;
    error = message.message;
    render();
  }
};

/** Forget whatever the engine is working on: its answer will be ignored. */
function cancelSearch(): void {
  requestId += 1;
  thinking = false;
}

function loadEngine(): void {
  ready = false;
  error = null;
  engineInfo = "";
  render();
  const file = network?.file ?? entry.key;
  worker.postMessage({
    kind: "load",
    model: asset(`models/${file}.onnx`),
    manifest: asset(`models/${file}.json`),
  } satisfies ToEngine);
}

// ---------------------------------------------------------------- game state

/**
 * Every position along the game, and each move's name, kept in step with the
 * move list.
 *
 * Browsing needs any position in the game, and undo cuts the game short, so
 * this keeps them all rather than only the last. It is brought up to date by
 * finding how much of the old game the new move list shares and replaying only
 * the rest: a move made adds one position, an undo only truncates, and a chess
 * game is never replayed from the start just to step back one move. `moves` is
 * replaced rather than changed, so the array itself says whether this is current.
 */
let history: {
  game: Game<unknown>; start: string | null; moves: number[]; states: unknown[]; labels: string[];
} = { game, start: null, moves: [], states: [game.initialState()], labels: [] };

/** The position a game begins in: the usual one, or where the editor set it up. */
function firstPosition(): unknown {
  return startFen && game.positionFrom ? game.positionFrom(startFen) : game.initialState();
}

function positions(): typeof history {
  if (history.moves === moves && history.game === game && history.start === startFen) return history;
  const same = history.game === game && history.start === startFen;
  const old = same ? history.moves : [];
  let common = 0;
  while (common < old.length && common < moves.length && old[common] === moves[common]) common++;
  const states = same ? history.states.slice(0, common + 1) : [firstPosition()];
  const labels = same ? history.labels.slice(0, common) : [];
  for (let index = common; index < moves.length; index++) {
    labels.push(moveLabel(entry.key, states[index], moves[index]));
    states.push(game.apply(states[index], moves[index]));
  }
  history = { game, start: startFen, moves, states, labels };
  return history;
}

/** The game as it stands. */
const state = () => positions().states[moves.length];
/** How many moves in the board is showing: all of them, unless stepping back. */
const shownPly = () => viewing ?? moves.length;

/** The seat the human plays: 0 moves first. */
const humanSeat = () => (humanFirst ? 0 : 1);
// Asked of the game rather than counted from the move list: after a Dots & Boxes
// bonus move the same player is to move again, and parity would say otherwise.
const humanToMove = () => game.toPlay(state()) === humanSeat();
const finished = () => resigned || game.terminalValue(state()) !== null;

/**
 * True when the mover's only legal action is the game's pass.
 *
 * Asked of the game rather than guessed from where the action sits. The first
 * version assumed the pass was the last action index, which is true for Reversi
 * and false everywhere else: in Four in a Row, with only the seventh column
 * open, it dropped the disc for you and announced that you had passed.
 */
function mustPass(): boolean {
  const pass = game.passAction;
  if (pass === undefined) return false;
  const legal = game.legalActions(state());
  return legal[pass] && legal.filter(Boolean).length === 1;
}

/** Whether the last move kept the turn - a closed box, in Dots & Boxes. */
function lastMoveKeptTurn(): boolean {
  if (moves.length === 0) return false;
  const { states } = positions();
  return game.toPlay(states[moves.length - 1]) === game.toPlay(states[moves.length]);
}

function play(action: number): void {
  if (thinking || !ready || finished() || !humanToMove()) return;
  if (!game.legalActions(state())[action]) return;
  report = null;
  pending = null;
  moves = [...moves, action];
  saveCurrent();
  render();
  advance();
}

/**
 * Move the game on: let the engine think, or pass for a stuck player.
 *
 * Reversi can leave someone with nothing to do. Asking them to click a "pass"
 * button would be demanding a decision they do not have, so it is taken for them
 * after a beat - long enough that the turn does not appear to have been skipped.
 */
function advance(): void {
  if (!ready || thinking || finished() || route.name !== "play" || sheetOpen || editing) return;

  if (humanToMove()) {
    if (mustPass()) {
      render();
      setTimeout(() => {
        moves = [...moves, game.passAction!];
        render();
        advance();
      }, 700);
    }
    return;
  }

  thinking = true;
  requestId += 1;
  render();
  worker.postMessage({
    kind: "move", id: requestId, moves, simulations, start: startFen ?? undefined,
  } satisfies ToEngine);
}

function reset(): void {
  cancelSearch();
  moves = [];
  gameId = null;
  startedAt = null;
  resigned = false;
  viewing = null;
  report = null;
  pending = null;
  render();
  advance();
}

/** Take back the player's last move, and the engine's reply with it. */
function undo(): void {
  const target = undoPoint();
  if (target === null) return;
  cancelSearch();
  moves = moves.slice(0, target);
  resigned = false;
  viewing = null;
  report = null;
  pending = null;
  saveCurrent();
  render();
  advance();
}

/** Hand the browser a file to save. Nothing is sent anywhere: the file is made here. */
function download(name: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: "application/x-chess-pgn" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Copy to the clipboard, saying on the button whether it worked. */
async function copyText(text: string, button: HTMLButtonElement): Promise<void> {
  let copied = false;
  try {
    await navigator.clipboard.writeText(text);
    copied = true;
  } catch {
    // The clipboard API needs a secure context and permission; the old way does not.
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    try {
      copied = document.execCommand("copy");
    } catch {
      copied = false;
    }
    area.remove();
  }
  flash(button, copied ? "Copied" : "Couldn't copy");
}

/** Show a word on a button for a moment, then put its label back. */
function flash(button: HTMLButtonElement, word: string): void {
  const label = button.dataset.label ?? button.textContent ?? "";
  button.dataset.label = label;
  button.textContent = word;
  window.setTimeout(() => {
    button.textContent = label;
  }, 1400);
}

/**
 * Clearing the whole history takes two clicks: the first arms the button, the
 * second, within a few seconds, clears. A single stray click costs nothing.
 */
let clearArmed = 0;
function armOrClear(button: HTMLButtonElement): void {
  if (Date.now() - clearArmed < 4000) {
    clearGames(store);
    clearArmed = 0;
    renderGames();
    return;
  }
  clearArmed = Date.now();
  button.textContent = "Click again to clear";
  window.setTimeout(() => {
    if (Date.now() - clearArmed >= 4000) button.textContent = "Clear history";
  }, 4000);
}

/** Resign the chess game: it ends here, as a loss, and is saved that way. */
function resign(): void {
  if (entry.key !== "chess" || finished() || !ready) return;
  cancelSearch();
  resigned = true;
  viewing = null;
  pending = null;
  saveCurrent();
  render();
}

/**
 * The current chess game as a record, or null while there is nothing of yours
 * in it yet - a game the engine opened and you never answered is not kept.
 */
function currentRecord(): GameRecord | null {
  if (entry.key !== "chess" || moves.length === 0 || !yourMoveMade()) return null;
  gameId ??= newGameId();
  startedAt ??= new Date().toISOString();
  const level = LEVELS[settings.level];
  const outcome = chessOutcome(state() as ChessState, humanFirst, resigned);
  return {
    id: gameId,
    game: "chess",
    started: startedAt,
    updated: new Date().toISOString(),
    moves,
    humanWhite: humanFirst,
    level: level.label,
    simulations: level.simulations,
    network: network?.file ?? entry.key,
    networkLabel: network?.label ?? "",
    rating: ratings.get(network?.file ?? "")?.get(level.label)?.rating,
    result: outcome.result,
    termination: outcome.termination,
    ...(startFen ? { start: startFen } : {}),
  };
}

/**
 * Keep the history in step with the game; storage failing never stops play.
 *
 * Undo can rewind a game past your first move, leaving nothing of yours in it.
 * The record then goes too: left behind, it would still say "resigned" about a
 * game that has been taken back to its first position.
 */
function saveCurrent(): void {
  const record = currentRecord();
  if (record) saveGame(store, record);
  else if (gameId && entry.key === "chess") deleteGame(store, gameId);
}

function undoPoint(): number | null {
  if (!ready) return null;
  const { states } = positions();
  const seats = moves.map((_, index) => game.toPlay(states[index]));
  return undoTarget(seats, moves, humanSeat(), game.passAction);
}

/** Step the board to another point in the game; past the end means now. */
function showPly(ply: number): void {
  const clamped = Math.max(0, Math.min(ply, moves.length));
  viewing = clamped === moves.length ? null : clamped;
  pending = null;
  render();
}

// -------------------------------------------------------------------- routing

function applyRoute(): void {
  const next = parseRoute(location.hash);
  const changedGame = next.name === "play" && (route.name !== "play" || route.key !== next.key);
  route = next;

  if (route.name === "play") {
    const found = LADDER.find((item) => item.key === (route as { key: string }).key)!;
    document.title = `Caissa — ${found.title}`;
    if (changedGame) {
      entry = found;
      network = networks.get(entry.key)?.[0] ?? null;
      game = createGame(entry.key);
      view = VIEWS[entry.key];
      cancelSearch();
      moves = [];
      startFen = null;
      viewing = null;
      report = null;
      pending = null;
      settings = { ...settings, network: network?.file ?? null };
      render();
      loadEngine();
      openSheet(false);
      return;
    }
  } else {
    document.title = route.name === "about" ? "Caissa — How it works"
      : route.name === "games" ? "Caissa — Your games" : "Caissa";
  }
  render();
}

// ------------------------------------------------------------------ skeleton

const app = document.getElementById("app")!;
app.innerHTML = `
  <header class="masthead">
    <a class="brand" href="#/" aria-label="All games">
      <span class="mark" aria-hidden="true"></span>
      <span class="wordmark">Caissa</span>
    </a>
    <nav class="masthead-links">
      <a href="#/how-it-works" class="quiet-link" id="about-link">How it works</a>
    </nav>
  </header>

  <main id="screens">
    <section class="screen" id="gallery">
      <h1 class="section-title">Choose a game</h1>
      <div class="cards" id="cards"></div>
    </section>

    <section class="screen" id="play" hidden>
      <div class="play-head">
        <a class="back" href="#/" aria-label="All games" title="All games">
          <svg viewBox="0 0 24 24" aria-hidden="true" class="chevron"><path d="M15 5l-7 7 7 7"/></svg>
        </a>
        <h1 class="play-title" id="play-title"></h1>
      </div>
      <div class="layout">
        <div class="stage">
          <div id="opponent"></div>
          <div class="board" id="board"></div>
          <div id="you"></div>
          <div class="status" id="status"></div>
        </div>
        <aside class="side">
          <section class="panel game-panel">
            <div class="move-head">
              <span>Moves</span>
              <span class="move-where" id="move-where"></span>
            </div>
            <div class="move-list" id="move-list"></div>
            <div class="move-nav" role="group" aria-label="Step through the game">
              <button class="button nav" data-nav="first" aria-label="Start of the game" title="Start (Home)">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M17 5l-7 7 7 7M7 5v14"/></svg>
              </button>
              <button class="button nav" data-nav="back" aria-label="Previous move" title="Back (←)">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 5l-7 7 7 7"/></svg>
              </button>
              <button class="button nav" data-nav="forward" aria-label="Next move" title="Forward (→)">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5l7 7-7 7"/></svg>
              </button>
              <button class="button nav" data-nav="last" aria-label="Back to the game" title="Now (End)">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5l7 7-7 7M17 5v14"/></svg>
              </button>
            </div>
            <div class="move-actions">
              <button class="button" id="undo">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 14L4 9l5-5M4 9h10a6 6 0 0 1 0 12h-3"/></svg>
                Undo
              </button>
              <button class="button primary" id="new">New game</button>
            </div>
            <div class="chess-actions" id="chess-actions" hidden>
              <div class="export-row">
                <button class="button small" id="copy-pgn">Copy PGN</button>
                <button class="button small" id="download-pgn">Download PGN</button>
              </div>
              <div class="chess-footer">
                <button class="button small ghost" id="resign">
                  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 21V4m0 0h11l-2 4 2 4H5"/></svg>
                  Resign
                </button>
                <a class="quiet-link" id="games-link" href="#/games/chess">Your games</a>
              </div>
            </div>
          </section>
          <div class="details">
            <button class="quiet-link toggle" id="details-toggle" aria-expanded="false">
              Engine details
            </button>
            <section class="panel" id="analysis" hidden></section>
            <section class="panel" id="notes" hidden></section>
          </div>
        </aside>
      </div>
      <div class="editor" id="editor" hidden>
        <div class="editor-stage">
          <div class="board chess editor-board" id="editor-board"></div>
        </div>
        <aside class="editor-side">
          <section class="panel editor-panel">
            <h2 class="editor-title">Set up a position</h2>
            <p class="editor-hint">Pick a piece, then click squares to place it. Click a piece again to take it off.</p>
            <div id="editor-palette"></div>
            <div class="editor-tools">
              <button class="button small" data-editor="clear">Clear board</button>
              <button class="button small" data-editor="standard">Starting position</button>
              <button class="button small" data-editor="current" id="editor-current">This game</button>
            </div>
            <div id="editor-rules"></div>
            <label class="fen-field">
              <span>FEN</span>
              <input id="editor-fen" spellcheck="false" autocomplete="off" autocapitalize="off">
            </label>
            <p class="fen-error" id="editor-fen-error" hidden></p>
            <div id="editor-problems"></div>
            <div class="sheet-actions">
              <button class="button ghost" data-editor="cancel">Cancel</button>
              <button class="button primary" data-editor="use" id="editor-use">Use this position</button>
            </div>
          </section>
        </aside>
      </div>
      <div class="sheet-backdrop" id="sheet" hidden></div>
    </section>

    <section class="screen" id="games" hidden></section>

    <section class="screen" id="about" hidden></section>
  </main>

  <footer>
    <span>Caissa</span>
    <a class="quiet-link" href="#/how-it-works">How it works</a>
    <a class="quiet-link" href="${asset("THIRD_PARTY_NOTICES.txt")}">Licences</a>
  </footer>
`;

const cardsEl = document.getElementById("cards")!;
const boardEl = document.getElementById("board")!;
const statusEl = document.getElementById("status")!;
const analysisEl = document.getElementById("analysis")!;
const notesEl = document.getElementById("notes")!;
const detailsEl = document.querySelector<HTMLElement>(".details")!;
const detailsToggle = document.getElementById("details-toggle") as HTMLButtonElement;
const opponentEl = document.getElementById("opponent")!;
const moveListEl = document.getElementById("move-list")!;
const moveWhereEl = document.getElementById("move-where")!;
const undoEl = document.getElementById("undo") as HTMLButtonElement;
const navEls = [...document.querySelectorAll<HTMLButtonElement>("[data-nav]")];
const youEl = document.getElementById("you")!;
const sheetEl = document.getElementById("sheet")!;
const playTitleEl = document.getElementById("play-title")!;
const aboutEl = document.getElementById("about")!;
const gamesEl = document.getElementById("games")!;
const chessActionsEl = document.getElementById("chess-actions")!;
const resignEl = document.getElementById("resign") as HTMLButtonElement;
const copyEl = document.getElementById("copy-pgn") as HTMLButtonElement;
const downloadEl = document.getElementById("download-pgn") as HTMLButtonElement;
const gamesLinkEl = document.getElementById("games-link")!;
const editorEl = document.getElementById("editor")!;
const editorBoardEl = document.getElementById("editor-board")!;
const editorPaletteEl = document.getElementById("editor-palette")!;
const editorRulesEl = document.getElementById("editor-rules")!;
const editorFenEl = document.getElementById("editor-fen") as HTMLInputElement;
const editorFenErrorEl = document.getElementById("editor-fen-error")!;
const editorProblemsEl = document.getElementById("editor-problems")!;
const editorUseEl = document.getElementById("editor-use") as HTMLButtonElement;
const editorCurrentEl = document.getElementById("editor-current") as HTMLButtonElement;
const layoutEl = document.querySelector<HTMLElement>("#play .layout")!;
const screens: Record<string, HTMLElement> = {
  gallery: document.getElementById("gallery")!,
  play: document.getElementById("play")!,
  games: gamesEl,
  about: aboutEl,
};

boardEl.addEventListener("click", (event) => {
  const target = (event.target as HTMLElement).closest<HTMLElement>(
    "[data-action], [data-square]",
  );
  if (!target) return;

  const selection = view.select(target, context());
  if (!selection) return;
  if (selection.kind === "action") {
    pending = null;
    play(selection.action);
  } else {
    pending = selection.pending;
    render();
  }
});
document.getElementById("new")!.addEventListener("click", () => openSheet(true));
undoEl.addEventListener("click", undo);
resignEl.addEventListener("click", resign);
copyEl.addEventListener("click", () => {
  const record = currentRecord();
  if (record) void copyText(recordPgn(record), copyEl);
});
downloadEl.addEventListener("click", () => {
  const record = currentRecord();
  if (record) download(pgnFileName(record), recordPgn(record));
});
gamesEl.addEventListener("click", (event) => {
  const target = (event.target as HTMLElement).closest<HTMLButtonElement>("button[data-games], button[data-game-action]");
  if (!target) return;
  const all = loadGames(store).filter((record) => record.game === "chess");
  const action = target.dataset.games ?? target.dataset.gameAction;
  const record = all.find((game) => game.id === target.closest<HTMLElement>("[data-id]")?.dataset.id);

  if (action === "all" && all.length) download("caissa-games.pgn", collectionPgn(all));
  else if (action === "clear") armOrClear(target);
  else if (record && action === "copy") void copyText(recordPgn(record), target);
  else if (record && action === "download") download(pgnFileName(record), recordPgn(record));
  else if (record && action === "delete") {
    deleteGame(store, record.id);
    if (record.id === gameId) gameId = null;
    renderGames();
  }
});
statusEl.addEventListener("click", (event) => {
  if ((event.target as HTMLElement).closest("[data-nav-inline]")) showPly(moves.length);
});
moveListEl.addEventListener("click", (event) => {
  const ply = (event.target as HTMLElement).closest<HTMLElement>("[data-ply]");
  // A move's button shows the position after it was played.
  if (ply) showPly(Number(ply.dataset.ply) + 1);
});
for (const button of navEls) {
  button.addEventListener("click", () => step(button.dataset.nav!));
}
document.addEventListener("keydown", (event) => {
  if (route.name !== "play" || sheetOpen || editing) return;
  // The target may be the document itself, which has no closest(): not an element, not typing.
  const target = event.target instanceof Element ? event.target : null;
  const typing = target?.closest("input, select, textarea, [contenteditable]");
  if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
  const keys: Record<string, string> = {
    ArrowLeft: "back", ArrowRight: "forward", Home: "first", End: "last",
  };
  if (!keys[event.key]) return;
  event.preventDefault();
  step(keys[event.key]);
});

function step(direction: string): void {
  const at = shownPly();
  if (direction === "first") showPly(0);
  else if (direction === "back") showPly(at - 1);
  else if (direction === "forward") showPly(at + 1);
  else showPly(moves.length);
}
sheetEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.target as HTMLFormElement;
  const chosen = readSettings(new FormData(form), settings);
  closeSheet();
  startGame(chosen);
});
sheetEl.addEventListener("change", (event) => {
  // Ratings belong to a network, so switching the engine redraws the levels.
  if ((event.target as HTMLInputElement).name === "network") refreshSheet();
});
sheetEl.addEventListener("click", (event) => {
  const target = event.target as HTMLElement;
  const form = sheetEl.querySelector<HTMLFormElement>("form");
  if (target.closest("[data-sheet='setup']") && form) {
    // Keep what was chosen so far, then set up the position.
    settings = readSettings(new FormData(form), settings);
    closeSheet(false);
    openEditor(customStart ?? STANDARD_FEN);
    return;
  }
  if (target.closest("[data-sheet='standard']")) {
    customStart = null;
    refreshSheet();
    return;
  }
  // The backdrop itself, or the Cancel button - not a click inside the sheet.
  if (target === sheetEl || target.closest("[data-sheet='cancel']")) closeSheet();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && sheetOpen) closeSheet();
  else if (event.key === "Escape" && editing) leaveEditor(false);
});

// ------------------------------------------------------------------- editor

editorEl.addEventListener("click", (event) => {
  if (!editing) return;
  const target = event.target as HTMLElement;
  const square = target.closest<HTMLElement>("[data-square]");
  if (square) {
    const index = Number(square.dataset.square);
    // Placing a piece where the same piece stands takes it off: one tool does both.
    const same = editing.squares[index] === tool;
    editing = withPiece(editing, index, same ? null : tool);
    renderEditor(true);
    return;
  }
  const action = target.closest<HTMLElement>("[data-editor]")?.dataset.editor;
  if (action === "clear") editing = { ...parseFen("8/8/8/8/8/8/8/8 w - - 0 1")! };
  else if (action === "standard") editing = parseFen(STANDARD_FEN);
  else if (action === "current") editing = parseFen((state() as ChessState).fen);
  else if (action === "cancel") return leaveEditor(false);
  else if (action === "use") return leaveEditor(true);
  else return;
  renderEditor(true);
});

editorEl.addEventListener("change", (event) => {
  if (!editing) return;
  const input = event.target as HTMLInputElement;
  if (input.name === "tool") tool = (input.value || null) as Tool;
  else if (input.name === "tomove") editing = withSideToMove(editing, input.value === "w");
  else if (input.name === "castle") editing = withCastling(editing, input.value as keyof Castling, input.checked);
  else return;
  renderEditor(true);
});

editorFenEl.addEventListener("input", () => {
  // A FEN is usually pasted whole, sometimes typed: only a complete one moves
  // the board, and the box is never rewritten under the cursor.
  const parsed = parseFen(editorFenEl.value);
  editorFenErrorEl.hidden = parsed !== null || editorFenEl.value.trim() === "";
  editorFenErrorEl.textContent = "That isn't a complete FEN yet.";
  if (!parsed) return;
  editing = parsed;
  renderEditor(false);
});

function openEditor(fen: string): void {
  editing = parseFen(fen) ?? parseFen(STANDARD_FEN);
  editorFenErrorEl.hidden = true;
  editorEl.hidden = false;
  layoutEl.hidden = true;
  renderEditor(true);
}

/** Back to the sheet, keeping the position if asked and if it can be played. */
function leaveEditor(keep: boolean): void {
  if (keep && editing) {
    if (problems(editing).length) return;
    customStart = fenOf(editing);
  }
  editing = null;
  editorEl.hidden = true;
  layoutEl.hidden = false;
  openSheet(sheetCancellable);
}

function renderEditor(rewriteFen: boolean): void {
  if (!editing) return;
  // The board faces the side you are going to play; Random shows White below.
  editorBoardEl.innerHTML = editorBoardHtml(editing, settings.seat !== "second");
  editorPaletteEl.innerHTML = paletteHtml(tool);
  editorRulesEl.innerHTML = rulesHtml(editing);
  if (rewriteFen) editorFenEl.value = fenOf(editing);
  const found = problems(editing);
  editorProblemsEl.innerHTML = problemsHtml(found);
  editorUseEl.disabled = found.length > 0;
  editorCurrentEl.hidden = entry.key !== "chess" || (moves.length === 0 && startFen === null);
}
detailsToggle.addEventListener("click", () => {
  showDetails = !showDetails;
  try {
    localStorage.setItem(DETAILS_KEY, showDetails ? "1" : "0");
  } catch {
    // Not being able to remember the choice is not a reason to refuse it.
  }
  render();
});
window.addEventListener("hashchange", applyRoute);

/**
 * Open the new-game sheet over the board.
 *
 * Rendered here and not in render(): render runs on every engine message, and
 * re-rendering the form would throw away a choice the player is halfway through.
 */
function openSheet(cancellable: boolean): void {
  sheetCancellable = cancellable;
  drawSheet(settings);
  sheetOpen = true;
  sheetEl.hidden = false;
  // Everything behind the sheet stops taking focus and clicks until it closes.
  for (const element of document.querySelectorAll<HTMLElement>(".play-head, .layout")) {
    element.inert = true;
  }
  sheetEl.querySelector<HTMLButtonElement>("button[type=submit]")?.focus();
}

function drawSheet(chosen: Settings): void {
  sheetEl.innerHTML = sheetHtml({
    entry,
    settings: chosen,
    networks: (networks.get(entry.key) ?? []).map((n) => ({ file: n.file, label: n.label })),
    ratings,
    cancellable: sheetCancellable,
    start: entry.key === "chess" ? { fen: customStart } : undefined,
  });
}

/**
 * Redraw an open sheet - when ratings arrive after it opened, say - keeping
 * whatever the player has already picked in it rather than the saved settings.
 */
function refreshSheet(): void {
  const form = sheetEl.querySelector<HTMLFormElement>("form");
  if (!sheetOpen || !form) return;
  const focused = document.activeElement === sheetEl.querySelector("button[type=submit]");
  drawSheet(readSettings(new FormData(form), settings));
  if (focused) sheetEl.querySelector<HTMLButtonElement>("button[type=submit]")?.focus();
}

function closeSheet(resume = true): void {
  sheetOpen = false;
  sheetEl.hidden = true;
  for (const element of document.querySelectorAll<HTMLElement>(".play-head, .layout")) {
    element.inert = false;
  }
  if (!resume) return;
  // Closing without choosing still starts the game that was waiting.
  render();
  advance();
}

function startGame(chosen: Settings): void {
  settings = chosen;
  startFen = entry.key === "chess" ? customStart : null;
  simulations = LEVELS[settings.level].simulations;
  humanFirst = resolveSeat(settings.seat);
  const next = networks.get(entry.key)?.find((n) => n.file === settings.network);
  if (next && next.file !== network?.file) {
    // A different network is a different opponent: load it, then play.
    network = next;
    cancelSearch();
    moves = [];
    viewing = null;
    report = null;
    pending = null;
    loadEngine();
    return;
  }
  reset();
}

/** The moves up to the position on the board, one array per point, so views can cache on it. */
let shownMoves: { moves: number[]; ply: number; slice: number[] } = { moves, ply: 0, slice: [] };

/**
 * What the board is showing, for the views: by default the position being
 * looked at, or `ply` to ask about another - the analysis describes the engine's
 * latest search, which belongs to the game as it stands, not to a move in the past.
 */
function context(ply = shownPly()) {
  if (shownMoves.moves !== moves || shownMoves.ply !== ply) {
    shownMoves = { moves, ply, slice: ply === moves.length ? moves : moves.slice(0, ply) };
  }
  const looking = ply !== moves.length;
  const locked = looking || thinking || finished() || !humanToMove() || !ready;
  const { states } = positions();
  return {
    game,
    state: states[ply],
    previous: ply > 0 ? states[ply - 1] : null,
    moves: shownMoves.slice,
    humanFirst,
    locked,
    pending: looking ? null : pending,
  };
}

// ------------------------------------------------------------------- screens

function renderCards(): void {
  cardsEl.innerHTML = LADDER.map((item) => {
    const available = trained.has(item.key);
    const tag = available ? "a" : "div";
    const href = available ? ` href="#/play/${item.key}"` : "";
    return `<${tag} class="card${available ? "" : " unavailable"}"${href} data-game="${item.key}">
      <span class="card-icon">${iconFor(item.key)}</span>
      <span class="card-text">
        <span class="card-title">${item.title}${
          available ? "" : `<span class="soon">soon</span>`
        }</span>
        <span class="card-blurb">${item.blurb}</span>
      </span>
    </${tag}>`;
  }).join("");
}

const REPO = "https://github.com/rtarik/caissa";

/** A link to a file in the repository, so a claim can be checked. */
function code(path: string, label = path): string {
  return `<a class="code-link" href="${REPO}/blob/main/${path}">${label}</a>`;
}

function renderAbout(): void {
  const generations = new Map(
    [...networks.entries()].map(([key, list]) => [key, list[0]?.generation ?? null]),
  );

  aboutEl.innerHTML = `
    <div class="prose">
      <h1>How it works</h1>
      <p class="lede">
        Six board games, and one opponent that taught itself to play all of them.
      </p>
      <p>
        Nothing here was told how to play. Each game's engine was given the rules,
        a way to see the board, and no advice at all: no openings, no tactics, no
        games by people. It got good by playing itself and keeping what worked.
        This page is a walk through how that is done, and where each game made it
        harder.
      </p>

      <h2>1. What you are playing against</h2>
      <p>
        Two pieces, and it helps to keep them apart, because they fail in
        different ways.
      </p>
      <p>
        The <strong>network</strong> is a small neural net (a few hundred thousand
        numbers, ${code("src/caissa/network.py", "network.py")}). Shown a position,
        it answers two questions at once: which moves look worth considering, and
        who is winning. It answers instantly, and it is often wrong. Playing at
        <em>Beginner</em> is playing the network alone, with no thinking on top.
      </p>
      <p>
        The <strong>search</strong> is what turns those hunches into a move
        (${code("src/caissa/mcts.py", "mcts.py")}). It plays out promising lines
        one at a time, a few hundred times, keeping a count of how often each move
        was explored and how those lines turned out. Moves the network likes get
        looked at first; moves that keep working get looked at more. When the
        budget runs out, the move that was explored most is played.
      </p>
      <p>
        The difficulty setting is that budget, and nothing else. It is worth a lot:
        in Reversi, going from 50 lines to 200 is worth about as much as a large
        jump in playing strength, while the network itself never changes.
      </p>

      <h2>2. How it learned</h2>
      <p>A loop, repeated until the network stops improving.</p>
      <ol class="steps">
        <li>
          <strong>Play itself.</strong> A few hundred games, both sides the same
          network, with the search running on every move
          (${code("src/caissa/selfplay.py", "selfplay.py")}). Early moves are
          picked with deliberate randomness, or every game would be the same game.
        </li>
        <li>
          <strong>Write down two answers per position.</strong> What the search
          explored, and how the game actually ended for whoever was to move there.
        </li>
        <li>
          <strong>Train on them.</strong> The network is adjusted to expect what
          the search found, and to predict how games end
          (${code("src/caissa/train.py", "train.py")}).
        </li>
      </ol>
      <p>
        The whole thing rests on one idea: <strong>searching is better than
        guessing</strong>. The search, using the network, finds better moves than
        the network alone would play, so training the network to imitate the search
        makes it better, which makes the next search better, and round it goes.
        AlphaZero's recipe, on one laptop instead of a data centre.
      </p>
      <p>
        It can also fail, and it did here. If the network's sense of who is winning
        becomes confident faster than it becomes accurate, the search starts
        believing it over its own findings, and the loop teaches the network to
        imitate a worse teacher every round. Every number inside training keeps
        improving while the agent gets weaker, which is why strength is measured
        from outside, by playing old versions against new ones
        (${code("src/caissa/arena.py", "arena.py")}).
      </p>

      <h2>3. The six games, and what each one changed</h2>
      <p>
        The engine is the same for all of them: it never mentions a game by name,
        and everything it needs sits behind one interface
        (${code("src/caissa/games/base.py", "games/base.py")}). Each game was added
        because it breaks an assumption the previous ones let stand.
      </p>
      <div class="game-notes">
        ${LADDER.map((item) => `<section class="game-note">
          <h3>
            <span class="ladder-icon" data-game="${item.key}">${iconFor(item.key)}</span>
            ${item.title}
          </h3>
          <p>${item.detail}</p>
          <dl class="facts">
            <div><dt>Board</dt><dd>${item.board}</dd></div>
            <div><dt>Moves to choose from</dt><dd>${item.moves}</dd></div>
            <div><dt>How it learned</dt><dd>${item.learned}</dd></div>
            ${generations.get(item.key)
              ? `<div><dt>Version playing here</dt><dd>generation ${generations.get(item.key)}</dd></div>`
              : ""}
          </dl>
          <p class="rules-link">Rules: ${code(`src/caissa/games/${item.key}.py`, `${item.key}.py`)}</p>
        </section>`).join("")}
      </div>

      <h2>4. Chess is the exception</h2>
      <p>
        Every other game here started from nothing. Chess did not, and the reason
        is arithmetic: AlphaZero learned chess from scratch using roughly a million
        times the computing power available here. Starting from zero was not a
        method choice, it was a budget.
      </p>
      <p>
        So chess began by copying people. Lichess publishes every game played on
        the site under a public licence, about 14 GB for a single month. January
        2020 was downloaded and filtered down to the games where
        <strong>both players were rated 2200 or above</strong>, which is a strong
        club player: 530,000 games, 39 million positions
        (${code("scripts/lichess.py", "lichess.py")}). The network was then trained
        to predict the move the human played and how the game ended
        (${code("scripts/imitate.py", "imitate.py")}), and it reaches about half of
        their moves exactly.
      </p>
      <p>
        That gives a chess engine with a real opening repertoire and no experience
        of its own. Self-play from there is unfinished work: the first attempt made
        it measurably weaker, for reasons that took a day to pin down and are
        written up in ${code("PLAN.md")} under <em>Chess, self-play stage 1</em>.
        The chess opponent you can play is the imitation network, so it plays a bit
        like the people it learned from.
      </p>
      <h3 class="prose-sub">How strong is it?</h3>
      <p>
        Each level played Stockfish, the strongest open-source engine, told to play
        at a chosen strength: 1320 up to 2500, both colours, 24 games at every step
        (${code("scripts/stockfish.py", "stockfish.py")}). One rating per level is then
        fitted to all of its results at once, the rating that makes those scores most
        likely (${code("src/caissa/rating.py", "rating.py")}).
      </p>
      ${chessRatingsTable()}
      <p>
        Two honest caveats. Stockfish's strength setting is calibrated against other
        engines rather than people, so these sit on roughly the FIDE scale rather
        than on it. And the step from Strong to Master is worth about 100 points
        here, where the same two levels played against each other showed nearly
        400. The same network at two depths shares its blind spots, so the deeper
        search knows exactly where its twin will go wrong; an outside opponent does
        not make those tailored mistakes. Ratings measured inside one family of
        players stretch the gaps between them, which is why these came from
        outside it.
      </p>
      <p>
        One smaller difference worth naming: <strong>Gomoku</strong> was trained
        with a restriction, where stones could only be played next to existing
        ones. Without it, a search that knows nothing never stumbles into a
        finished game and there is nothing to learn from. That restriction is a
        training aid, not a rule, so it is switched off for the game you play, and
        the network turned out not to need it.
      </p>

      <h2>5. How strong are the four levels?</h2>
      <p>
        Every level plays the one below it, sixty games each, same network on both
        sides (${code("scripts/levels.py", "levels.py")}). The numbers are Elo, the
        gap a rating system would put between them, and they only compare levels
        <em>within</em> one game.
      </p>
      <div id="levels-table"><p class="teaches">Measuring…</p></div>
      <p>
        The spread is the interesting part. Search is worth far more in some games
        than others: a Reversi flip three moves ahead is invisible to the network
        and obvious to a search, while in Gomoku, with eighty-one places to put a
        stone, tripling the budget barely deepens anything.
      </p>

      <h2>6. On your device</h2>
      <p>
        Every move is computed on your device. Nothing you play is uploaded, and
        there is no server to upload it to. Each network is a few hundred kilobytes
        and runs in a background thread in your browser
        (${code("web/src/engine/worker.ts", "worker.ts")}); the rules exist twice,
        once in Python for training and once in TypeScript for playing, and both
        are checked against the same recorded positions so they cannot drift apart
        (${code("scripts/testvectors.py", "testvectors.py")}).
      </p>

      <h2>The code</h2>
      <p>
        The whole project is on GitHub, including a long write-up of what was
        tried, what was measured and what went wrong.
      </p>
      <p class="prose-links">
        <a class="quiet-link" href="${REPO}">github.com/rtarik/caissa</a>
        <a class="quiet-link" href="${REPO}/blob/main/PLAN.md">The plan and decision log</a>
      </p>
    </div>
  `;
}

/** Your saved chess games, newest first, each with a way out to other tools. */
function renderGames(): void {
  const games = loadGames(store).filter((record) => record.game === "chess");
  const when = (iso: string) => {
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? "" : date.toLocaleString(undefined, {
      day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
    });
  };
  const verdict = (record: GameRecord) => {
    if (record.result === "*") return ["unfinished", "Unfinished"];
    if (record.result === "1/2-1/2") return ["drawn", "Drawn"];
    const youWon = (record.result === "1-0") === record.humanWhite;
    return youWon ? ["won", "Won"] : ["lost", "Lost"];
  };

  const rows = games.map((record) => {
    const [tone, word] = verdict(record);
    const plies = record.moves.length;
    const details = [
      when(record.started),
      `${Math.ceil(plies / 2)} ${plies > 2 ? "moves" : "move"}`,
      record.termination,
    ].filter(Boolean).join(" · ");
    return `<li class="game-row" data-id="${record.id}">
      <span class="player-token piece">${pieceSvg("k", record.humanWhite ? "w" : "b")}</span>
      <div class="game-main">
        <div class="game-title">
          <span>You vs ${engineName(record)}</span>
          <span class="verdict ${tone}">${word}</span>
        </div>
        <div class="game-meta">${details}</div>
        <details class="game-moves"><summary>Moves</summary><p>${movesText(record)}</p></details>
      </div>
      <div class="game-actions">
        <button class="button small" data-game-action="copy">Copy PGN</button>
        <button class="button small" data-game-action="download">Download</button>
        <button class="button small ghost" data-game-action="delete">Delete</button>
      </div>
    </li>`;
  }).join("");

  gamesEl.innerHTML = `
    <div class="play-head">
      <a class="back" href="#/play/chess" aria-label="Back to chess" title="Back to chess">
        <svg viewBox="0 0 24 24" aria-hidden="true" class="chevron"><path d="M15 5l-7 7 7 7"/></svg>
      </a>
      <h1 class="play-title"><span class="play-icon">${iconFor("chess")}</span>Your chess games</h1>
    </div>
    <p class="games-note">
      Kept in this browser only. Download a game to open it in any analysis tool;
      every game here is saved as it is played, so unfinished ones are here too.
    </p>
    ${games.length ? `
      <div class="games-actions">
        <button class="button" data-games="all">Download all (${games.length})</button>
        <button class="button ghost" data-games="clear">Clear history</button>
      </div>
      <ol class="game-list">${rows}</ol>`
    : `<p class="games-empty">No games yet. Play a game of chess and it will appear here.</p>`}
  `;
}

/** The chess levels' measured ratings, as a table, once they have loaded. */
function chessRatingsTable(): string {
  const chessNetworks = networks.get("chess") ?? [];
  const measuredOn = chessNetworks.find((n) => ratings.get(n.file)?.size);
  const measured = measuredOn ? ratings.get(measuredOn.file) : undefined;
  if (!measuredOn || !measured?.size) return "";
  const rows = LEVELS.map((level) => {
    const rating = measured.get(level.label);
    return rating
      ? `<tr><td>${level.label}</td><td class="gain">about ${approximately(rating.rating)}</td>
         <td>${rating.low}–${rating.high}</td></tr>`
      : "";
  }).join("");
  return `<table class="levels">
    <thead><tr><th>Level</th><th>Rating</th><th>95% range</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
  <p class="teaches">Measured for the ${measuredOn.label} engine; the others are not rated yet.</p>`;
}

/** The measured ladder, fetched once and only when the page asks for it. */
let ladderTable: string | null = null;

async function fillLevels(): Promise<void> {
  const host = document.getElementById("levels-table");
  if (!host) return;
  if (ladderTable !== null) {
    host.innerHTML = ladderTable;
    return;
  }
  try {
    const data: {
      games: Record<string, { level: string; elo: number; score: number }[]>;
    } = await fetch(asset("levels.json")).then((r) => r.json());

    const steps = LEVELS.slice(1).map((l) => l.label);
    const rows = LADDER.filter((item) => data.games[item.key]).map((item) => {
      const cells = data.games[item.key]
        .map((rung) => {
          // A clean sweep has no upper end to report, only a floor.
          const gain = rung.score >= 1 ? "off the scale" : `+${Math.round(rung.elo)}`;
          return `<td class="gain">${gain}</td>`;
        })
        .join("");
      return `<tr><td>${item.title}</td>${cells}</tr>`;
    });
    ladderTable = `<table class="levels">
      <thead><tr><th>Game</th>${steps
        .map((name, index) => `<th>${name}<br><span class="muted">over ${LEVELS[index].label}</span></th>`)
        .join("")}</tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>`;
    host.innerHTML = ladderTable;
  } catch {
    // The file is optional; without it the page simply says less.
    host.innerHTML = "";
  }
}

function render(): void {
  for (const [name, element] of Object.entries(screens)) {
    element.hidden = route.name !== name;
  }
  document.body.dataset.screen = route.name;

  if (route.name === "gallery") {
    document.body.removeAttribute("data-game");
    renderCards();
    return;
  }
  if (route.name === "about") {
    document.body.removeAttribute("data-game");
    renderAbout();
    void fillLevels();
    return;
  }
  if (route.name === "games") {
    document.body.dataset.game = route.key;
    renderGames();
    return;
  }

  document.body.dataset.game = entry.key;
  const ctx = context();
  playTitleEl.innerHTML = `<span class="play-icon">${iconFor(entry.key)}</span>${entry.title}`;

  const [engine, you] = playerCards();
  opponentEl.innerHTML = playerCardHtml(engine, tokenFor(entry.key, "engine", !humanFirst));
  youEl.innerHTML = playerCardHtml(you, tokenFor(entry.key, "you", humanFirst));

  boardEl.className = `board ${view.layout}${ctx.locked ? " locked" : ""}${viewing !== null ? " looking-back" : ""}`;
  boardEl.innerHTML = view.board(ctx);
  statusEl.innerHTML = statusText() + view.detail(ctx);
  renderMoves();

  detailsToggle.setAttribute("aria-expanded", String(showDetails));
  detailsToggle.textContent = showDetails ? "Hide engine details" : "Engine details";
  detailsEl.classList.toggle("open", showDetails);
  analysisEl.hidden = !showDetails;
  const notes = showDetails ? view.notes?.(ctx) : undefined;
  notesEl.hidden = !notes;
  if (showDetails) {
    analysisEl.innerHTML = analysisPanel();
    notesEl.innerHTML = notes ?? "";
  }
}

/**
 * The two players, as the cards above and below the board show them.
 *
 * Whose turn it is lives here now, on the card of the side to move, rather than
 * in a sentence under the board; the status line keeps what needs a sentence.
 */
function playerCards(): [PlayerCard, PlayerCard] {
  const level = LEVELS[settings.level];
  const rating = ratings.get(network?.file ?? "")?.get(level.label);
  const outcome = ready ? game.terminalValue(state()) : null;
  const over = resigned || outcome !== null;
  const humanWon = !resigned && over && outcome !== 0 && (outcome! > 0) === humanToMove();
  const engineWon = resigned || (over && outcome !== 0 && !humanWon);
  const yourTurn = ready && !over && humanToMove();

  const [first, second] = entry.seats ?? ["", ""];
  const yourSide = entry.seats
    ? (humanFirst ? first : second)
    : (humanFirst ? "Moving first" : "Moving second");

  return [
    {
      side: "engine",
      name: `Caissa <span class="player-level">${level.label}</span>`,
      detail: rating ? `Rated about ${approximately(rating.rating)}` : level.note,
      status: thinking ? "Thinking…" : "",
      active: ready && !over && !humanToMove(),
      winner: engineWon,
    },
    {
      side: "you",
      name: "You",
      detail: yourSide,
      status: yourTurn && pending === null ? "Your move" : "",
      active: yourTurn,
      winner: humanWon,
    },
  ];
}

/**
 * The move list, the stepping buttons and Undo.
 *
 * Every game has one: each is a list of moves, which is all a move list needs.
 * Chess names its moves in standard notation; the rest by column, square or line.
 */
function renderMoves(): void {
  const { states, labels } = positions();
  const plies: Ply[] = labels.map((label, index) => ({
    index, seat: game.toPlay(states[index]), label,
  }));
  const current = shownPly() - 1;
  const cell = (list: Ply[]) => list
    .map((ply) => `<button class="ply${ply.index === current ? " current" : ""}${
      ply.seat === humanSeat() ? " yours" : ""}" data-ply="${ply.index}">${ply.label}</button>`)
    .join("");

  moveListEl.innerHTML = plies.length === 0
    ? `<p class="move-empty">${humanFirst ? "Make your first move." : "Caissa moves first."}</p>`
    : rowsOf(plies, startFen ? parseFen(startFen)?.fullmove ?? 1 : 1).map((row) => `<div class="move-row">
        <span class="move-number">${row.number}</span>
        <span class="move-cell">${cell(row.cells[0])}</span>
        <span class="move-cell">${cell(row.cells[1])}</span>
      </div>`).join("");

  // Keep the move being shown in view, scrolling the list and never the page.
  const shown = moveListEl.querySelector<HTMLElement>(".ply.current");
  if (shown) {
    const top = shown.offsetTop - moveListEl.offsetTop;
    if (top < moveListEl.scrollTop || top > moveListEl.scrollTop + moveListEl.clientHeight - 28) {
      moveListEl.scrollTop = top - moveListEl.clientHeight / 2;
    }
  } else if (viewing === null) {
    moveListEl.scrollTop = moveListEl.scrollHeight;
  }

  moveWhereEl.textContent = viewing === null ? "" : `move ${viewing} of ${moves.length}`;
  const at = shownPly();
  for (const button of navEls) {
    const nav = button.dataset.nav;
    button.disabled = nav === "first" || nav === "back" ? at === 0 : at === moves.length;
  }
  undoEl.disabled = undoPoint() === null;

  chessActionsEl.hidden = entry.key !== "chess";
  if (entry.key === "chess") {
    const exportable = moves.length > 0 && yourMoveMade();
    resignEl.disabled = !ready || finished() || moves.length === 0;
    copyEl.disabled = !exportable;
    downloadEl.disabled = !exportable;
    const saved = loadGames(store).filter((record) => record.game === "chess").length;
    gamesLinkEl.textContent = saved ? `Your games (${saved})` : "Your games";
  }
}

/** Whether you have played a move in this game yet: before that there is nothing to keep. */
function yourMoveMade(): boolean {
  const { states } = positions();
  return moves.some((_, index) => game.toPlay(states[index]) === humanSeat());
}

/** What needs a sentence: the result, a prompt, a pass, a bonus move, a problem. */
function statusText(): string {
  if (error) return `<span class="muted">The engine didn't load: ${error}</span>`;
  if (!ready) return `<span class="muted">Loading ${entry.title}…</span>`;
  if (viewing !== null) {
    return `<span class="muted">Looking back at move ${viewing} of ${moves.length}.</span>
      <button class="quiet-link" data-nav-inline="last">Back to the game</button>`;
  }

  if (resigned) return `<span class="result">You resigned.</span>`;
  const outcome = game.terminalValue(state());
  if (outcome !== null) {
    if (outcome === 0) {
      const why = entry.key === "chess" ? chessOutcome(state() as ChessState, humanFirst, false).termination : "";
      return `<span class="result">Drawn${why ? ` by ${why}` : ""}.</span>`;
    }
    // terminalValue is for the player to move: +1 means they are ahead.
    const humanWon = outcome > 0 === humanToMove();
    return humanWon
      ? `<span class="result">You win.</span>`
      : `<span class="result">Caissa wins.</span>`;
  }
  if (thinking && lastMoveKeptTurn()) {
    return `<span class="muted">Caissa closed a box and moves again.</span>`;
  }
  if (humanToMove() && mustPass()) return `<span class="muted">No legal move, so you pass.</span>`;
  if (humanToMove() && pending !== null) {
    return `<span>${view.prompt?.(context()) ?? "Now finish the move."}</span>`;
  }
  if (humanToMove() && lastMoveKeptTurn()) {
    return `<span>Box closed, so it's your move again.</span>`;
  }
  return "";
}

function analysisPanel(): string {
  if (!report) {
    return `<h2>Analysis</h2><p class="teaches">${
      ready ? "Appears after the engine's first move." : engineInfo || "Loading…"
    }</p>`;
  }
  // The engine reports its root value for itself; flipped so the bar always
  // reads from the human's point of view.
  const percent = ((-report.value + 1) / 2) * 100;

  return `
    <h2>Analysis</h2>
    <div class="evalbar"><span style="width:${percent.toFixed(1)}%"></span></div>
    <div class="meta"><span>You ${percent.toFixed(0)}%</span><span>Engine ${(100 - percent).toFixed(0)}%</span></div>
    ${view.visits(report.visits, context(moves.length))}
    <div class="meta">
      <span>${report.simulations > 0 ? `${report.simulations} simulations` : "Network only, no search"}</span>
      <span>${Math.round(report.ms)} ms</span>
    </div>
    <div class="meta"><span>${engineInfo}</span></div>
  `;
}

async function start(): Promise<void> {
  try {
    const index: ModelIndex = await fetch(asset("models/index.json")).then((r) => r.json());
    networks = networksByGame(index.models);
    trained = new Set(networks.keys());
  } catch {
    // No index: fall back to offering the games that ship a view, so a missing
    // file degrades to "try it and see" rather than an empty menu.
    trained = new Set(Object.keys(VIEWS));
  }
  const requested = parseRoute(location.hash);
  const key = requested.name === "play" ? requested.key : null;
  entry = LADDER.find((item) => item.key === key && trained.has(item.key))
    ?? LADDER.find((item) => trained.has(item.key))
    ?? LADDER[0];
  network = networks.get(entry.key)?.[0] ?? null;
  game = createGame(entry.key);
  view = VIEWS[entry.key];
  settings = { ...settings, network: network?.file ?? null };
  void loadRatings();
  render();
  if (requested.name === "play") {
    route = { name: "play", key: entry.key };
    render();
    loadEngine();
    openSheet(false);
  } else {
    applyRoute();
  }
}

/**
 * Measured ratings, where they exist. Optional: without the file the level
 * cards simply say less, and nothing waits for it.
 */
async function loadRatings(): Promise<void> {
  try {
    const data: {
      networks: Record<string, { levels: { label: string; rating: number; low: number; high: number }[] }>;
    } = await fetch(asset("ratings.json")).then((r) => r.json());
    ratings = new Map(Object.entries(data.networks).map(([file, measured]) => [
      file,
      new Map(measured.levels.map((l) => [l.label, { rating: l.rating, low: l.low, high: l.high }])),
    ]));
    // Both the play screen and the guide show them; the gallery does not.
    if (route.name !== "gallery") render();
    refreshSheet();
  } catch {
    ratings = new Map();
  }
}

render();
void start();
