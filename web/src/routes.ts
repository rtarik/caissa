/**
 * Which screen the URL is asking for.
 *
 * Kept apart from `main.ts`, which starts a worker and rewrites the document the
 * moment it is imported, so the rules about what a URL means can be read - and
 * tested - on their own.
 */
import { LADDER } from "./games/ladder";

export type Route =
  | { name: "gallery" }
  | { name: "play"; key: string }
  | { name: "about" };

/** The link that reaches a screen. */
export function href(route: Route): string {
  if (route.name === "about") return "#/how-it-works";
  if (route.name === "play") return `#/play/${route.key}`;
  return "#/";
}

/**
 * Anything unrecognised is the gallery, deliberately.
 *
 * A stale bookmark, a game that has been renamed, a hand-typed hash: all of them
 * land somewhere that works rather than on an empty screen or an error.
 */
export function parseRoute(hash: string): Route {
  const path = hash.replace(/^#\/?/, "");
  if (path === "how-it-works") return { name: "about" };
  if (path.startsWith("play/")) {
    const key = path.slice("play/".length);
    if (LADDER.some((item) => item.key === key)) return { name: "play", key };
  }
  return { name: "gallery" };
}
