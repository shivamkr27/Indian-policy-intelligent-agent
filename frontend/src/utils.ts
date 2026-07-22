/**
 * crypto.randomUUID() only exists in secure contexts (HTTPS, or localhost
 * during dev) — plain HTTP on a non-localhost origin (e.g. an IP address,
 * like the current OCI deployment) doesn't have it, and calling it throws
 * "crypto.randomUUID is not a function" at module-eval time, blanking the
 * whole app. This falls back to a manual UUID-v4-shaped generator that
 * works everywhere. Not cryptographically strong, but this ID is only ever
 * used as a client-side session/device identifier, not a security credential.
 */
export function generateId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
