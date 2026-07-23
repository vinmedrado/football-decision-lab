import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(root, "site");
const output = path.join(root, "dist");

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await cp(source, output, { recursive: true });

const publicAssets = {
  "/": {
    content: await readFile(path.join(source, "index.html"), "utf8"),
    type: "text/html; charset=utf-8"
  },
  "/index.html": {
    content: await readFile(path.join(source, "index.html"), "utf8"),
    type: "text/html; charset=utf-8"
  },
  "/styles.css": {
    content: await readFile(path.join(source, "styles.css"), "utf8"),
    type: "text/css; charset=utf-8"
  },
  "/app.js": {
    content: await readFile(path.join(source, "app.js"), "utf8"),
    type: "text/javascript; charset=utf-8"
  },
  "/data/snapshot.json": {
    content: await readFile(path.join(source, "data", "snapshot.json"), "utf8"),
    type: "application/json; charset=utf-8"
  }
};

const workerSource = `const ASSETS = ${JSON.stringify(publicAssets)};

const SECURITY_HEADERS = {
  "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY"
};

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const asset = ASSETS[url.pathname] || ASSETS["/"];
    return new Response(request.method === "HEAD" ? null : asset.content, {
      headers: { ...SECURITY_HEADERS, "Content-Type": asset.type }
    });
  }
};
`;

await mkdir(path.join(output, "server"), { recursive: true });
await mkdir(path.join(output, ".openai"), { recursive: true });
await writeFile(path.join(output, "server", "index.js"), workerSource, "utf8");
await cp(
  path.join(root, ".openai", "hosting.json"),
  path.join(output, ".openai", "hosting.json")
);

console.log(`Static build created at ${output}`);
