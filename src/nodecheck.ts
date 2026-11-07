/** Imported first by the CLI, so an old Node says so plainly instead of failing somewhere inside a library. */
const major = Number(process.versions.node.split(".")[0]);
if (major < 22) {
  console.error(`spunup needs Node.js 22 or newer (this is ${process.version}). Update Node.js (nodejs.org), or use the one-line installer, which brings its own.`);
  process.exit(1);
}
export {};
