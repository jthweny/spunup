// Loaded before every test file (node --import): tests never touch the real ~/.spunup.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
process.env.SPUNUP_HOME = fs.mkdtempSync(path.join(os.tmpdir(), "spunup-test-"));
// ...and never the system keychain, which every install shares.
process.env.SPUNUP_NO_KEYCHAIN = "1";
