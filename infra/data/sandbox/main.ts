/**
 * Patched version of jsr:@langchain/pyodide-sandbox@0.0.4/main.ts
 *
 * FIX: Removed the `replace(/\\n/g, "\n")` on the `-c` code path (line ~388
 * in the original). That regex converted *every* literal backslash-n sequence
 * into a real newline, which broke Python strings containing `\n` escape
 * sequences (e.g. f-strings like `f"hello\nworld"`).
 *
 * When code is passed via `asyncio.create_subprocess_exec` (Python list args)
 * newlines are already real newlines — no post-processing is needed.
 *
 * Source: https://jsr.io/@langchain/pyodide-sandbox/0.0.4/main.ts
 *
 * FIX 2: Changed npm:pyodide from ^0.27.4 to 0.29.0 to align with the
 * locally-installed Pyodide distribution (/app/pyodide/).  The 0.27.x
 * package index ships urllib3 2.2.3 / chardet 6.x / charset-normalizer 3.3.2,
 * which triggers a RequestsDependencyWarning in the `requests` package.
 * Pyodide 0.29.0 ships compatible versions (requests 2.32.4, urllib3 2.5.0,
 * charset-normalizer 3.4.3, chardet 5.2.0).
 *
 * FIX 3: Added --stdin flag support to read Python code from stdin instead of
 * CLI arguments. This avoids Linux kernel's MAX_ARG_STRLEN 128KB limit on
 * individual command-line arguments when passing large code payloads.
 * Issue: EliteaAI/elitea_issues#6249
 *
 * FIX 4: Loads Pyodide from <SANDBOX_BASE>/pyodide, the local distribution, with the Node globals it expects,
 * and takes the path and argument helpers from node: built-ins, so the sandbox loads no npm or JSR module.
 */

import { createRequire } from "node:module";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const nodeGlobals = globalThis as any;
nodeGlobals.require ??= createRequire(import.meta.url);
nodeGlobals.__filename ??= fileURLToPath(import.meta.url);
nodeGlobals.__dirname ??= fileURLToPath(new URL(".", import.meta.url));

const sandboxBase = Deno.env.get("SANDBOX_BASE") ?? fileURLToPath(new URL("../../", import.meta.url));
const { loadPyodide } = await import(pathToFileURL(join(sandboxBase, "pyodide", "pyodide.mjs")).href);


const pkgVersion = "0.0.4-patched";

// Python environment preparation code
// This code was adapted from
// https://github.com/alexmojaki/pyodide-worker-runner/blob/master/lib/pyodide_worker_runner.py
const prepareEnvCode = `
import datetime
import importlib
import json
import sys
from typing import Union, TypedDict, List, Any, Callable, Literal

try:
    from pyodide.code import find_imports  # noqa
except ImportError:
    from pyodide import find_imports  # noqa

import pyodide_js  # noqa

sys.setrecursionlimit(400)


class InstallEntry(TypedDict):
    module: str
    package: str


def find_imports_to_install(imports: list[str]) -> list[InstallEntry]:
    """
    Given a list of module names being imported, return a list of dicts
    representing the packages that need to be installed to import those modules.
    The returned list will only contain modules that aren't already installed.
    Each returned dict has the following keys:
      - module: the name of the module being imported
      - package: the name of the package that needs to be installed
    """
    try:
        to_package_name = pyodide_js._module._import_name_to_package_name.to_py()
    except AttributeError:
        to_package_name = pyodide_js._api._import_name_to_package_name.to_py()

    to_install: list[InstallEntry] = []
    for module in imports:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            to_install.append(
                dict(
                    module=module,
                    package=to_package_name.get(module, module),
                )
            )
    return to_install


async def install_imports(
    source_code_or_imports: Union[str, list[str]],
    additional_packages: list[str] = [],
    message_callback: Callable[
          [
              Literal[
                "failed",
              ],
              Union[InstallEntry, list[InstallEntry]],
          ],
          None,
      ] = lambda event_type, data: None,
) -> List[InstallEntry]:
    if isinstance(source_code_or_imports, str):
        try:
            imports: list[str] = find_imports(source_code_or_imports)
        except SyntaxError:
            return
    else:
        imports: list[str] = source_code_or_imports

    to_install = find_imports_to_install(imports)
    # Merge with additional packages
    for package in additional_packages:
        if package not in to_install:
            to_install.append(dict(module=package, package=package))

    if to_install:
        try:
            import micropip  # noqa
        except ModuleNotFoundError:
            await pyodide_js.loadPackage("micropip")
            import micropip  # noqa

        # Pin chardet < 6 — requests 2.x only supports chardet 3.x-5.x
        _PINNED_VERSIONS = {"chardet": "chardet<6"}

        for entry in to_install:
            try:
                pkg_spec = _PINNED_VERSIONS.get(entry["package"], entry["package"])
                await micropip.install(pkg_spec)
            except Exception as e:
                message_callback("failed", entry["package"])
                break # Fail fast
    return to_install


def load_session_bytes(session_bytes: bytes) -> list[str]:
    """Load the session module."""
    import dill
    import io

    buffer = io.BytesIO(session_bytes.to_py())
    dill.session.load_session(filename=buffer)


def dump_session_bytes() -> bytes:
    """Dump the session module."""
    import dill
    import io

    buffer = io.BytesIO()
    dill.session.dump_session(filename=buffer)
    return buffer.getvalue()


def robust_serialize(obj):
    """Recursively converts an arbitrary Python object into a JSON-serializable structure.

    The function handles:
      - Primitives: str, int, float, bool, None are returned as is.
      - Lists and tuples: Each element is recursively processed.
      - Dictionaries: Keys are converted to strings (if needed) and values are recursively processed.
      - Sets: Converted to lists.
      - Date and datetime objects: Converted to their ISO format strings.
      - For unsupported/unknown objects, a dictionary containing a 'type'
        indicator and the object's repr is returned.
    """
    # Base case: primitives that are already JSON-serializable
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # Process lists or tuples recursively.
    if isinstance(obj, (list, tuple)):
        return [robust_serialize(item) for item in obj]

    # Process dictionaries.
    if isinstance(obj, dict):
        # Convert keys to strings if necessary and process values recursively.
        return {str(key): robust_serialize(value) for key, value in obj.items()}

    # Process sets by converting them to lists.
    if isinstance(obj, (set, frozenset)):
        return [robust_serialize(item) for item in obj]

    # Process known datetime objects.
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()

    # Fallback: for objects that are not directly serializable,
    # return a dictionary with type indicator and repr.
    return {"type": "not_serializable", "repr": repr(obj)}


def dumps(result: Any) -> str:
    """Get the result of the session."""
    result = robust_serialize(result)
    return json.dumps(result)
`;

interface SessionMetadata {
  created: string;
  lastModified: string;
  packages: string[];
}

interface PyodideResult {
  success: boolean;
  result?: any;
  stdout?: string[];
  stderr?: string[];
  error?: string;
  jsonResult?: string;
  sessionBytes?: Uint8Array;
  sessionMetadata?: SessionMetadata;
}

async function initPyodide(pyodide: any): Promise<void> {
  const sys = pyodide.pyimport("sys");
  const pathlib = pyodide.pyimport("pathlib");

  const dirPath = "/tmp/pyodide_worker_runner/";
  sys.path.append(dirPath);
  pathlib.Path(dirPath).mkdir();
  pathlib.Path(dirPath + "prepare_env.py").write_text(prepareEnvCode);
}

async function runPython(
  pythonCode: string,
  options: {
    stateful?: boolean;
    sessionBytes?: string;
    sessionMetadata?: string;
  }
): Promise<PyodideResult> {
  const output: string[] = [];
  const err_output: string[] = [];
  const originalLog = console.log;
  console.log = (...args: any[]) => {}

  try {
    const pyodide = await loadPyodide({
      stdout: (msg) => output.push(msg),
      stderr: (msg) => err_output.push(msg),
    })
    await pyodide.loadPackage(["micropip"], {
      messageCallback: () => {},
      errorCallback: (msg: string) => {
        output.push(`install error: ${msg}`)
      },
    });
    await initPyodide(pyodide);

    // Determine session directory
    let sessionMetadata: SessionMetadata;
    if (options.sessionMetadata) {
      sessionMetadata = JSON.parse(options.sessionMetadata);
    } else {
      sessionMetadata = {
        created: new Date().toISOString(),
        lastModified: new Date().toISOString(),
        packages: [],
      };
    };
    let sessionData: Uint8Array | null = null;

    if (options.sessionBytes && !options.sessionMetadata) {
      console.error("sessionMetadata is required when providing sessionBytes");
      return { success: false, error: "sessionMetadata is required when providing sessionBytes" };
    }

    // Import our prepared environment module
    const prepare_env = pyodide.pyimport("prepare_env");
    // Prepare additional packages to install (include dill)
    const defaultPackages = options.stateful ? ["dill"] : [];
    const additionalPackagesToInstall = options.sessionBytes
      ? [...new Set([...defaultPackages, ...sessionMetadata.packages])]
      : defaultPackages;

    let installErrors: string[] = []

    const installedPackages = await prepare_env.install_imports(
      pythonCode,
      additionalPackagesToInstall,
      (event_type: string, data: string) => {
        if (event_type === "failed") {
          installErrors.push(data)
        }
      }
    );

    if (installErrors.length > 0) {
      // Restore the original console.log function
      console.log = originalLog;
      return {
        success: false,
        error: `Failed to install required Python packages: ${installErrors.join(", ")}. ` +
          `This is likely because these packages are not available in the Pyodide environment. ` +
          `Pyodide is a Python runtime that runs in the browser and has a limited set of ` +
          `pre-built packages. You may need to use alternative packages that are compatible ` +
          `with Pyodide.`
      };
    }

    if (options.sessionBytes) {
      sessionData = Uint8Array.from(JSON.parse(options.sessionBytes));
      // Run session preamble
      await prepare_env.load_session_bytes(sessionData);
    }

    const packages = installedPackages.map((pkg: any) => pkg.get("package"));

    // Restore the original console.log function
    console.log = originalLog;
    // Run the Python code
    const rawValue = await pyodide.runPythonAsync(pythonCode);
    // Dump result to string
    const jsonValue = await prepare_env.dumps(rawValue);

    // Update session metadata with installed packages
    sessionMetadata.packages = [
      ...new Set([...sessionMetadata.packages, ...packages]),
    ];
    sessionMetadata.lastModified = new Date().toISOString();

    if (options.stateful) {
      // Save session state to sessionBytes
      sessionData = await prepare_env.dump_session_bytes() as Uint8Array;
    };
    // Return the result with stdout and stderr output
    const result: PyodideResult = {
      success: true,
      result: rawValue,
      jsonResult: jsonValue,
      stdout: output,
      stderr: err_output,
      sessionMetadata: sessionMetadata,
    };
    if (options.stateful && sessionData) {
      result["sessionBytes"] = sessionData;
    }
    return result;
  } catch (error: any) {
    return {
      success: false,
      error: error.message,
      stdout: output,
      stderr: err_output
    };
  }
}

// --bridge: line-delimited JSON over stdin/stdout so a running program can call tools in the worker (#7578).
const DEFAULT_BRIDGE_MAX_LINE_BYTES = 4 * 1024 * 1024;
const BRIDGE_FIRST_LINE_TIMEOUT_MS = 30000;
const bridgeEncoder = new TextEncoder();

function writeBridgeLine(message: any, maxLineBytes: number): boolean {
  const bytes = bridgeEncoder.encode(JSON.stringify(message) + "\n");
  if (bytes.length - 1 > maxLineBytes) {
    return false;
  }
  // Sync writes, so output printed by synchronous Python code streams instead of waiting for the next await.
  let offset = 0;
  while (offset < bytes.length) {
    offset += Deno.stdout.writeSync(bytes.subarray(offset));
  }
  return true;
}

function writeBridgeText(type: string, text: string, maxLineBytes: number): void {
  if (!writeBridgeLine({ type, text }, maxLineBytes)) {
    const keep = Math.floor(maxLineBytes / 8);
    writeBridgeLine({ type, text: `${text.slice(0, keep)}... [truncated: line over ${maxLineBytes} bytes]\n` }, maxLineBytes);
  }
}

async function* readBridgeLines(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let scanFrom = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      if (buffer) {
        yield buffer;
      }
      return;
    }
    buffer += value;
    let index;
    while ((index = buffer.indexOf("\n", scanFrom)) >= 0) {
      yield buffer.slice(0, index);
      buffer = buffer.slice(index + 1);
      scanFrom = 0;
    }
    scanFrom = buffer.length;
  }
}

function classifyBridgeError(error: any): string {
  const message = String(error?.message ?? error);
  if (error?.type === "MemoryError" || error instanceof RangeError ||
      /MemoryError|out of memory|memory access out of bounds/i.test(message)) {
    return "out_of_memory";
  }
  if (error?.type || error?.constructor?.name === "PythonError") {
    return "python_exception";
  }
  return "sandbox_crashed";
}

async function runBridge(maxLineBytes: number): Promise<void> {
  let finalSent = false;
  const sendFinal = (fields: Record<string, unknown>) => {
    if (finalSent) {
      return;
    }
    finalSent = true;
    const message = { type: "final", success: false, result: null, error: null, error_type: null, ...fields };
    if (!writeBridgeLine(message, maxLineBytes)) {
      writeBridgeLine({
        type: "final", success: false, result: null,
        error: `The program result is over the ${maxLineBytes}-byte message limit.`,
        error_type: "result_too_large",
      }, maxLineBytes);
    }
  };
  // Anything logged by JS must not corrupt the protocol on stdout.
  console.log = (...args: any[]) => console.error(...args);

  try {
    const lines = readBridgeLines(Deno.stdin.readable);
    let timer: number | undefined;
    const first = await Promise.race([
      lines.next(),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`stdin read timeout after ${BRIDGE_FIRST_LINE_TIMEOUT_MS}ms`)), BRIDGE_FIRST_LINE_TIMEOUT_MS);
      }),
    ]).finally(() => clearTimeout(timer));
    const codeMessage = first.done ? null : JSON.parse(first.value as string);
    if (!codeMessage || codeMessage.type !== "code" || typeof codeMessage.code !== "string" || !codeMessage.code.trim()) {
      throw new Error("The first bridge message must be {\"type\": \"code\", \"code\": \"...\"}.");
    }
    const pythonCode = codeMessage.code;

    const pyodide = await loadPyodide({
      stdout: (msg: string) => writeBridgeText("stdout", `${msg}\n`, maxLineBytes),
      stderr: (msg: string) => writeBridgeText("stderr", `${msg}\n`, maxLineBytes),
    });
    await initPyodide(pyodide);

    const pending = new Map<number, { resolve: (value: string) => void; reject: (error: Error) => void }>();
    let nextId = 1;
    let closed = false;
    pyodide.registerJsModule("elitea_bridge", {
      call: (name: string, argsJson: string) => {
        if (closed) {
          return Promise.reject(new Error("bridge closed"));
        }
        const id = nextId++;
        return new Promise<string>((resolve, reject) => {
          pending.set(id, { resolve, reject });
          if (!writeBridgeLine({ type: "call", id, name: String(name), args: String(argsJson ?? "{}") }, maxLineBytes)) {
            pending.delete(id);
            reject(new Error(`Tool call arguments are over the ${maxLineBytes}-byte message limit.`));
          }
        });
      },
    });

    (async () => {
      for await (const line of lines) {
        let message: any;
        try {
          message = JSON.parse(line);
        } catch (_error) {
          continue;
        }
        const waiter = message?.type === "result" ? pending.get(message.id) : undefined;
        if (!waiter) {
          continue;
        }
        pending.delete(message.id);
        if (message.ok) {
          waiter.resolve(typeof message.value === "string" ? message.value : JSON.stringify(message.value ?? null));
        } else {
          waiter.reject(new Error(String(message.error ?? "tool call failed")));
        }
      }
      closed = true;
      for (const waiter of pending.values()) {
        waiter.reject(new Error("bridge closed"));
      }
      pending.clear();
    })().catch((error: any) => console.error("bridge reader failed:", error?.message ?? error));

    const prepareEnv = pyodide.pyimport("prepare_env");
    const installErrors: string[] = [];
    await prepareEnv.install_imports(pythonCode, [], (eventType: string, data: string) => {
      if (eventType === "failed") {
        installErrors.push(data);
      }
    });
    if (installErrors.length > 0) {
      sendFinal({
        error: `Failed to install required Python packages: ${installErrors.join(", ")}.`,
        error_type: "python_exception",
      });
    } else {
      const rawValue = await pyodide.runPythonAsync(pythonCode);
      const jsonValue = await prepareEnv.dumps(rawValue);
      sendFinal({ success: true, result: JSON.parse(jsonValue) });
    }
  } catch (error: any) {
    sendFinal({ error: String(error?.message ?? error), error_type: classifyBridgeError(error) });
  }
  Deno.exit(0);
}

async function main(): Promise<void> {
  const { values: flags } = parseArgs({
    args: Deno.args,
    options: {
      code: { type: "string", short: "c" },
      file: { type: "string", short: "f" },
      "session-bytes": { type: "string", short: "b" },
      "session-metadata": { type: "string", short: "m" },
      help: { type: "boolean", short: "h", default: false },
      version: { type: "boolean", short: "V", default: false },
      stateful: { type: "boolean", short: "s", default: false },
      stdin: { type: "boolean", default: false },
      bridge: { type: "boolean", default: false },
      "max-line-bytes": { type: "string" },
    },
  });

  if (flags.help) {
    console.log(`
pyodide-sandbox ${pkgVersion}
Run Python code in a sandboxed environment using Pyodide

OPTIONS:
  -c, --code <code>            Python code to execute
  -f, --file <path>            Path to Python file to execute
      --stdin                  Read Python code from stdin
      --bridge                 Line-delimited JSON protocol with tool calls over stdin/stdout
      --max-line-bytes <n>     Per-line size cap in --bridge mode
  -s, --stateful <bool>        Use a stateful session
  -b, --session-bytes <bytes>  Session bytes
  -m, --session-metadata       Session metadata
  -h, --help                   Display help
  -V, --version                Display version
`);
    return;
  }

  if (flags.version) {
    console.log(pkgVersion)
    return
  }

  if (flags.bridge) {
    await runBridge(Number(flags["max-line-bytes"]) || DEFAULT_BRIDGE_MAX_LINE_BYTES);
    return;
  }

  const options = {
    code: flags.code,
    file: flags.file,
    stdin: flags.stdin,
    stateful: flags.stateful,
    sessionBytes: flags["session-bytes"],
    sessionMetadata: flags["session-metadata"],
  };

  if (!options.code && !options.file && !options.stdin) {
    console.error(
      "Error: You must provide Python code using -c/--code, -f/--file, or --stdin option.\nUse --help for usage information."
    );
    Deno.exit(1);
  }

  // Get Python code from file or command line argument
  let pythonCode = "";

  if (options.file) {
    try {
      // Resolve relative or absolute file path
      const filePath = options.file.startsWith("/")
        ? options.file
        : join(Deno.cwd(), options.file);
      pythonCode = await Deno.readTextFile(filePath);
    } catch (error: any) {
      console.error(`Error reading file ${options.file}:`, error.message);
      Deno.exit(1);
    }
  } else if (options.stdin) {
    // Read code from stdin - avoids Linux kernel's MAX_ARG_STRLEN 128KB limit
    // on individual CLI arguments when passing large code payloads
    pythonCode = await new Response(Deno.stdin.readable).text();
  } else {
    // FIX: Do NOT replace \\n with real newlines.
    // When code is passed via asyncio.create_subprocess_exec (Python list args),
    // newlines are already real newlines. The original replace(/\\n/g, "\n")
    // corrupted Python escape sequences like \n inside f-strings.
    pythonCode = options.code ?? "";
  }

  const result = await runPython(pythonCode, {
    stateful: options.stateful,
    sessionBytes: options.sessionBytes,
    sessionMetadata: options.sessionMetadata,
  });

  // Exit with error code if Python execution failed
  // Create output JSON with stdout, stderr, and result
  const outputJson = {
    stdout: result.stdout?.join('') || null,
    stderr: result.success ? (result.stderr?.join('') || null) : result.error || null,
    result: result.success ? JSON.parse(result.jsonResult || 'null') : null,
    success: result.success,
    sessionBytes: result.sessionBytes,
    sessionMetadata: result.sessionMetadata,
  };

  // Output as JSON to stdout
  console.log(JSON.stringify(outputJson));

  // Exit with error code if Python execution failed
  if (!result.success) {
    Deno.exit(1);
  }
}

// If this module is run directly
if (import.meta.main) {
  main().catch((err) => {
    console.error("Unhandled error:", err);
    Deno.exit(1);
  });
}
