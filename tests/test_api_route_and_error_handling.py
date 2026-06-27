"""Regression tests for API route ordering and frontend error formatting."""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(PROJECT_ROOT, "python")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)


class TestReviewRouteOrder(unittest.TestCase):
    """Ensure literal auto-review routes are registered before /api/review/{entry_id}."""

    @classmethod
    def setUpClass(cls):
        try:
            from server import app

            cls.app = app
        except ImportError:
            cls.app = None

    def setUp(self):
        if self.app is None:
            self.skipTest("FastAPI server module not available")

    def test_auto_review_routes_precede_parameterized_review_route(self):
        route_positions = {}

        for index, route in enumerate(self.app.routes):
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", set())
            for method in methods:
                route_positions[(method, path)] = index

        parameterized = route_positions[("POST", "/api/review/{entry_id}")]
        self.assertLess(route_positions[("POST", "/api/review/auto")], parameterized)
        self.assertLess(route_positions[("GET", "/api/review/auto/{job_id}")], parameterized)
        self.assertLess(route_positions[("POST", "/api/review/auto/entry/{entry_id}")], parameterized)


class TestFrontendApiErrors(unittest.TestCase):
    """Validate apiPost formats FastAPI validation errors readably."""

    def test_api_post_joins_fastapi_validation_messages(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not available")

        script = textwrap.dedent(
            f"""
            const fs = require("fs");
            const path = require("path");
            const vm = require("vm");

            const projectRoot = {json.dumps(PROJECT_ROOT)};
            const ts = require(path.join(projectRoot, "node_modules", "typescript"));
            const source = fs.readFileSync(path.join(projectRoot, "src", "lib", "api.ts"), "utf8");
            const compiled = ts.transpileModule(source, {{
              compilerOptions: {{
                module: ts.ModuleKind.CommonJS,
                target: ts.ScriptTarget.ES2020,
              }},
            }}).outputText;

            const sandbox = {{
              exports: {{}},
              fetch: async () => ({{
                ok: false,
                statusText: "Unprocessable Entity",
                json: async () => ({{
                  detail: [
                    {{ msg: "Input should be a valid integer" }},
                    {{ msg: "Field required" }},
                  ],
                }}),
              }}),
            }};

            vm.createContext(sandbox);
            vm.runInContext(compiled, sandbox);

            (async () => {{
              try {{
                await sandbox.exports.apiPost("/api/review/auto", {{}});
                console.error("apiPost unexpectedly resolved");
                process.exit(1);
              }} catch (error) {{
                const expected = "API /api/review/auto: Input should be a valid integer; Field required";
                if (error.message !== expected) {{
                  console.error(error.message);
                  process.exit(1);
                }}
              }}
            }})();
            """
        )

        result = subprocess.run(
            [node, "-e", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
