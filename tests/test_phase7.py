"""Tests for Phase 7: Desktop App (FastAPI Sidecar + React Frontend)."""

import unittest
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class TestFastAPIServer(unittest.TestCase):
    """Test the FastAPI sidecar server."""

    def test_server_importable(self):
        """Test that the server module can be imported."""
        sys.path.insert(0, os.path.join(_project_root, "python"))
        try:
            from server import app
            self.assertIsNotNone(app)
        except ImportError:
            self.skipTest("FastAPI not installed or server module not found")

    def test_server_has_rest_endpoints(self):
        """Test that the server has the expected REST endpoints."""
        sys.path.insert(0, os.path.join(_project_root, "python"))
        try:
            from server import app
            paths = [route.path for route in app.routes if hasattr(route, 'path')]
            self.assertIn("/api/hardware", paths)
            self.assertIn("/api/models", paths)
            self.assertIn("/api/probe", paths)
            self.assertIn("/api/train", paths)
            self.assertIn("/api/evaluate", paths)
        except ImportError:
            self.skipTest("FastAPI not installed")

    def test_server_has_websocket(self):
        """Test that the server has a WebSocket endpoint."""
        sys.path.insert(0, os.path.join(_project_root, "python"))
        try:
            from server import app
            ws_routes = [r for r in app.routes if hasattr(r, 'path') and r.path == "/ws"]
            self.assertGreater(len(ws_routes), 0, "WebSocket /ws endpoint not found")
        except ImportError:
            self.skipTest("FastAPI not installed")


class TestReactFrontend(unittest.TestCase):
    """Test that React frontend files exist."""

    def test_app_tsx_exists(self):
        path = os.path.join(_project_root, "src", "App.tsx")
        self.assertTrue(os.path.exists(path), "src/App.tsx not found")

    def test_main_tsx_exists(self):
        path = os.path.join(_project_root, "src", "main.tsx")
        self.assertTrue(os.path.exists(path), "src/main.tsx not found")

    def test_index_css_exists(self):
        path = os.path.join(_project_root, "src", "index.css")
        self.assertTrue(os.path.exists(path), "src/index.css not found")

    def test_all_screens_exist(self):
        screens = [
            "ModelSetup.tsx", "DomainSetup.tsx", "ProbingProgress.tsx",
            "ReviewDashboard.tsx", "FormatExport.tsx", "TrainEvaluate.tsx",
            "Stats.tsx",
        ]
        for screen in screens:
            path = os.path.join(_project_root, "src", "screens", screen)
            self.assertTrue(os.path.exists(path), f"src/screens/{screen} not found")

    def test_all_components_exist(self):
        components = ["Sidebar.tsx", "StatusBar.tsx", "Caduceus.tsx"]
        for comp in components:
            path = os.path.join(_project_root, "src", "components", comp)
            self.assertTrue(os.path.exists(path), f"src/components/{comp} not found")

    def test_hooks_exist(self):
        hooks_dir = os.path.join(_project_root, "src", "hooks")
        self.assertTrue(os.path.isdir(hooks_dir), "src/hooks/ directory not found")

    def test_lib_files_exist(self):
        for f in ["theme.ts", "api.ts"]:
            path = os.path.join(_project_root, "src", "lib", f)
            self.assertTrue(os.path.exists(path), f"src/lib/{f} not found")


class TestThemeColors(unittest.TestCase):
    """Test that the Hermes theme colors are correctly defined."""

    def test_theme_file_exists(self):
        path = os.path.join(_project_root, "src", "lib", "theme.ts")
        self.assertTrue(os.path.exists(path))

    def test_theme_has_hermes_colors(self):
        path = os.path.join(_project_root, "src", "lib", "theme.ts")
        with open(path, "r") as f:
            content = f.read()
        # Check for key Hermes theme colors
        self.assertIn("#041C1C", content)  # Background
        self.assertIn("#FFD700", content)  # Accent gold
        self.assertIn("#FFF8DC", content)  # Text primary
        self.assertIn("#CD7F32", content)  # Border bronze


class TestTauriConfig(unittest.TestCase):
    """Test Tauri configuration files."""

    def test_tauri_conf_exists(self):
        path = os.path.join(_project_root, "src-tauri", "tauri.conf.json")
        self.assertTrue(os.path.exists(path))

    def test_cargo_toml_exists(self):
        path = os.path.join(_project_root, "src-tauri", "Cargo.toml")
        self.assertTrue(os.path.exists(path))

    def test_main_rs_exists(self):
        path = os.path.join(_project_root, "src-tauri", "src", "main.rs")
        self.assertTrue(os.path.exists(path))

    def test_tauri_conf_has_window_config(self):
        import json
        path = os.path.join(_project_root, "src-tauri", "tauri.conf.json")
        with open(path, "r") as f:
            conf = json.load(f)
        windows = conf.get("app", {}).get("windows", [])
        self.assertGreater(len(windows), 0)
        self.assertEqual(windows[0].get("title"), "MindForge")


class TestPackageFiles(unittest.TestCase):
    """Test package configuration files."""

    def test_package_json_exists(self):
        path = os.path.join(_project_root, "package.json")
        self.assertTrue(os.path.exists(path))

    def test_package_json_has_react(self):
        import json
        path = os.path.join(_project_root, "package.json")
        with open(path, "r") as f:
            pkg = json.load(f)
        deps = pkg.get("dependencies", {})
        self.assertIn("react", deps)

    def test_requirements_desktop_exists(self):
        path = os.path.join(_project_root, "python", "requirements.txt")
        self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
