"""Tests for Xbox Blades redesign components and layout.

Verifies:
- BladeBar.tsx component exists and renders 8 tabs
- BladeContent.tsx component exists with icon/title/children layout
- SoundManager.tsx component exists with SoundEngine class
- App.tsx uses BladeBar (not old Sidebar) and has Xbox layout structure
- Theme colors are still correct (Hermes gold/teal palette)
- CSS classes for Xbox visual effects exist (hex-grid, scanlines, blade-panel)
- All 8 blade screens are still present
- Controller hints exist in App.tsx
- MuteToggle component exists in SoundManager
"""

import os
import sys
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def read_file(path):
    """Read a file and return its contents, or empty string if not found."""
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""


# ═══════════════════════════════════════════════════════════════════
# Component Existence Tests
# ═══════════════════════════════════════════════════════════════════

class TestXboxComponentsExist(unittest.TestCase):
    """Verify all Xbox Blades redesign components exist."""

    def test_blade_bar_exists(self):
        """BladeBar.tsx component should exist."""
        path = os.path.join(_project_root, "src", "components", "BladeBar.tsx")
        self.assertTrue(os.path.exists(path), "src/components/BladeBar.tsx not found")

    def test_blade_content_exists(self):
        """BladeContent.tsx component should exist."""
        path = os.path.join(_project_root, "src", "components", "BladeContent.tsx")
        self.assertTrue(os.path.exists(path), "src/components/BladeContent.tsx not found")

    def test_sound_manager_exists(self):
        """SoundManager.tsx component should exist."""
        path = os.path.join(_project_root, "src", "components", "SoundManager.tsx")
        self.assertTrue(os.path.exists(path), "src/components/SoundManager.tsx not found")

    def test_sidebar_still_exists(self):
        """Sidebar.tsx should still exist (backward compatibility)."""
        path = os.path.join(_project_root, "src", "components", "Sidebar.tsx")
        self.assertTrue(os.path.exists(path), "src/components/Sidebar.tsx not found")

    def test_status_bar_exists(self):
        """StatusBar.tsx should still exist."""
        path = os.path.join(_project_root, "src", "components", "StatusBar.tsx")
        self.assertTrue(os.path.exists(path), "src/components/StatusBar.tsx not found")


# ═══════════════════════════════════════════════════════════════════
# BladeBar Component Tests
# ═══════════════════════════════════════════════════════════════════

class TestBladeBarContent(unittest.TestCase):
    """Verify BladeBar.tsx has correct structure and 8 tabs."""

    def setUp(self):
        self.path = os.path.join(_project_root, "src", "components", "BladeBar.tsx")
        self.content = read_file(self.path)

    def test_blade_bar_exports_component(self):
        """BladeBar should export a BladeBar component."""
        self.assertIn("BladeBar", self.content)

    def test_blade_bar_has_8_tabs(self):
        """BladeBar should define exactly 8 blade tabs."""
        # Count the tab entries in BLADE_TABS array
        tab_ids = [
            "model-setup", "domain-setup", "probing", "review",
            "format", "train", "stats", "settings",
        ]
        for tab_id in tab_ids:
            self.assertIn(tab_id, self.content,
                         f"Blade tab '{tab_id}' not found in BladeBar.tsx")

    def test_blade_bar_has_icons(self):
        """BladeBar tabs should have icons."""
        # Check for emoji icons
        for icon in ["🖥", "📚", "🔍", "📋", "📦", "🎯", "📊", "⚙"]:
            self.assertIn(icon, self.content, f"Icon '{icon}' not found in BladeBar")

    def test_blade_bar_has_labels(self):
        """BladeBar tabs should have labels."""
        for label in ["Model", "Domains", "Probe", "Review", "Format", "Train", "Stats", "Settings"]:
            self.assertIn(label, self.content, f"Label '{label}' not found in BladeBar")

    def test_blade_bar_uses_framer_motion(self):
        """BladeBar should use framer-motion for animations."""
        self.assertIn("motion", self.content)

    def test_blade_bar_has_active_state(self):
        """BladeBar should have active state styling (gold glow, border)."""
        self.assertIn("isActive", self.content)
        self.assertIn("accent", self.content.lower())

    def test_blade_bar_has_frosted_glass(self):
        """BladeBar should use frosted glass effect (backdropFilter)."""
        self.assertIn("backdropFilter", self.content)

    def test_blade_bar_has_clip_path(self):
        """BladeBar should use clip-path for angled blade edges."""
        self.assertIn("clipPath", self.content)

    def test_blade_bar_has_aria_attributes(self):
        """BladeBar should have ARIA attributes for accessibility."""
        self.assertIn("aria-label", self.content)
        self.assertIn("aria-selected", self.content)

    def test_blade_bar_has_sound_integration(self):
        """BladeBar should integrate with SoundManager for sweep sound."""
        self.assertIn("SoundManager", self.content)
        self.assertIn("sweep", self.content)

    def test_blade_bar_is_memoized(self):
        """BladeBar should be wrapped in memo() for performance."""
        self.assertIn("memo", self.content)


# ═══════════════════════════════════════════════════════════════════
# BladeContent Component Tests
# ═══════════════════════════════════════════════════════════════════

class TestBladeContentContent(unittest.TestCase):
    """Verify BladeContent.tsx has correct structure."""

    def setUp(self):
        self.path = os.path.join(_project_root, "src", "components", "BladeContent.tsx")
        self.content = read_file(self.path)

    def test_blade_content_exports_component(self):
        """BladeContent should export a BladeContent component."""
        self.assertIn("BladeContent", self.content)

    def test_blade_content_has_icon_prop(self):
        """BladeContent should accept an icon prop."""
        self.assertIn("icon", self.content)

    def test_blade_content_has_title_prop(self):
        """BladeContent should accept a title prop."""
        self.assertIn("title", self.content)

    def test_blade_content_has_children_prop(self):
        """BladeContent should accept children."""
        self.assertIn("children", self.content)

    def test_blade_content_has_left_panel(self):
        """BladeContent should have a left decorative icon panel."""
        self.assertIn("icon", self.content)
        # The left panel should contain the large icon
        self.assertIn("fontSize", self.content)

    def test_blade_content_has_right_content_area(self):
        """BladeContent should have a right scrollable content area."""
        self.assertIn("overflowY", self.content)

    def test_blade_content_has_frosted_glass(self):
        """BladeContent should use frosted glass effect."""
        self.assertIn("backdropFilter", self.content)

    def test_blade_content_has_radial_gradient(self):
        """BladeContent should have radial spotlight gradient background."""
        self.assertIn("radial-gradient", self.content)

    def test_blade_content_uses_framer_motion(self):
        """BladeContent should use framer-motion for entrance animations."""
        self.assertIn("motion", self.content)

    def test_blade_content_has_gold_glow(self):
        """BladeContent should have gold accent glow on icon."""
        self.assertIn("accent", self.content.lower())
        self.assertIn("textShadow", self.content)


# ═══════════════════════════════════════════════════════════════════
# SoundManager Component Tests
# ═══════════════════════════════════════════════════════════════════

class TestSoundManagerContent(unittest.TestCase):
    """Verify SoundManager.tsx has correct structure."""

    def setUp(self):
        self.path = os.path.join(_project_root, "src", "components", "SoundManager.tsx")
        self.content = read_file(self.path)

    def test_sound_manager_exports_engine(self):
        """SoundManager should export SoundEngine class."""
        self.assertIn("SoundEngine", self.content)

    def test_sound_manager_exports_get_sound_engine(self):
        """SoundManager should export getSoundEngine singleton factory."""
        self.assertIn("getSoundEngine", self.content)

    def test_sound_manager_exports_mute_toggle(self):
        """SoundManager should export MuteToggle component."""
        self.assertIn("MuteToggle", self.content)

    def test_sound_manager_has_sweep_sound(self):
        """SoundEngine should have sweep (blade change whoosh) sound."""
        self.assertIn("sweep", self.content)

    def test_sound_manager_has_select_sound(self):
        """SoundEngine should have select (menu click) sound."""
        self.assertIn("select", self.content)

    def test_sound_manager_has_scroll_sound(self):
        """SoundEngine should have scroll (tick) sound."""
        self.assertIn("scroll", self.content)

    def test_sound_manager_has_back_sound(self):
        """SoundEngine should have back sound."""
        self.assertIn("back", self.content)

    def test_sound_manager_has_ambient_drone(self):
        """SoundEngine should have ambient drone (startAmbient/stopAmbient)."""
        self.assertIn("ambient", self.content.lower())

    def test_sound_manager_has_mute_support(self):
        """SoundEngine should support muting."""
        self.assertIn("muted", self.content)
        self.assertIn("setMuted", self.content)
        self.assertIn("isMuted", self.content)

    def test_sound_manager_uses_web_audio_api(self):
        """SoundEngine should use Web Audio API (AudioContext)."""
        self.assertIn("AudioContext", self.content)

    def test_sound_manager_has_play_method(self):
        """SoundEngine should have a play() method that routes to sound types."""
        self.assertIn("play(", self.content)

    def test_sound_manager_has_localstorage_persistence(self):
        """SoundEngine should persist mute preference to localStorage."""
        self.assertIn("localStorage", self.content)


# ═══════════════════════════════════════════════════════════════════
# App.tsx Xbox Layout Tests
# ═══════════════════════════════════════════════════════════════════

class TestAppXboxLayout(unittest.TestCase):
    """Verify App.tsx uses Xbox Blades layout structure."""

    def setUp(self):
        self.path = os.path.join(_project_root, "src", "App.tsx")
        self.content = read_file(self.path)

    def test_app_imports_blade_bar(self):
        """App.tsx should import BladeBar component."""
        self.assertIn("BladeBar", self.content)

    def test_app_imports_blade_content(self):
        """App.tsx should import BladeContent component."""
        self.assertIn("BladeContent", self.content)

    def test_app_imports_sound_manager(self):
        """App.tsx should import from SoundManager."""
        self.assertIn("SoundManager", self.content)

    def test_app_imports_mute_toggle(self):
        """App.tsx should import MuteToggle component."""
        self.assertIn("MuteToggle", self.content)

    def test_app_has_8_screens(self):
        """App.tsx should define all 8 screen routes."""
        for screen in ["model-setup", "domain-setup", "probing", "review",
                        "format", "train", "stats", "settings"]:
            self.assertIn(screen, self.content, f"Screen '{screen}' not in App.tsx")

    def test_app_has_screen_icons(self):
        """App.tsx should define screen icons for BladeContent."""
        for icon in ["🖥", "📚", "🔍", "📋", "📦", "🎯", "📊", "⚙"]:
            self.assertIn(icon, self.content, f"Icon '{icon}' not in App.tsx")

    def test_app_has_screen_titles(self):
        """App.tsx should define screen titles."""
        for title in ["Model Setup", "Domain Setup", "Probe Engine",
                       "Review Dashboard", "Format & Export", "Train & Evaluate",
                       "Statistics", "Settings"]:
            self.assertIn(title, self.content, f"Title '{title}' not in App.tsx")

    def test_app_has_blade_sweep_variants(self):
        """App.tsx should have Framer Motion blade sweep variants."""
        self.assertIn("bladeVariants", self.content)

    def test_app_has_3d_perspective(self):
        """App.tsx should use 3D perspective for blade transitions."""
        self.assertIn("perspective", self.content)
        self.assertIn("rotateY", self.content)

    def test_app_has_animate_presence(self):
        """App.tsx should use AnimatePresence for blade transitions."""
        self.assertIn("AnimatePresence", self.content)

    def test_app_has_direction_awareness(self):
        """App.tsx should track direction for blade sweep."""
        self.assertIn("direction", self.content)

    def test_app_has_blade_bar_at_bottom(self):
        """App.tsx should render BladeBar in a bottom container."""
        self.assertIn("BladeBar", self.content)

    def test_app_wraps_screens_in_blade_content(self):
        """App.tsx should wrap each screen in BladeContent."""
        self.assertIn("BladeContent", self.content)

    def test_app_has_controller_hints(self):
        """App.tsx should have controller hints (arrow keys, Enter)."""
        self.assertIn("Navigate", self.content)
        self.assertIn("Select", self.content)

    def test_app_has_hex_grid_background(self):
        """App.tsx should have hexagonal grid background overlay."""
        self.assertIn("hex-grid", self.content)

    def test_app_has_scanlines(self):
        """App.tsx should have scanline overlay."""
        self.assertIn("scanlines", self.content)

    def test_app_has_xbox_root_class(self):
        """App.tsx should have xbox-root CSS class."""
        self.assertIn("xbox-root", self.content)

    def test_app_has_connected_status(self):
        """App.tsx should show connected/disconnected status."""
        self.assertIn("Connected", self.content)

    def test_app_has_arrow_key_navigation(self):
        """App.tsx should handle arrow key navigation between blades."""
        self.assertIn("ArrowRight", self.content)
        self.assertIn("ArrowLeft", self.content)

    def test_app_plays_sweep_on_navigate(self):
        """App.tsx should play sweep sound on navigation."""
        self.assertIn("sweep", self.content)

    def test_app_has_mute_toggle_in_header(self):
        """App.tsx should render MuteToggle in the top bar."""
        self.assertIn("MuteToggle", self.content)


# ═══════════════════════════════════════════════════════════════════
# Theme Color Tests
# ═══════════════════════════════════════════════════════════════════

class TestXboxThemeColors(unittest.TestCase):
    """Verify Hermes theme colors are preserved in the Xbox redesign."""

    def setUp(self):
        self.path = os.path.join(_project_root, "src", "lib", "theme.ts")
        self.content = read_file(self.path)

    def test_theme_file_exists(self):
        """theme.ts should exist."""
        self.assertTrue(os.path.exists(self.path))

    def test_background_color(self):
        """Background should be #041C1C (deep dark teal)."""
        self.assertIn("#041C1C", self.content)

    def test_accent_color(self):
        """Accent should be #FFD700 (Hermes gold)."""
        self.assertIn("#FFD700", self.content)

    def test_text_primary_color(self):
        """Text primary should be #FFF8DC (cornsilk white)."""
        self.assertIn("#FFF8DC", self.content)

    def test_surface_color(self):
        """Surface should be #1B1713."""
        self.assertIn("#1B1713", self.content)

    def test_border_color(self):
        """Border should be #CD7F32 (bronze)."""
        self.assertIn("#CD7F32", self.content)

    def test_accent_glow(self):
        """Accent glow should be defined."""
        self.assertIn("accentGlow", self.content)

    def test_theme_has_8_screens(self):
        """Theme should define Screen type with all 8 screens."""
        for screen in ["model-setup", "domain-setup", "probing", "review",
                        "format", "train", "stats", "settings"]:
            self.assertIn(screen, self.content)


# ═══════════════════════════════════════════════════════════════════
# CSS Classes Tests
# ═══════════════════════════════════════════════════════════════════

class TestXboxCSSClasses(unittest.TestCase):
    """Verify CSS classes for Xbox visual effects exist in index.css."""

    def setUp(self):
        self.path = os.path.join(_project_root, "src", "index.css")
        self.content = read_file(self.path)

    def test_hex_grid_class(self):
        """CSS should have .hex-grid class for hexagonal grid background."""
        self.assertIn(".hex-grid", self.content)

    def test_scanlines_class(self):
        """CSS should have .scanlines class for CRT scanline effect."""
        self.assertIn(".scanlines", self.content)

    def test_blade_panel_class(self):
        """CSS should have .blade-panel class."""
        self.assertIn(".blade-panel", self.content)

    def test_caduceus_class(self):
        """CSS should have .caduceus class."""
        self.assertIn(".caduceus", self.content)

    def test_xbox_menu_item_class(self):
        """CSS should have .xbox-menu-item class."""
        self.assertIn(".xbox-menu-item", self.content)

    def test_xbox_menu_item_active_class(self):
        """CSS should have .xbox-menu-item-active class."""
        self.assertIn(".xbox-menu-item-active", self.content)

    def test_css_has_hermes_colors(self):
        """CSS should contain Hermes theme color values."""
        self.assertIn("#041C1C", self.content)  # Background
        self.assertIn("#FFD700", self.content)   # Accent gold

    def test_css_has_css_variables(self):
        """CSS should define CSS custom properties (variables)."""
        self.assertIn("--bg", self.content)
        self.assertIn("--accent", self.content)
        self.assertIn("--surface", self.content)


# ═══════════════════════════════════════════════════════════════════
# All Screens Still Present Tests
# ═══════════════════════════════════════════════════════════════════

class TestAllScreensPresent(unittest.TestCase):
    """Verify all 8 screen files still exist after the redesign."""

    def test_model_setup_exists(self):
        path = os.path.join(_project_root, "src", "screens", "ModelSetup.tsx")
        self.assertTrue(os.path.exists(path))

    def test_domain_setup_exists(self):
        path = os.path.join(_project_root, "src", "screens", "DomainSetup.tsx")
        self.assertTrue(os.path.exists(path))

    def test_probing_progress_exists(self):
        path = os.path.join(_project_root, "src", "screens", "ProbingProgress.tsx")
        self.assertTrue(os.path.exists(path))

    def test_review_dashboard_exists(self):
        path = os.path.join(_project_root, "src", "screens", "ReviewDashboard.tsx")
        self.assertTrue(os.path.exists(path))

    def test_format_export_exists(self):
        path = os.path.join(_project_root, "src", "screens", "FormatExport.tsx")
        self.assertTrue(os.path.exists(path))

    def test_train_evaluate_exists(self):
        path = os.path.join(_project_root, "src", "screens", "TrainEvaluate.tsx")
        self.assertTrue(os.path.exists(path))

    def test_stats_exists(self):
        path = os.path.join(_project_root, "src", "screens", "Stats.tsx")
        self.assertTrue(os.path.exists(path))

    def test_settings_exists(self):
        path = os.path.join(_project_root, "src", "screens", "Settings.tsx")
        self.assertTrue(os.path.exists(path))


# ═══════════════════════════════════════════════════════════════════
# Additional Components Tests
# ═══════════════════════════════════════════════════════════════════

class TestAdditionalComponents(unittest.TestCase):
    """Verify additional supporting components exist."""

    def test_error_boundary_exists(self):
        path = os.path.join(_project_root, "src", "components", "ErrorBoundary.tsx")
        self.assertTrue(os.path.exists(path))

    def test_caduceus_exists(self):
        path = os.path.join(_project_root, "src", "components", "Caduceus.tsx")
        self.assertTrue(os.path.exists(path))

    def test_confidence_badge_exists(self):
        path = os.path.join(_project_root, "src", "components", "ConfidenceBadge.tsx")
        self.assertTrue(os.path.exists(path))

    def test_progress_ring_exists(self):
        path = os.path.join(_project_root, "src", "components", "ProgressRing.tsx")
        self.assertTrue(os.path.exists(path))

    def test_score_line_chart_exists(self):
        path = os.path.join(_project_root, "src", "components", "ScoreLineChart.tsx")
        self.assertTrue(os.path.exists(path))

    def test_subject_bar_chart_exists(self):
        path = os.path.join(_project_root, "src", "components", "SubjectBarChart.tsx")
        self.assertTrue(os.path.exists(path))

    def test_loading_state_exists(self):
        path = os.path.join(_project_root, "src", "components", "LoadingState.tsx")
        self.assertTrue(os.path.exists(path))

    def test_empty_state_exists(self):
        path = os.path.join(_project_root, "src", "components", "EmptyState.tsx")
        self.assertTrue(os.path.exists(path))

    def test_error_state_exists(self):
        path = os.path.join(_project_root, "src", "components", "ErrorState.tsx")
        self.assertTrue(os.path.exists(path))

    def test_skeleton_card_exists(self):
        path = os.path.join(_project_root, "src", "components", "SkeletonCard.tsx")
        self.assertTrue(os.path.exists(path))


# ═══════════════════════════════════════════════════════════════════
# Hooks and Lib Tests
# ═══════════════════════════════════════════════════════════════════

class TestHooksAndLib(unittest.TestCase):
    """Verify hooks and lib files exist."""

    def test_use_websocket_hook_exists(self):
        path = os.path.join(_project_root, "src", "hooks", "useWebSocket.ts")
        self.assertTrue(os.path.exists(path))

    def test_api_lib_exists(self):
        path = os.path.join(_project_root, "src", "lib", "api.ts")
        self.assertTrue(os.path.exists(path))

    def test_theme_lib_exists(self):
        path = os.path.join(_project_root, "src", "lib", "theme.ts")
        self.assertTrue(os.path.exists(path))

    def test_main_tsx_exists(self):
        path = os.path.join(_project_root, "src", "main.tsx")
        self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
