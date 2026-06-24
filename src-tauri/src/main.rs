// MindForge Tauri entry point
// Prevents additional console window on Windows
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::Command;

fn main() {
    // Spawn the Python FastAPI sidecar
    let sidecar = Command::new("python3")
        .arg("python/server.py")
        .spawn();

    // Initialize Tauri app
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running MindForge application");
}
