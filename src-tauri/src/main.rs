// MindForge Tauri entry point
// Provides IPC commands for Python FastAPI sidecar lifecycle management.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::env;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;

/// Holds the optional Python sidecar child process.
/// Wrapped in a Mutex so it can be safely shared across Tauri command handlers.
pub struct SidecarState {
    pub child: Mutex<Option<Child>>,
}

impl Default for SidecarState {
    fn default() -> Self {
        SidecarState {
            child: Mutex::new(None),
        }
    }
}

/// Resolve the path to `python/server.py` robustly.
///
/// Search order:
/// 1. Relative to CWD (works in dev mode when run from project root)
/// 2. Relative to the directory containing the current executable
///    (works for bare binaries in target/release/)
/// 3. Relative to the parent of the executable's directory
///    (works inside .app bundles: Contents/MacOS/ -> Contents/ -> .app root -> project root)
///
/// Returns the resolved path if found, or an error listing attempted paths.
fn resolve_sidecar_path() -> Result<PathBuf, String> {
    let relative = PathBuf::from("python/server.py");

    // 1. CWD (dev mode)
    if relative.exists() {
        return Ok(relative);
    }

    let mut attempted = vec![relative.display().to_string()];

    // 2. Relative to executable directory
    if let Ok(exe) = env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let candidate = exe_dir.join("python/server.py");
            if candidate.exists() {
                return Ok(candidate);
            }
            attempted.push(candidate.display().to_string());

            // 3. Relative to parent of executable directory
            if let Some(exe_parent) = exe_dir.parent() {
                let candidate2 = exe_parent.join("python/server.py");
                if candidate2.exists() {
                    return Ok(candidate2);
                }
                attempted.push(candidate2.display().to_string());

                // 4. One more level up (inside .app bundle: MacOS -> Contents -> app -> project root)
                if let Some(exe_grandparent) = exe_parent.parent() {
                    let candidate3 = exe_grandparent.join("python/server.py");
                    if candidate3.exists() {
                        return Ok(candidate3);
                    }
                    attempted.push(candidate3.display().to_string());
                }
            }
        }
    }

    Err(format!(
        "Could not find python/server.py. Attempted paths:\n  {}",
        attempted.join("\n  ")
    ))
}

/// Launch the Python FastAPI sidecar process.
/// Returns the PID as a string on success.
#[tauri::command]
fn start_sidecar(state: tauri::State<'_, SidecarState>) -> Result<String, String> {
    let mut guard = state.child.lock().map_err(|e| e.to_string())?;

    // Check if a sidecar is already running
    if let Some(ref mut child) = *guard {
        match child.try_wait() {
            Ok(None) => {
                // Still running
                return Err("Sidecar is already running".to_string());
            }
            Ok(Some(_)) => {
                // Process exited -- fall through to start a new one
            }
            Err(_) => {
                return Err("Failed to check sidecar status".to_string());
            }
        }
    }

    // Resolve the sidecar script path robustly
    let server_path = resolve_sidecar_path()?;

    // Spawn the Python FastAPI sidecar
    let child = Command::new("python3")
        .arg(&server_path)
        .spawn()
        .map_err(|e| format!("Failed to start sidecar: {}", e))?;

    let pid = child.id();
    *guard = Some(child);
    Ok(pid.to_string())
}

/// Stop the Python sidecar process if it is running.
/// Idempotent -- returns Ok(()) if no sidecar is running.
#[tauri::command]
fn stop_sidecar(state: tauri::State<'_, SidecarState>) -> Result<(), String> {
    let mut guard = state.child.lock().map_err(|e| e.to_string())?;

    if let Some(mut child) = guard.take() {
        child
            .kill()
            .map_err(|e| format!("Failed to kill sidecar: {}", e))?;
        // Wait to reap the zombie process
        let _ = child.wait();
    }

    Ok(())
}

/// Check whether the sidecar is currently running.
/// Returns true if the child process exists and has not exited.
/// Cleans up the state if the process has exited.
#[tauri::command]
fn sidecar_status(state: tauri::State<'_, SidecarState>) -> Result<bool, String> {
    let mut guard = state.child.lock().map_err(|e| e.to_string())?;

    if let Some(ref mut child) = *guard {
        match child.try_wait() {
            Ok(None) => Ok(true), // Still running
            Ok(Some(_)) => {
                // Process has exited -- clean up
                *guard = None;
                Ok(false)
            }
            Err(_) => {
                *guard = None;
                Ok(false)
            }
        }
    } else {
        Ok(false)
    }
}

fn main() {
    tauri::Builder::default()
        .manage(SidecarState::default())
        .invoke_handler(tauri::generate_handler![
            start_sidecar,
            stop_sidecar,
            sidecar_status
        ])
        .run(tauri::generate_context!())
        .expect("error while running MindForge application");
}
