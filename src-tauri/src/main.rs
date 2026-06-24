// MindForge Tauri entry point
// Provides IPC commands for Python FastAPI sidecar lifecycle management.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

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

    // Spawn the Python FastAPI sidecar
    let child = Command::new("python3")
        .arg("python/server.py")
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
        child.kill().map_err(|e| format!("Failed to kill sidecar: {}", e))?;
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
            Ok(None) => Ok(true),  // Still running
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
