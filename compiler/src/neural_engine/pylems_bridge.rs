// 🐍 PYLEMS VALIDATION BRIDGE
// Real Python integration for scientific validation with PyLEMS

use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::io::Write;
use serde_json::{json, Value};

/// Bridge to PyLEMS for scientific validation
pub struct PyLEMSBridge {
    /// Python executable path
    python_path: String,
    
    /// PyLEMS validation script
    validation_script: String,
    
    /// Cached NeuroML model
    neuroml_model: Option<String>,
}

/// Validation result from PyLEMS comparison
#[derive(Debug, Clone)]
pub struct PyLEMSValidationResult {
    /// Is the simulation valid according to PyLEMS?
    pub is_valid: bool,
    
    /// Divergence from PyLEMS reference
    pub divergence_from_pylems: f64,
    
    /// Correlation coefficient with PyLEMS
    pub correlation: f64,
    
    /// PyLEMS simulation time (ms)
    pub pylems_runtime: f64,
    
    /// Our simulation time (ms)
    pub our_runtime: f64,
    
    /// Performance ratio (our_time / pylems_time)
    pub performance_ratio: f64,
    
    /// Detailed error analysis
    pub error_analysis: ErrorAnalysis,
}

#[derive(Debug, Clone)]
pub struct ErrorAnalysis {
    /// Maximum absolute error
    pub max_absolute_error: f64,
    
    /// Root mean square error
    pub rmse: f64,
    
    /// Mean absolute percentage error
    pub mape: f64,
    
    /// Error at spike peaks
    pub spike_error: f64,
    
    /// Phase lag in oscillations (ms)
    pub phase_lag: f64,
}

#[derive(Debug)]
pub enum PyLEMSError {
    PythonNotFound,
    PyLEMSNotInstalled,
    ModelConversionError(String),
    ValidationError(String),
    RuntimeError(String),
}

impl PyLEMSBridge {
    /// Create new PyLEMS bridge
    pub fn new() -> Result<Self, PyLEMSError> {
        let python_path = Self::find_python()?;
        Self::check_pylems_installation(&python_path)?;
        
        let validation_script = Self::generate_validation_script();
        
        Ok(PyLEMSBridge {
            python_path,
            validation_script,
            neuroml_model: None,
        })
    }
    
    /// Create bridge with custom Python path
    pub fn with_python_path(python_path: String) -> Result<Self, PyLEMSError> {
        Self::check_pylems_installation(&python_path)?;
        
        let validation_script = Self::generate_validation_script();
        
        Ok(PyLEMSBridge {
            python_path,
            validation_script,
            neuroml_model: None,
        })
    }
    
    /// Validate our simulation against PyLEMS reference
    pub fn validate_simulation(
        &mut self, 
        our_results: &[f64], 
        simulation_params: &SimulationParameters
    ) -> Result<PyLEMSValidationResult, PyLEMSError> {
        
        // Generate NeuroML model if not cached
        if self.neuroml_model.is_none() {
            self.neuroml_model = Some(self.generate_neuroml_model(simulation_params)?);
        }
        
        // Run PyLEMS simulation
        let pylems_results = self.run_pylems_simulation(simulation_params)?;
        
        // Compare results
        let validation_result = self.compare_results(our_results, &pylems_results, simulation_params)?;
        
        Ok(validation_result)
    }
    
    /// Validate single timestep
    pub fn validate_timestep(
        &mut self,
        our_voltage: f64,
        our_gates: &HashMap<String, f64>,
        simulation_time: f64,
        dt: f64
    ) -> Result<f64, PyLEMSError> {
        // For real-time validation, we'd maintain a running PyLEMS simulation
        // and compare state at each timestep
        
        // Simplified: just validate voltage difference
        let expected_voltage = self.get_expected_voltage_at_time(simulation_time)?;
        let divergence = (our_voltage - expected_voltage).abs() / expected_voltage.abs().max(1.0);
        
        Ok(divergence)
    }
    
    /// Find Python executable
    fn find_python() -> Result<String, PyLEMSError> {
        // Try different Python executable names
        for python_cmd in &["python3", "python", "python3.8", "python3.9", "python3.10", "python3.11"] {
            if let Ok(_) = Command::new(python_cmd).arg("--version").output() {
                return Ok(python_cmd.to_string());
            }
        }
        
        Err(PyLEMSError::PythonNotFound)
    }
    
    /// Check if PyLEMS is installed
    fn check_pylems_installation(python_path: &str) -> Result<(), PyLEMSError> {
        let output = Command::new(python_path)
            .arg("-c")
            .arg("import pylems; print(pylems.__version__)")
            .output()
            .map_err(|_| PyLEMSError::PythonNotFound)?;
        
        if !output.status.success() {
            return Err(PyLEMSError::PyLEMSNotInstalled);
        }
        
        Ok(())
    }
    
    /// Generate PyLEMS validation script
    fn generate_validation_script() -> String {
        r#"
import json
import sys
import numpy as np
from pylems import Simulation
import matplotlib.pyplot as plt
from io import StringIO
import traceback

def run_hodgkin_huxley_simulation(params):
    """Run Hodgkin-Huxley simulation with PyLEMS"""
    try:
        # Create simulation
        sim = Simulation("HH_validation", dt=params['dt'], target="hhcell")
        
        # Add Hodgkin-Huxley cell
        sim.add_cell_type("hhcell", {
            'C': params.get('C', 1.0),
            'gNa': params.get('gNa', 120.0),
            'gK': params.get('gK', 36.0),
            'gL': params.get('gL', 0.3),
            'ENa': params.get('ENa', 50.0),
            'EK': params.get('EK', -77.0),
            'EL': params.get('EL', -54.4),
        })
        
        # Set initial conditions
        sim.set_initial_conditions({
            'v': params.get('v_init', -65.0),
            'm': params.get('m_init', 0.05),
            'h': params.get('h_init', 0.6),
            'n': params.get('n_init', 0.32),
        })
        
        # Add current injection
        if 'current_injection' in params:
            for t, i in params['current_injection']:
                sim.add_current_injection(t, i)
        
        # Run simulation
        runtime = params.get('runtime', 20.0)
        results = sim.run(runtime)
        
        return {
            'time': results['time'].tolist(),
            'voltage': results['v'].tolist(),
            'gates': {
                'm': results['m'].tolist(),
                'h': results['h'].tolist(),
                'n': results['n'].tolist(),
            },
            'success': True,
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }

def calculate_validation_metrics(our_data, pylems_data):
    """Calculate validation metrics between our results and PyLEMS"""
    
    our_v = np.array(our_data['voltage'])
    pylems_v = np.array(pylems_data['voltage'])
    
    # Ensure same length (interpolate if needed)
    if len(our_v) != len(pylems_v):
        pylems_t = np.array(pylems_data['time'])
        our_t = np.linspace(pylems_t[0], pylems_t[-1], len(our_v))
        pylems_v = np.interp(our_t, pylems_t, pylems_v)
    
    # Calculate metrics
    diff = our_v - pylems_v
    
    metrics = {
        'max_absolute_error': float(np.max(np.abs(diff))),
        'rmse': float(np.sqrt(np.mean(diff**2))),
        'correlation': float(np.corrcoef(our_v, pylems_v)[0, 1]) if len(our_v) > 1 else 1.0,
        'divergence': float(np.mean(np.abs(diff)) / np.mean(np.abs(pylems_v))),
    }
    
    # Calculate MAPE (avoiding division by zero)
    pylems_nonzero = pylems_v[np.abs(pylems_v) > 1e-6]
    our_nonzero = our_v[np.abs(pylems_v) > 1e-6]
    if len(pylems_nonzero) > 0:
        mape = np.mean(np.abs((pylems_nonzero - our_nonzero) / pylems_nonzero)) * 100
        metrics['mape'] = float(mape)
    else:
        metrics['mape'] = 0.0
    
    # Detect spikes and calculate spike error
    spike_threshold = -20.0  # mV
    our_spikes = np.where(our_v > spike_threshold)[0]
    pylems_spikes = np.where(pylems_v > spike_threshold)[0]
    
    if len(our_spikes) > 0 and len(pylems_spikes) > 0:
        spike_error = np.mean([np.abs(our_v[spike] - pylems_v[spike]) for spike in our_spikes if spike < len(pylems_v)])
        metrics['spike_error'] = float(spike_error)
    else:
        metrics['spike_error'] = 0.0
    
    # Calculate phase lag (simplified)
    if len(our_spikes) > 0 and len(pylems_spikes) > 0:
        dt = pylems_data['time'][1] - pylems_data['time'][0] if len(pylems_data['time']) > 1 else 0.01
        phase_lag = (our_spikes[0] - pylems_spikes[0]) * dt
        metrics['phase_lag'] = float(phase_lag)
    else:
        metrics['phase_lag'] = 0.0
    
    return metrics

if __name__ == "__main__":
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        
        if input_data['command'] == 'simulate':
            result = run_hodgkin_huxley_simulation(input_data['params'])
            print(json.dumps(result))
            
        elif input_data['command'] == 'validate':
            our_data = input_data['our_data']
            params = input_data['params']
            
            # Run PyLEMS simulation
            pylems_result = run_hodgkin_huxley_simulation(params)
            
            if pylems_result['success']:
                # Calculate validation metrics
                metrics = calculate_validation_metrics(our_data, pylems_result)
                
                result = {
                    'success': True,
                    'pylems_data': pylems_result,
                    'validation_metrics': metrics,
                }
            else:
                result = pylems_result
            
            print(json.dumps(result))
            
        else:
            print(json.dumps({'success': False, 'error': 'Unknown command'}))
            
    except Exception as e:
        print(json.dumps({
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc()
        }))
"#.to_string()
    }
    
    /// Generate NeuroML model for the simulation
    fn generate_neuroml_model(&self, params: &SimulationParameters) -> Result<String, PyLEMSError> {
        let neuroml = format!(r#"
<?xml version="1.0" encoding="UTF-8"?>
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2">
    
    <ionChannel id="na" type="ionChannelHH" conductance="10pS">
        <gateHHrates id="m" instances="3">
            <forwardRate type="HHExpLinearRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
            <reverseRate type="HHExpRate" rate="4per_ms" midpoint="-65mV" scale="-18mV"/>
        </gateHHrates>
        <gateHHrates id="h" instances="1">
            <forwardRate type="HHExpRate" rate="0.07per_ms" midpoint="-65mV" scale="-20mV"/>
            <reverseRate type="HHSigmoidRate" rate="1per_ms" midpoint="-35mV" scale="-10mV"/>
        </gateHHrates>
    </ionChannel>
    
    <ionChannel id="k" type="ionChannelHH" conductance="10pS">
        <gateHHrates id="n" instances="4">
            <forwardRate type="HHExpLinearRate" rate="0.1per_ms" midpoint="-55mV" scale="10mV"/>
            <reverseRate type="HHExpRate" rate="0.125per_ms" midpoint="-65mV" scale="-80mV"/>
        </gateHHrates>
    </ionChannel>
    
    <cell id="hhcell">
        <morphology id="morphology_hhcell">
            <segment id="0" name="soma">
                <proximal x="0" y="0" z="0" diameter="10"/>
                <distal x="0" y="0" z="0" diameter="10"/>
            </segment>
        </morphology>
        
        <biophysicalProperties id="biophys_hhcell">
            <membraneProperties>
                <channelDensity id="na_density" ionChannel="na" condDensity="{na_conductance}mS_per_cm2" erev="50mV"/>
                <channelDensity id="k_density" ionChannel="k" condDensity="{k_conductance}mS_per_cm2" erev="-77mV"/>
                <channelDensity id="leak" ionChannel="passiveChan" condDensity="{leak_conductance}mS_per_cm2" erev="-54.4mV"/>
                
                <spikeThresh value="0mV"/>
                <specificCapacitance value="{capacitance}uF_per_cm2"/>
                <initMembPotential value="{v_init}mV"/>
            </membraneProperties>
        </biophysicalProperties>
    </cell>
    
</neuroml>
"#, 
            na_conductance = params.gna,
            k_conductance = params.gk, 
            leak_conductance = params.gl,
            capacitance = params.c,
            v_init = params.v_init
        );
        
        Ok(neuroml)
    }
    
    /// Run PyLEMS simulation
    fn run_pylems_simulation(&self, params: &SimulationParameters) -> Result<Vec<f64>, PyLEMSError> {
        let input_data = json!({
            "command": "simulate",
            "params": {
                "dt": params.dt,
                "runtime": params.runtime,
                "C": params.c,
                "gNa": params.gna,
                "gK": params.gk,
                "gL": params.gl,
                "ENa": params.ena,
                "EK": params.ek,
                "EL": params.el,
                "v_init": params.v_init,
                "m_init": params.m_init,
                "h_init": params.h_init,
                "n_init": params.n_init,
                "current_injection": params.current_injection.as_ref().map(|inj| {
                    inj.iter().enumerate().map(|(i, &current)| {
                        [i as f64 * params.dt, current]
                    }).collect::<Vec<_>>()
                })
            }
        });
        
        let mut child = Command::new(&self.python_path)
            .arg("-c")
            .arg(&self.validation_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| PyLEMSError::RuntimeError(e.to_string()))?;
        
        // Send input data
        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(input_data.to_string().as_bytes())
                .map_err(|e| PyLEMSError::RuntimeError(e.to_string()))?;
        }
        
        // Get output
        let output = child.wait_with_output()
            .map_err(|e| PyLEMSError::RuntimeError(e.to_string()))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PyLEMSError::RuntimeError(format!("Python script failed: {}", stderr)));
        }
        
        // Parse results
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: Value = serde_json::from_str(&stdout)
            .map_err(|e| PyLEMSError::RuntimeError(format!("Failed to parse JSON: {}", e)))?;
        
        if !result["success"].as_bool().unwrap_or(false) {
            let error = result["error"].as_str().unwrap_or("Unknown error");
            return Err(PyLEMSError::ValidationError(error.to_string()));
        }
        
        // Extract voltage data
        let voltage_data = result["voltage"].as_array()
            .ok_or_else(|| PyLEMSError::ValidationError("No voltage data in response".to_string()))?;
        
        let voltages: Result<Vec<f64>, _> = voltage_data.iter()
            .map(|v| v.as_f64().ok_or_else(|| PyLEMSError::ValidationError("Invalid voltage value".to_string())))
            .collect();
        
        voltages
    }
    
    /// Compare our results with PyLEMS
    fn compare_results(
        &self, 
        our_results: &[f64], 
        pylems_results: &[f64],
        params: &SimulationParameters
    ) -> Result<PyLEMSValidationResult, PyLEMSError> {
        
        let input_data = json!({
            "command": "validate",
            "our_data": {
                "voltage": our_results,
                "time": (0..our_results.len()).map(|i| i as f64 * params.dt).collect::<Vec<_>>()
            },
            "params": {
                "dt": params.dt,
                "runtime": params.runtime,
                "C": params.c,
                "gNa": params.gna,
                "gK": params.gk,
                "gL": params.gl,
                "ENa": params.ena,
                "EK": params.ek,
                "EL": params.el,
                "v_init": params.v_init,
                "m_init": params.m_init,
                "h_init": params.h_init,
                "n_init": params.n_init,
            }
        });
        
        let mut child = Command::new(&self.python_path)
            .arg("-c")
            .arg(&self.validation_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| PyLEMSError::RuntimeError(e.to_string()))?;
        
        // Send input data
        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(input_data.to_string().as_bytes())
                .map_err(|e| PyLEMSError::RuntimeError(e.to_string()))?;
        }
        
        // Get output
        let output = child.wait_with_output()
            .map_err(|e| PyLEMSError::RuntimeError(e.to_string()))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PyLEMSError::RuntimeError(format!("Validation script failed: {}", stderr)));
        }
        
        // Parse results
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: Value = serde_json::from_str(&stdout)
            .map_err(|e| PyLEMSError::RuntimeError(format!("Failed to parse validation JSON: {}", e)))?;
        
        if !result["success"].as_bool().unwrap_or(false) {
            let error = result["error"].as_str().unwrap_or("Unknown validation error");
            return Err(PyLEMSError::ValidationError(error.to_string()));
        }
        
        // Extract validation metrics
        let metrics = &result["validation_metrics"];
        
        let error_analysis = ErrorAnalysis {
            max_absolute_error: metrics["max_absolute_error"].as_f64().unwrap_or(0.0),
            rmse: metrics["rmse"].as_f64().unwrap_or(0.0),
            mape: metrics["mape"].as_f64().unwrap_or(0.0),
            spike_error: metrics["spike_error"].as_f64().unwrap_or(0.0),
            phase_lag: metrics["phase_lag"].as_f64().unwrap_or(0.0),
        };
        
        let divergence = metrics["divergence"].as_f64().unwrap_or(1.0);
        let correlation = metrics["correlation"].as_f64().unwrap_or(0.0);
        
        // Determine if simulation is valid
        let is_valid = divergence < 0.1 && correlation > 0.9 && error_analysis.rmse < 5.0;
        
        Ok(PyLEMSValidationResult {
            is_valid,
            divergence_from_pylems: divergence,
            correlation,
            pylems_runtime: 0.0, // Would be measured in real implementation
            our_runtime: 0.0,    // Would be measured in real implementation 
            performance_ratio: 1.0,
            error_analysis,
        })
    }
    
    /// Get expected voltage at specific time (for real-time validation)
    fn get_expected_voltage_at_time(&self, _time: f64) -> Result<f64, PyLEMSError> {
        // In real implementation, would maintain running PyLEMS simulation
        // For now, return placeholder
        Ok(-65.0)
    }
}

/// Parameters for PyLEMS simulation
#[derive(Debug, Clone)]
pub struct SimulationParameters {
    pub dt: f64,
    pub runtime: f64,
    pub c: f64,
    pub gna: f64,
    pub gk: f64,
    pub gl: f64,
    pub ena: f64,
    pub ek: f64,
    pub el: f64,
    pub v_init: f64,
    pub m_init: f64,
    pub h_init: f64,
    pub n_init: f64,
    pub current_injection: Option<Vec<f64>>,
}

impl Default for SimulationParameters {
    fn default() -> Self {
        SimulationParameters {
            dt: 0.01,
            runtime: 20.0,
            c: 1.0,
            gna: 120.0,
            gk: 36.0,
            gl: 0.3,
            ena: 50.0,
            ek: -77.0,
            el: -54.4,
            v_init: -65.0,
            m_init: 0.05,
            h_init: 0.6,
            n_init: 0.32,
            current_injection: None,
        }
    }
}

impl std::fmt::Display for PyLEMSError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PyLEMSError::PythonNotFound => write!(f, "Python interpreter not found"),
            PyLEMSError::PyLEMSNotInstalled => write!(f, "PyLEMS not installed (pip install pylems)"),
            PyLEMSError::ModelConversionError(msg) => write!(f, "Model conversion error: {}", msg),
            PyLEMSError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            PyLEMSError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
        }
    }
}

impl std::error::Error for PyLEMSError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_python_detection() {
        // This test might fail if Python is not installed
        match PyLEMSBridge::find_python() {
            Ok(python_path) => {
                println!("Found Python at: {}", python_path);
                assert!(!python_path.is_empty());
            }
            Err(PyLEMSError::PythonNotFound) => {
                println!("Python not found - skipping test");
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }
    
    #[test]
    fn test_simulation_parameters() {
        let params = SimulationParameters::default();
        assert_eq!(params.dt, 0.01);
        assert_eq!(params.gna, 120.0);
        assert_eq!(params.v_init, -65.0);
    }
    
    #[test]
    fn test_validation_script_generation() {
        let script = PyLEMSBridge::generate_validation_script();
        assert!(script.contains("run_hodgkin_huxley_simulation"));
        assert!(script.contains("calculate_validation_metrics"));
        assert!(script.contains("import pylems"));
    }
}