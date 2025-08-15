// 🔬 LEMS DYNAMICS ENGINE
// Low Entropy Model Specification engine for accurate neural simulation

use std::collections::HashMap;
use std::f64::consts::{E, PI};
use super::expression_parser::{ExpressionParser, EvaluationContext, ParseError};

/// LEMS simulation engine for precise neural dynamics
pub struct LEMSEngine {
    /// Integration timestep (ms)
    dt: f64,
    
    /// Current simulation time (ms)
    time: f64,
    
    /// State variables for all components
    component_states: HashMap<String, ComponentState>,
    
    /// Component definitions
    component_definitions: HashMap<String, LEMSComponent>,
    
    /// Simulation parameters
    simulation_params: SimulationParams,
    
    /// Expression parser for dynamics equations
    expression_parser: ExpressionParser,
}

/// State variables for a LEMS component
#[derive(Clone, Debug)]
pub struct ComponentState {
    /// State variable values
    pub variables: HashMap<String, f64>,
    
    /// Parameter values
    pub parameters: HashMap<String, f64>,
    
    /// Derived variables (computed each step)
    pub derived: HashMap<String, f64>,
    
    /// Event flags
    pub events: HashMap<String, bool>,
}

/// LEMS component definition
#[derive(Clone, Debug)]
pub struct LEMSComponent {
    /// Component type name
    pub component_type: String,
    
    /// State variable definitions
    pub state_variables: Vec<StateVariable>,
    
    /// Parameter definitions
    pub parameters: Vec<Parameter>,
    
    /// Derived variable definitions
    pub derived_variables: Vec<DerivedVariable>,
    
    /// Time derivative definitions
    pub time_derivatives: Vec<TimeDerivative>,
    
    /// Event definitions
    pub events: Vec<Event>,
    
    /// Conditional derived variables
    pub conditional_derived: Vec<ConditionalDerived>,
}

#[derive(Clone, Debug)]
pub struct StateVariable {
    pub name: String,
    pub initial_value: f64,
    pub dimension: String,
}

#[derive(Clone, Debug)]
pub struct Parameter {
    pub name: String,
    pub dimension: String,
    pub default_value: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct DerivedVariable {
    pub name: String,
    pub expression: String,
    pub dimension: String,
}

#[derive(Clone, Debug)]
pub struct TimeDerivative {
    pub variable: String,
    pub expression: String,
}

#[derive(Clone, Debug)]
pub struct Event {
    pub port: String,
    pub condition: String,
}

#[derive(Clone, Debug)]
pub struct ConditionalDerived {
    pub name: String,
    pub if_condition: String,
    pub then_expression: String,
    pub else_expression: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SimulationParams {
    /// Integration method
    pub method: IntegrationMethod,
    
    /// Absolute tolerance
    pub abs_tolerance: f64,
    
    /// Relative tolerance
    pub rel_tolerance: f64,
    
    /// Maximum timestep
    pub max_dt: f64,
    
    /// Minimum timestep
    pub min_dt: f64,
}

#[derive(Clone, Debug)]
pub enum IntegrationMethod {
    Euler,
    RungeKutta4,
    AdaptiveRungeKutta,
}

impl LEMSEngine {
    pub fn new(dt: f64) -> Self {
        LEMSEngine {
            dt,
            time: 0.0,
            component_states: HashMap::new(),
            component_definitions: HashMap::new(),
            simulation_params: SimulationParams::default(),
            expression_parser: ExpressionParser::new(),
        }
    }
    
    /// Initialize with Hodgkin-Huxley LEMS definition
    pub fn with_hodgkin_huxley() -> Self {
        let mut engine = Self::new(0.01); // 0.01 ms timestep
        engine.load_hodgkin_huxley_component();
        engine
    }
    
    /// Load the Hodgkin-Huxley component definition
    fn load_hodgkin_huxley_component(&mut self) {
        let hh_component = LEMSComponent {
            component_type: "HodgkinHuxleyCell".to_string(),
            
            state_variables: vec![
                StateVariable {
                    name: "v".to_string(),
                    initial_value: -65.0,
                    dimension: "voltage".to_string(),
                },
                StateVariable {
                    name: "m".to_string(),
                    initial_value: 0.05,
                    dimension: "dimensionless".to_string(),
                },
                StateVariable {
                    name: "h".to_string(),
                    initial_value: 0.6,
                    dimension: "dimensionless".to_string(),
                },
                StateVariable {
                    name: "n".to_string(),
                    initial_value: 0.32,
                    dimension: "dimensionless".to_string(),
                },
            ],
            
            parameters: vec![
                Parameter {
                    name: "C".to_string(),
                    dimension: "capacitance".to_string(),
                    default_value: Some(1.0), // 1 μF/cm²
                },
                Parameter {
                    name: "gNa".to_string(),
                    dimension: "conductance".to_string(),
                    default_value: Some(120.0), // 120 mS/cm²
                },
                Parameter {
                    name: "gK".to_string(),
                    dimension: "conductance".to_string(),
                    default_value: Some(36.0), // 36 mS/cm²
                },
                Parameter {
                    name: "gL".to_string(),
                    dimension: "conductance".to_string(),
                    default_value: Some(0.3), // 0.3 mS/cm²
                },
                Parameter {
                    name: "ENa".to_string(),
                    dimension: "voltage".to_string(),
                    default_value: Some(50.0), // 50 mV
                },
                Parameter {
                    name: "EK".to_string(),
                    dimension: "voltage".to_string(),
                    default_value: Some(-77.0), // -77 mV
                },
                Parameter {
                    name: "EL".to_string(),
                    dimension: "voltage".to_string(),
                    default_value: Some(-54.4), // -54.4 mV
                },
            ],
            
            derived_variables: vec![
                // Sodium channel rate constants
                DerivedVariable {
                    name: "alpha_m".to_string(),
                    expression: "0.1 * (v + 40) / (1 - exp(-(v + 40) / 10))".to_string(),
                    dimension: "per_time".to_string(),
                },
                DerivedVariable {
                    name: "beta_m".to_string(),
                    expression: "4 * exp(-(v + 65) / 18)".to_string(),
                    dimension: "per_time".to_string(),
                },
                DerivedVariable {
                    name: "alpha_h".to_string(),
                    expression: "0.07 * exp(-(v + 65) / 20)".to_string(),
                    dimension: "per_time".to_string(),
                },
                DerivedVariable {
                    name: "beta_h".to_string(),
                    expression: "1 / (1 + exp(-(v + 35) / 10))".to_string(),
                    dimension: "per_time".to_string(),
                },
                // Potassium channel rate constants
                DerivedVariable {
                    name: "alpha_n".to_string(),
                    expression: "0.01 * (v + 55) / (1 - exp(-(v + 55) / 10))".to_string(),
                    dimension: "per_time".to_string(),
                },
                DerivedVariable {
                    name: "beta_n".to_string(),
                    expression: "0.125 * exp(-(v + 65) / 80)".to_string(),
                    dimension: "per_time".to_string(),
                },
                // Ionic currents
                DerivedVariable {
                    name: "iNa".to_string(),
                    expression: "gNa * m^3 * h * (v - ENa)".to_string(),
                    dimension: "current".to_string(),
                },
                DerivedVariable {
                    name: "iK".to_string(),
                    expression: "gK * n^4 * (v - EK)".to_string(),
                    dimension: "current".to_string(),
                },
                DerivedVariable {
                    name: "iL".to_string(),
                    expression: "gL * (v - EL)".to_string(),
                    dimension: "current".to_string(),
                },
            ],
            
            time_derivatives: vec![
                TimeDerivative {
                    variable: "v".to_string(),
                    expression: "(iExt - iNa - iK - iL) / C".to_string(),
                },
                TimeDerivative {
                    variable: "m".to_string(),
                    expression: "alpha_m * (1 - m) - beta_m * m".to_string(),
                },
                TimeDerivative {
                    variable: "h".to_string(),
                    expression: "alpha_h * (1 - h) - beta_h * h".to_string(),
                },
                TimeDerivative {
                    variable: "n".to_string(),
                    expression: "alpha_n * (1 - n) - beta_n * n".to_string(),
                },
            ],
            
            events: vec![
                Event {
                    port: "spike".to_string(),
                    condition: "v > 0".to_string(),
                },
            ],
            
            conditional_derived: vec![],
        };
        
        self.component_definitions.insert("HodgkinHuxleyCell".to_string(), hh_component);
    }
    
    /// Add a component instance to the simulation
    pub fn add_component(&mut self, instance_id: String, component_type: &str) 
        -> Result<(), LEMSError> {
        
        if let Some(definition) = self.component_definitions.get(component_type) {
            let mut state = ComponentState {
                variables: HashMap::new(),
                parameters: HashMap::new(),
                derived: HashMap::new(),
                events: HashMap::new(),
            };
            
            // Initialize state variables
            for var in &definition.state_variables {
                state.variables.insert(var.name.clone(), var.initial_value);
            }
            
            // Initialize parameters
            for param in &definition.parameters {
                if let Some(default) = param.default_value {
                    state.parameters.insert(param.name.clone(), default);
                }
            }
            
            // Initialize events
            for event in &definition.events {
                state.events.insert(event.port.clone(), false);
            }
            
            self.component_states.insert(instance_id, state);
            Ok(())
        } else {
            Err(LEMSError::ComponentNotFound(component_type.to_string()))
        }
    }
    
    /// Set a parameter value for a component
    pub fn set_parameter(&mut self, instance_id: &str, param_name: &str, value: f64) 
        -> Result<(), LEMSError> {
        
        if let Some(state) = self.component_states.get_mut(instance_id) {
            state.parameters.insert(param_name.to_string(), value);
            Ok(())
        } else {
            Err(LEMSError::InstanceNotFound(instance_id.to_string()))
        }
    }
    
    /// Set external current for Hodgkin-Huxley
    pub fn set_external_current(&mut self, instance_id: &str, current: f64) -> Result<(), LEMSError> {
        self.set_parameter(instance_id, "iExt", current)
    }
    
    /// Run one simulation step
    pub fn step(&mut self) -> Result<(), LEMSError> {
        match self.simulation_params.method {
            IntegrationMethod::Euler => self.step_euler(),
            IntegrationMethod::RungeKutta4 => self.step_rk4(),
            IntegrationMethod::AdaptiveRungeKutta => self.step_adaptive_rk(),
        }
    }
    
    fn step_euler(&mut self) -> Result<(), LEMSError> {
        // Store component type for each instance
        let mut instance_types = HashMap::new();
        for (instance_id, _) in &self.component_states {
            // For now, assume all are HH cells - in real implementation, track this
            instance_types.insert(instance_id.clone(), "HodgkinHuxleyCell".to_string());
        }
        
        // Calculate derivatives for all components
        let mut derivatives = HashMap::new();
        
        for (instance_id, state) in &self.component_states {
            if let Some(component_type) = instance_types.get(instance_id) {
                if let Some(definition) = self.component_definitions.get(component_type) {
                    // Update derived variables first
                    let mut derived = HashMap::new();
                    for derived_var in &definition.derived_variables {
                        let value = self.evaluate_expression(&derived_var.expression, state)?;
                        derived.insert(derived_var.name.clone(), value);
                    }
                    
                    // Calculate time derivatives
                    let mut state_with_derived = state.clone();
                    state_with_derived.derived = derived;
                    
                    let mut instance_derivatives = HashMap::new();
                    for time_deriv in &definition.time_derivatives {
                        let deriv_value = self.evaluate_expression(&time_deriv.expression, &state_with_derived)?;
                        instance_derivatives.insert(time_deriv.variable.clone(), deriv_value);
                    }
                    
                    derivatives.insert(instance_id.clone(), instance_derivatives);
                }
            }
        }
        
        // Apply Euler integration
        for (instance_id, instance_derivatives) in derivatives {
            if let Some(state) = self.component_states.get_mut(&instance_id) {
                for (var_name, derivative) in instance_derivatives {
                    if let Some(current_value) = state.variables.get_mut(&var_name) {
                        *current_value += derivative * self.dt;
                    }
                }
            }
        }
        
        // Update time
        self.time += self.dt;
        
        // Check for events
        self.check_events()?;
        
        Ok(())
    }
    
    fn step_rk4(&mut self) -> Result<(), LEMSError> {
        // Runge-Kutta 4th order integration
        // This is more accurate than Euler for oscillatory systems
        
        // Store initial state
        let initial_states: HashMap<String, ComponentState> = self.component_states.clone();
        
        // k1 = f(t, y)
        let k1 = self.calculate_derivatives()?;
        
        // k2 = f(t + dt/2, y + k1*dt/2)
        self.apply_derivatives(&k1, self.dt / 2.0);
        let k2 = self.calculate_derivatives()?;
        
        // k3 = f(t + dt/2, y + k2*dt/2)
        self.component_states = initial_states.clone();
        self.apply_derivatives(&k2, self.dt / 2.0);
        let k3 = self.calculate_derivatives()?;
        
        // k4 = f(t + dt, y + k3*dt)
        self.component_states = initial_states.clone();
        self.apply_derivatives(&k3, self.dt);
        let k4 = self.calculate_derivatives()?;
        
        // y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        self.component_states = initial_states;
        for (instance_id, _) in &self.component_states.clone() {
            if let (Some(k1_vals), Some(k2_vals), Some(k3_vals), Some(k4_vals)) = (
                k1.get(instance_id),
                k2.get(instance_id),
                k3.get(instance_id),
                k4.get(instance_id)
            ) {
                if let Some(state) = self.component_states.get_mut(instance_id) {
                    for var_name in state.variables.keys().cloned().collect::<Vec<_>>() {
                        if let (Some(&k1_val), Some(&k2_val), Some(&k3_val), Some(&k4_val)) = (
                            k1_vals.get(&var_name),
                            k2_vals.get(&var_name),
                            k3_vals.get(&var_name),
                            k4_vals.get(&var_name)
                        ) {
                            let rk4_derivative = (k1_val + 2.0 * k2_val + 2.0 * k3_val + k4_val) / 6.0;
                            if let Some(current_value) = state.variables.get_mut(&var_name) {
                                *current_value += rk4_derivative * self.dt;
                            }
                        }
                    }
                }
            }
        }
        
        self.time += self.dt;
        self.check_events()?;
        
        Ok(())
    }
    
    fn step_adaptive_rk(&mut self) -> Result<(), LEMSError> {
        // Adaptive step size Runge-Kutta
        // Start with current dt, then adapt based on error estimate
        
        let initial_dt = self.dt;
        let mut current_dt = initial_dt;
        let mut step_accepted = false;
        
        while !step_accepted {
            // Store initial state
            let initial_states = self.component_states.clone();
            
            // Take one step with current_dt
            self.dt = current_dt;
            let _ = self.step_rk4();
            let state_full_step = self.component_states.clone();
            
            // Take two steps with current_dt/2
            self.component_states = initial_states.clone();
            self.dt = current_dt / 2.0;
            let _ = self.step_rk4();
            let _ = self.step_rk4();
            let state_half_steps = self.component_states.clone();
            
            // Estimate error
            let mut max_error = 0.0;
            for (instance_id, state_full) in &state_full_step {
                if let Some(state_half) = state_half_steps.get(instance_id) {
                    for (var_name, &val_full) in &state_full.variables {
                        if let Some(&val_half) = state_half.variables.get(var_name) {
                            let error = ((val_full - val_half) / 15.0).abs(); // RK4 error estimate
                            max_error = max_error.max(error);
                        }
                    }
                }
            }
            
            // Check if error is acceptable
            if max_error < self.simulation_params.abs_tolerance {
                step_accepted = true;
                self.component_states = state_half_steps; // Use more accurate result
                self.time += current_dt;
            } else {
                // Reduce timestep
                current_dt *= 0.8 * (self.simulation_params.abs_tolerance / max_error).powf(0.2);
                current_dt = current_dt.max(self.simulation_params.min_dt);
            }
            
            // Prevent infinite loop
            if current_dt < self.simulation_params.min_dt {
                step_accepted = true;
                self.component_states = state_full_step;
                self.time += current_dt;
            }
        }
        
        // Restore original dt for next step
        self.dt = initial_dt;
        
        Ok(())
    }
    
    fn calculate_derivatives(&self) -> Result<HashMap<String, HashMap<String, f64>>, LEMSError> {
        let mut derivatives = HashMap::new();
        
        for (instance_id, state) in &self.component_states {
            // For now, assume all are HH cells
            if let Some(definition) = self.component_definitions.get("HodgkinHuxleyCell") {
                // Update derived variables
                let mut derived = HashMap::new();
                for derived_var in &definition.derived_variables {
                    let value = self.evaluate_expression(&derived_var.expression, state)?;
                    derived.insert(derived_var.name.clone(), value);
                }
                
                // Calculate time derivatives
                let mut state_with_derived = state.clone();
                state_with_derived.derived = derived;
                
                let mut instance_derivatives = HashMap::new();
                for time_deriv in &definition.time_derivatives {
                    let deriv_value = self.evaluate_expression(&time_deriv.expression, &state_with_derived)?;
                    instance_derivatives.insert(time_deriv.variable.clone(), deriv_value);
                }
                
                derivatives.insert(instance_id.clone(), instance_derivatives);
            }
        }
        
        Ok(derivatives)
    }
    
    fn apply_derivatives(&mut self, derivatives: &HashMap<String, HashMap<String, f64>>, dt: f64) {
        for (instance_id, instance_derivatives) in derivatives {
            if let Some(state) = self.component_states.get_mut(instance_id) {
                for (var_name, &derivative) in instance_derivatives {
                    if let Some(current_value) = state.variables.get_mut(var_name) {
                        *current_value += derivative * dt;
                    }
                }
            }
        }
    }
    
    fn check_events(&mut self) -> Result<(), LEMSError> {
        for (instance_id, state) in &mut self.component_states {
            // For now, just check HH spike event
            if let Some(v) = state.variables.get("v") {
                let spiked = *v > 0.0;
                state.events.insert("spike".to_string(), spiked);
            }
        }
        Ok(())
    }
    
    /// Evaluate LEMS expressions using proper parser
    fn evaluate_expression(&self, expression: &str, state: &ComponentState) -> Result<f64, LEMSError> {
        // Create evaluation context from component state
        let mut context = EvaluationContext::new();
        
        // Add all state variables
        for (name, &value) in &state.variables {
            context.set_variable(name, value);
        }
        
        // Add all parameters
        for (name, &value) in &state.parameters {
            context.set_variable(name, value);
        }
        
        // Add all derived variables
        for (name, &value) in &state.derived {
            context.set_variable(name, value);
        }
        
        // Evaluate using expression parser
        self.expression_parser.evaluate_string(expression, &context)
            .map_err(|e| LEMSError::ParseError(format!("Expression evaluation failed: {}", e)))
    }
    
    // This method is now deprecated - using proper expression parser instead
    #[deprecated(note = "Use evaluate_expression with proper parser instead")]
    fn evaluate_arithmetic(&self, expr: &str, state: &ComponentState) -> Result<f64, LEMSError> {
        // Fallback to proper expression parser
        self.evaluate_expression(expr, state)
    }
    
    /// Get the current state of a component
    pub fn get_component_state(&self, instance_id: &str) -> Option<&ComponentState> {
        self.component_states.get(instance_id)
    }
    
    /// Get current simulation time
    pub fn get_time(&self) -> f64 {
        self.time
    }
    
    /// Check if component had an event this timestep
    pub fn check_event(&self, instance_id: &str, event_port: &str) -> bool {
        if let Some(state) = self.component_states.get(instance_id) {
            state.events.get(event_port).copied().unwrap_or(false)
        } else {
            false
        }
    }
}

impl SimulationParams {
    pub fn default() -> Self {
        SimulationParams {
            method: IntegrationMethod::RungeKutta4,
            abs_tolerance: 1e-6,
            rel_tolerance: 1e-3,
            max_dt: 0.1,
            min_dt: 1e-6,
        }
    }
    
    pub fn high_accuracy() -> Self {
        SimulationParams {
            method: IntegrationMethod::AdaptiveRungeKutta,
            abs_tolerance: 1e-8,
            rel_tolerance: 1e-6,
            max_dt: 0.01,
            min_dt: 1e-9,
        }
    }
}

#[derive(Debug)]
pub enum LEMSError {
    ComponentNotFound(String),
    InstanceNotFound(String),
    ParseError(String),
    IntegrationError(String),
}

impl std::fmt::Display for LEMSError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LEMSError::ComponentNotFound(name) => write!(f, "Component type not found: {}", name),
            LEMSError::InstanceNotFound(id) => write!(f, "Component instance not found: {}", id),
            LEMSError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            LEMSError::IntegrationError(msg) => write!(f, "Integration error: {}", msg),
        }
    }
}

impl std::error::Error for LEMSError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hodgkin_huxley_component() {
        let mut engine = LEMSEngine::with_hodgkin_huxley();
        
        // Add HH neuron instance
        engine.add_component("neuron1".to_string(), "HodgkinHuxleyCell").unwrap();
        
        // Check initial state
        let state = engine.get_component_state("neuron1").unwrap();
        assert!((state.variables["v"] - (-65.0)).abs() < 1e-6);
        assert!((state.variables["m"] - 0.05).abs() < 1e-6);
    }
    
    #[test]
    fn test_hodgkin_huxley_spike() {
        let mut engine = LEMSEngine::with_hodgkin_huxley();
        engine.add_component("neuron1".to_string(), "HodgkinHuxleyCell").unwrap();
        
        // Inject strong current
        engine.set_external_current("neuron1", 10.0).unwrap();
        
        // Run simulation until spike
        let mut spiked = false;
        for _ in 0..1000 {
            engine.step().unwrap();
            if engine.check_event("neuron1", "spike") {
                spiked = true;
                break;
            }
        }
        
        assert!(spiked, "Hodgkin-Huxley neuron should spike with strong current");
    }
    
    #[test]
    fn test_integration_methods() {
        let mut euler_engine = LEMSEngine::new(0.01);
        euler_engine.load_hodgkin_huxley_component();
        euler_engine.simulation_params.method = IntegrationMethod::Euler;
        euler_engine.add_component("neuron".to_string(), "HodgkinHuxleyCell").unwrap();
        
        let mut rk4_engine = LEMSEngine::new(0.01);
        rk4_engine.load_hodgkin_huxley_component();
        rk4_engine.simulation_params.method = IntegrationMethod::RungeKutta4;
        rk4_engine.add_component("neuron".to_string(), "HodgkinHuxleyCell").unwrap();
        
        // Both should work without errors
        for _ in 0..100 {
            euler_engine.step().unwrap();
            rk4_engine.step().unwrap();
        }
        
        // RK4 should be more accurate (closer to steady state)
        let euler_v = euler_engine.get_component_state("neuron").unwrap().variables["v"];
        let rk4_v = rk4_engine.get_component_state("neuron").unwrap().variables["v"];
        
        // Both should be close to resting potential
        assert!((euler_v - (-65.0)).abs() < 10.0);
        assert!((rk4_v - (-65.0)).abs() < 10.0);
    }
}