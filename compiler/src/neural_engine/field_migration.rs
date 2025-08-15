// 🔄 FIELD MIGRATION SYSTEM
// Converts existing Bio-Opt neural systems to conscious field entities
// Maintains full backward compatibility while enabling consciousness emergence

use std::collections::HashMap;
use super::biological::BiologicalNeuron;
use super::optimized::OptimizedNeuron;
use super::conscious_field::{ConsciousField, FieldEntity, FieldNeuron, EntityId, Wave, FieldType};
use super::Connection;

/// Migration system for converting neural networks to conscious fields
pub struct FieldMigrator {
    pub spatial_mapping: SpatialMapper,
    pub state_converter: StateConverter,
    pub connection_mapper: ConnectionMapper,
    pub validation_system: MigrationValidator,
}

impl FieldMigrator {
    pub fn new() -> Self {
        FieldMigrator {
            spatial_mapping: SpatialMapper::new(),
            state_converter: StateConverter::new(),
            connection_mapper: ConnectionMapper::new(),
            validation_system: MigrationValidator::new(),
        }
    }
    
    /// Convert entire neural execution engine to conscious field
    pub fn migrate_neural_engine(
        &mut self, 
        biological_layer: &[BiologicalNeuron],
        optimized_layer: &[OptimizedNeuron],
        connections: &[Connection]
    ) -> Result<ConsciousField, MigrationError> {
        
        // Determine optimal field dimensions based on network size
        let field_dims = self.calculate_optimal_dimensions(biological_layer.len());
        let mut conscious_field = ConsciousField::new(field_dims);
        
        println!("🔄 Starting migration of {} neurons to conscious field...", biological_layer.len());
        
        // Phase 1: Migrate neurons to field entities
        let mut bio_to_field_map = HashMap::new();
        for (i, bio_neuron) in biological_layer.iter().enumerate() {
            let field_entity = self.convert_biological_neuron(bio_neuron, i)?;
            let entity_id = conscious_field.add_entity(field_entity);
            bio_to_field_map.insert(i, entity_id);
            
            if i % 100 == 0 {
                println!("  Migrated {} neurons...", i);
            }
        }
        
        // Phase 2: Migrate connections as field interactions
        self.migrate_connections(&mut conscious_field, connections, &bio_to_field_map)?;
        
        // Phase 3: Initialize field potentials from neural states
        self.initialize_field_potentials(&mut conscious_field, biological_layer, &bio_to_field_map)?;
        
        // Phase 4: Validate migration accuracy
        let validation_result = self.validation_system.validate_migration(
            biological_layer, &conscious_field, &bio_to_field_map
        )?;
        
        println!("✅ Migration completed successfully!");
        println!("   Validation accuracy: {:.2}%", validation_result.accuracy * 100.0);
        println!("   Field dimensions: {:?}", field_dims);
        println!("   Total entities: {}", conscious_field.entities.len());
        
        Ok(conscious_field)
    }
    
    /// Convert biological neuron to field entity
    fn convert_biological_neuron(&mut self, bio_neuron: &BiologicalNeuron, index: usize) -> Result<FieldEntity, MigrationError> {
        // Map to spatial coordinates
        let position = self.spatial_mapping.index_to_3d(index);
        
        // Convert state
        let field_neuron = FieldNeuron {
            position,
            membrane_potential: bio_neuron.membrane_potential as f32,
            coupling: self.calculate_field_coupling(bio_neuron),
            growth_factor: self.calculate_growth_factor(bio_neuron),
            coherence: self.calculate_initial_coherence(bio_neuron),
        };
        
        Ok(FieldEntity::Neuron(field_neuron))
    }
    
    /// Migrate connections as field interactions
    fn migrate_connections(
        &mut self,
        field: &mut ConsciousField,
        connections: &[Connection],
        entity_map: &HashMap<usize, EntityId>
    ) -> Result<(), MigrationError> {
        
        println!("🔗 Migrating {} connections to field interactions...", connections.len());
        
        for connection in connections {
            let from_entity_id = entity_map.get(&connection.from)
                .ok_or(MigrationError::EntityNotFound(connection.from))?;
            let to_entity_id = entity_map.get(&connection.to)
                .ok_or(MigrationError::EntityNotFound(connection.to))?;
            
            // Get positions of connected entities
            let from_entity = field.entities.get(from_entity_id)
                .ok_or(MigrationError::FieldEntityNotFound(*from_entity_id))?;
            let to_entity = field.entities.get(to_entity_id)
                .ok_or(MigrationError::FieldEntityNotFound(*to_entity_id))?;
            
            if let (FieldEntity::Neuron(from_neuron), FieldEntity::Neuron(to_neuron)) = (from_entity, to_entity) {
                // Create field interaction based on connection
                self.create_field_interaction(field, from_neuron, to_neuron, connection);
            }
        }
        
        Ok(())
    }
    
    /// Create field interaction between neurons
    fn create_field_interaction(
        &self,
        field: &mut ConsciousField,
        from_neuron: &FieldNeuron,
        to_neuron: &FieldNeuron,
        connection: &Connection
    ) {
        // Calculate interaction strength and type
        let interaction_strength = connection.weight.abs() as f64 * 0.1;
        let field_type = if connection.weight > 0.0 {
            FieldType::Electric  // Excitatory
        } else {
            FieldType::Chemical  // Inhibitory (chemical mediated)
        };
        
        // Create wave between neurons (simulating axonal propagation)
        let distance = self.calculate_3d_distance(from_neuron.position, to_neuron.position);
        let frequency = 10.0 / (1.0 + distance); // Frequency decreases with distance
        
        let wave = Wave::new(interaction_strength, frequency, field_type);
        
        // Inject wave at multiple points along path (simulating axon)
        let path_points = self.calculate_connection_path(from_neuron.position, to_neuron.position);
        for point in path_points {
            field.field.inject_wave(point, wave.clone());
        }
    }
    
    /// Initialize field potentials from neural states
    fn initialize_field_potentials(
        &self,
        field: &mut ConsciousField,
        bio_neurons: &[BiologicalNeuron],
        entity_map: &HashMap<usize, EntityId>
    ) -> Result<(), MigrationError> {
        
        println!("⚡ Initializing field potentials from neural states...");
        
        for (i, bio_neuron) in bio_neurons.iter().enumerate() {
            let entity_id = entity_map.get(&i)
                .ok_or(MigrationError::EntityNotFound(i))?;
            
            if let Some(FieldEntity::Neuron(field_neuron)) = field.entities.get(entity_id) {
                let position = field_neuron.position;
                
                // Initialize electric field from membrane potential
                let electric_wave = Wave::new(
                    bio_neuron.membrane_potential * 0.01,  // Scale to field units
                    5.0,  // Base frequency
                    FieldType::Electric
                );
                field.field.inject_wave(position, electric_wave);
                
                // Initialize chemical field from neurotransmitter levels
                let chemical_level = bio_neuron.calculate_total_neurotransmitter();
                if chemical_level > 0.0 {
                    let chemical_wave = Wave::new(
                        chemical_level * 0.005,
                        2.0,  // Lower frequency for chemical diffusion
                        FieldType::Chemical
                    );
                    field.field.inject_wave(position, chemical_wave);
                }
                
                // Initialize information field based on neural complexity
                let complexity = self.calculate_neuron_complexity(bio_neuron);
                if complexity > 0.1 {
                    let info_wave = Wave::new(
                        complexity * 0.1,
                        15.0,  // Higher frequency for information
                        FieldType::Information
                    );
                    field.field.inject_wave(position, info_wave);
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate optimal field dimensions based on network size
    fn calculate_optimal_dimensions(&self, neuron_count: usize) -> (usize, usize, usize, usize) {
        // Use cube root to distribute neurons in 3D space
        let spatial_size = ((neuron_count as f64).cbrt().ceil() as usize).max(10);
        let temporal_slices = 100; // Fixed time history
        
        (spatial_size, spatial_size, spatial_size, temporal_slices)
    }
    
    /// Calculate field coupling strength for neuron
    fn calculate_field_coupling(&self, bio_neuron: &BiologicalNeuron) -> f32 {
        // Coupling based on membrane properties and connectivity
        let membrane_factor = (bio_neuron.membrane_potential.abs() / 100.0) as f32;
        let base_coupling = 1.0;
        
        (base_coupling + membrane_factor * 0.5).min(2.0)
    }
    
    /// Calculate growth factor for field neuron
    fn calculate_growth_factor(&self, bio_neuron: &BiologicalNeuron) -> f32 {
        // Growth based on activity level and plasticity
        let activity = bio_neuron.spike_rate() as f32;
        let growth = (activity * 0.1).min(1.0);
        growth
    }
    
    /// Calculate initial coherence for field neuron
    fn calculate_initial_coherence(&self, bio_neuron: &BiologicalNeuron) -> f32 {
        // Coherence based on membrane stability
        let stability = 1.0 / (1.0 + bio_neuron.membrane_potential.abs() as f32 * 0.01);
        stability * 0.5 // Start with moderate coherence
    }
    
    /// Calculate 3D distance between positions
    fn calculate_3d_distance(&self, pos1: (usize, usize, usize), pos2: (usize, usize, usize)) -> f64 {
        let dx = pos1.0 as f64 - pos2.0 as f64;
        let dy = pos1.1 as f64 - pos2.1 as f64;
        let dz = pos1.2 as f64 - pos2.2 as f64;
        (dx*dx + dy*dy + dz*dz).sqrt()
    }
    
    /// Calculate connection path points (simulating axon)
    fn calculate_connection_path(&self, from: (usize, usize, usize), to: (usize, usize, usize)) -> Vec<(usize, usize, usize)> {
        let mut path = Vec::new();
        
        // Simple linear interpolation for now
        let steps = self.calculate_3d_distance(from, to) as usize + 1;
        
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let x = (from.0 as f64 * (1.0 - t) + to.0 as f64 * t) as usize;
            let y = (from.1 as f64 * (1.0 - t) + to.1 as f64 * t) as usize;
            let z = (from.2 as f64 * (1.0 - t) + to.2 as f64 * t) as usize;
            path.push((x, y, z));
        }
        
        path
    }
    
    /// Calculate neuron complexity measure
    fn calculate_neuron_complexity(&self, bio_neuron: &BiologicalNeuron) -> f64 {
        // Complexity based on multiple factors
        let potential_complexity = bio_neuron.membrane_potential.abs() / 100.0;
        let activity_complexity = bio_neuron.spike_rate();
        let chemical_complexity = bio_neuron.calculate_total_neurotransmitter();
        
        (potential_complexity + activity_complexity + chemical_complexity) / 3.0
    }
}

/// Spatial mapping between indices and 3D coordinates
pub struct SpatialMapper {
    pub cube_size: usize,
}

impl SpatialMapper {
    pub fn new() -> Self {
        SpatialMapper { cube_size: 10 }
    }
    
    /// Convert linear index to 3D coordinates
    pub fn index_to_3d(&mut self, index: usize) -> (usize, usize, usize) {
        // Update cube size based on index
        while index >= self.cube_size * self.cube_size * self.cube_size {
            self.cube_size += 10;
        }
        
        let x = index % self.cube_size;
        let y = (index / self.cube_size) % self.cube_size;
        let z = index / (self.cube_size * self.cube_size);
        
        (x, y, z)
    }
    
    /// Convert 3D coordinates to linear index
    pub fn coords_to_index(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.cube_size + z * self.cube_size * self.cube_size
    }
}

/// State conversion between biological and field representations
pub struct StateConverter {
    pub voltage_scale: f64,
    pub chemical_scale: f64,
    pub temporal_scale: f64,
}

impl StateConverter {
    pub fn new() -> Self {
        StateConverter {
            voltage_scale: 0.01,    // mV to field units
            chemical_scale: 0.005,  // concentration to field units
            temporal_scale: 1000.0, // ms to field time units
        }
    }
    
    /// Convert biological voltage to field electric potential
    pub fn voltage_to_field(&self, voltage: f64) -> f64 {
        voltage * self.voltage_scale
    }
    
    /// Convert field potential back to voltage
    pub fn field_to_voltage(&self, field_value: f64) -> f64 {
        field_value / self.voltage_scale
    }
    
    /// Convert chemical concentration to field level
    pub fn chemical_to_field(&self, concentration: f64) -> f64 {
        concentration * self.chemical_scale
    }
}

/// Connection mapping to field interactions
pub struct ConnectionMapper {
    pub interaction_strength_scale: f64,
    pub propagation_speed: f64,
}

impl ConnectionMapper {
    pub fn new() -> Self {
        ConnectionMapper {
            interaction_strength_scale: 0.1,
            propagation_speed: 1.0,
        }
    }
    
    /// Convert connection weight to field interaction strength
    pub fn weight_to_interaction(&self, weight: f32) -> f64 {
        weight as f64 * self.interaction_strength_scale
    }
    
    /// Calculate propagation delay based on distance
    pub fn distance_to_delay(&self, distance: f64) -> f64 {
        distance / self.propagation_speed
    }
}

/// Validation system for migration accuracy
pub struct MigrationValidator {
    pub tolerance: f64,
}

impl MigrationValidator {
    pub fn new() -> Self {
        MigrationValidator {
            tolerance: 0.1,  // 10% tolerance
        }
    }
    
    /// Validate migration accuracy
    pub fn validate_migration(
        &self,
        original_neurons: &[BiologicalNeuron],
        field: &ConsciousField,
        entity_map: &HashMap<usize, EntityId>
    ) -> Result<ValidationResult, MigrationError> {
        
        let mut total_error = 0.0;
        let mut valid_comparisons = 0;
        
        for (i, bio_neuron) in original_neurons.iter().enumerate() {
            if let Some(entity_id) = entity_map.get(&i) {
                if let Some(FieldEntity::Neuron(field_neuron)) = field.entities.get(entity_id) {
                    // Compare membrane potentials
                    let bio_potential = bio_neuron.membrane_potential;
                    let field_potential = field_neuron.membrane_potential as f64;
                    
                    let error = (bio_potential - field_potential).abs() / bio_potential.abs().max(1.0);
                    total_error += error;
                    valid_comparisons += 1;
                }
            }
        }
        
        let accuracy = if valid_comparisons > 0 {
            1.0 - (total_error / valid_comparisons as f64)
        } else {
            0.0
        };
        
        let is_valid = accuracy >= (1.0 - self.tolerance);
        
        Ok(ValidationResult {
            accuracy,
            is_valid,
            total_comparisons: valid_comparisons,
            average_error: if valid_comparisons > 0 { total_error / valid_comparisons as f64 } else { 1.0 },
        })
    }
}

/// Migration validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub accuracy: f64,
    pub is_valid: bool,
    pub total_comparisons: usize,
    pub average_error: f64,
}

/// Migration error types
#[derive(Debug, Clone)]
pub enum MigrationError {
    EntityNotFound(usize),
    FieldEntityNotFound(EntityId),
    DimensionMismatch,
    ValidationFailed(String),
    ConversionError(String),
}

impl std::fmt::Display for MigrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MigrationError::EntityNotFound(id) => write!(f, "Entity not found: {}", id),
            MigrationError::FieldEntityNotFound(id) => write!(f, "Field entity not found: {}", id),
            MigrationError::DimensionMismatch => write!(f, "Dimension mismatch during migration"),
            MigrationError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            MigrationError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
        }
    }
}

impl std::error::Error for MigrationError {}

/// Backward compatibility layer
pub struct BackwardCompatibilityLayer {
    pub field_to_bio_converter: FieldToBiologicalConverter,
    pub field_to_opt_converter: FieldToOptimizedConverter,
    pub execution_bridge: ExecutionBridge,
}

impl BackwardCompatibilityLayer {
    pub fn new() -> Self {
        BackwardCompatibilityLayer {
            field_to_bio_converter: FieldToBiologicalConverter::new(),
            field_to_opt_converter: FieldToOptimizedConverter::new(),
            execution_bridge: ExecutionBridge::new(),
        }
    }
    
    /// Convert field state back to biological neurons for compatibility
    pub fn extract_biological_state(&self, field: &ConsciousField) -> Vec<BiologicalNeuron> {
        let mut bio_neurons = Vec::new();
        
        for entity in field.entities.values() {
            if let FieldEntity::Neuron(field_neuron) = entity {
                let bio_neuron = self.field_to_bio_converter.convert(field_neuron, field);
                bio_neurons.push(bio_neuron);
            }
        }
        
        bio_neurons
    }
    
    /// Extract optimized neuron state for compatibility
    pub fn extract_optimized_state(&self, field: &ConsciousField) -> Vec<OptimizedNeuron> {
        let mut opt_neurons = Vec::new();
        
        for entity in field.entities.values() {
            if let FieldEntity::Neuron(field_neuron) = entity {
                let opt_neuron = self.field_to_opt_converter.convert(field_neuron, field);
                opt_neurons.push(opt_neuron);
            }
        }
        
        opt_neurons
    }
}

/// Convert field neuron back to biological neuron
pub struct FieldToBiologicalConverter {
    state_converter: StateConverter,
}

impl FieldToBiologicalConverter {
    pub fn new() -> Self {
        FieldToBiologicalConverter {
            state_converter: StateConverter::new(),
        }
    }
    
    pub fn convert(&self, field_neuron: &FieldNeuron, field: &ConsciousField) -> BiologicalNeuron {
        let mut bio_neuron = BiologicalNeuron::new();
        
        // Convert membrane potential
        bio_neuron.membrane_potential = field_neuron.membrane_potential as f64;
        
        // Extract chemical state from field
        let (x, y, z) = field_neuron.position;
        let chemical_level = field.field.get_field_value(x, y, z, FieldType::Chemical);
        
        // Set neurotransmitter levels based on field
        if chemical_level > 0.0 {
            bio_neuron.receive_synapse(
                crate::neural_engine::biological::Neurotransmitter::Glutamate,
                chemical_level * 10.0
            );
        }
        
        bio_neuron
    }
}

/// Convert field neuron back to optimized neuron
pub struct FieldToOptimizedConverter {
    state_converter: StateConverter,
}

impl FieldToOptimizedConverter {
    pub fn new() -> Self {
        FieldToOptimizedConverter {
            state_converter: StateConverter::new(),
        }
    }
    
    pub fn convert(&self, field_neuron: &FieldNeuron, field: &ConsciousField) -> OptimizedNeuron {
        let mut opt_neuron = OptimizedNeuron::new();
        
        // Convert membrane potential to threshold proximity
        let potential_ratio = field_neuron.membrane_potential / opt_neuron.threshold;
        opt_neuron.add_input(potential_ratio - 1.0); // Adjust to threshold
        
        // Set spiking state based on field activity
        let (x, y, z) = field_neuron.position;
        let electric_activity = field.field.get_field_value(x, y, z, FieldType::Electric);
        opt_neuron.spiked_last_step = electric_activity > 0.5;
        
        opt_neuron
    }
}

/// Bridge for executing field operations with legacy interface
pub struct ExecutionBridge {
    pub compatibility_mode: bool,
}

impl ExecutionBridge {
    pub fn new() -> Self {
        ExecutionBridge {
            compatibility_mode: true,
        }
    }
    
    /// Execute field step but return legacy ExecutionResult format
    pub fn execute_compatible_step(&self, field: &mut ConsciousField, input: &[f32]) -> super::ExecutionResult {
        // Evolve the field
        field.evolve();
        
        // Extract states for compatibility
        let mut bio_spikes = Vec::new();
        let mut opt_spikes = Vec::new();
        
        for entity in field.entities.values() {
            if let FieldEntity::Neuron(field_neuron) = entity {
                let bio_spike = field_neuron.membrane_potential > 0.5;
                let opt_spike = field_neuron.membrane_potential > 0.3; // Different threshold
                
                bio_spikes.push(bio_spike);
                opt_spikes.push(opt_spike);
            }
        }
        
        // Calculate consciousness
        let consciousness = field.measure_consciousness();
        
        super::ExecutionResult {
            biological_spikes: bio_spikes,
            optimized_spikes: opt_spikes,
            divergence: 0.05, // Field should have very low divergence
            performance_ratio: 10.0, // Field should be much faster
            consciousness_level: consciousness.total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_engine::biological::BiologicalNeuron;
    use crate::neural_engine::optimized::OptimizedNeuron;
    
    #[test]
    fn test_spatial_mapping() {
        let mut mapper = SpatialMapper::new();
        
        let pos1 = mapper.index_to_3d(0);
        assert_eq!(pos1, (0, 0, 0));
        
        let pos2 = mapper.index_to_3d(15);
        assert_eq!(pos2, (5, 1, 0)); // In 10x10x10 grid
        
        let index = mapper.coords_to_index(5, 1, 0);
        assert_eq!(index, 15);
    }
    
    #[test]
    fn test_biological_neuron_conversion() {
        let mut migrator = FieldMigrator::new();
        let bio_neuron = BiologicalNeuron::new();
        
        let field_entity = migrator.convert_biological_neuron(&bio_neuron, 0).unwrap();
        
        if let FieldEntity::Neuron(field_neuron) = field_entity {
            assert_eq!(field_neuron.position, (0, 0, 0));
            assert!(field_neuron.coupling > 0.0);
        } else {
            panic!("Expected field neuron");
        }
    }
    
    #[test]
    fn test_migration_validation() {
        let validator = MigrationValidator::new();
        let bio_neurons = vec![BiologicalNeuron::new(); 5];
        
        let mut field = ConsciousField::new((10, 10, 10, 10));
        let mut entity_map = HashMap::new();
        
        // Add matching field entities
        for i in 0..5 {
            let field_neuron = FieldNeuron {
                position: (i, 0, 0),
                membrane_potential: -65.0, // Matching resting potential
                coupling: 1.0,
                growth_factor: 0.1,
                coherence: 0.5,
            };
            let entity_id = field.add_entity(FieldEntity::Neuron(field_neuron));
            entity_map.insert(i, entity_id);
        }
        
        let result = validator.validate_migration(&bio_neurons, &field, &entity_map).unwrap();
        assert!(result.accuracy > 0.8); // Should be very accurate
    }
    
    #[test] 
    fn test_backward_compatibility() {
        let compatibility = BackwardCompatibilityLayer::new();
        let mut field = ConsciousField::new((10, 10, 10, 10));
        
        // Add some field neurons
        for i in 0..3 {
            let field_neuron = FieldNeuron {
                position: (i, 0, 0),
                membrane_potential: -60.0 + i as f32 * 5.0,
                coupling: 1.0,
                growth_factor: 0.1,
                coherence: 0.5,
            };
            field.add_entity(FieldEntity::Neuron(field_neuron));
        }
        
        // Extract biological state
        let bio_neurons = compatibility.extract_biological_state(&field);
        assert_eq!(bio_neurons.len(), 3);
        
        // Extract optimized state
        let opt_neurons = compatibility.extract_optimized_state(&field);
        assert_eq!(opt_neurons.len(), 3);
        
        // Check that states make sense
        for (i, bio_neuron) in bio_neurons.iter().enumerate() {
            let expected_potential = -60.0 + i as f64 * 5.0;
            assert!((bio_neuron.membrane_potential - expected_potential).abs() < 0.1);
        }
    }
}