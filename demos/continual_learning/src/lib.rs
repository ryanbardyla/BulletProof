pub mod real_implementation;
pub mod data_loader;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_real_backpropagation() {
        // Create a small network
        let mut network = real_implementation::RealTrinaryNetwork::new(vec![10, 5, 3]);
        
        // Create simple training data
        let input = Array1::from_vec(vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]);
        let target = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        
        // Forward pass
        let output = network.forward(&input);
        assert_eq!(output.len(), 3);
        
        // Backward pass
        network.backward(&output, &target);
        
        // Check gradients were computed
        for grad in &network.weight_gradients {
            assert!(grad.iter().any(|&x| x != 0.0), "Gradients should be non-zero");
        }
        
        // Update weights
        network.update_weights(1);
        
        // Verify weights are trinary
        for w in &network.weights {
            for &val in w.iter() {
                assert!(val == -1.0 || val == 0.0 || val == 1.0, 
                       "Weights must be trinary, got {}", val);
            }
        }
        
        println!("✅ Real backpropagation test passed!");
    }
    
    #[test]
    fn test_protein_synthesis() {
        let mut proteins = real_implementation::RealProteinSynthesis::new(10);
        
        // Simulate repeated strong activation
        let strong_activation = vec![1.0; 10];
        for t in 0..20 {
            proteins.update_proteins(&strong_activation, t);
        }
        
        // Check that proteins accumulated
        assert!(proteins.get_protection_factor(0) < 1.0, "Protection should increase with protein levels");
        assert!(proteins.is_consolidated(0) || proteins.is_consolidated(1), 
               "Some neurons should be consolidated after repeated activation");
        
        println!("✅ Protein synthesis test passed!");
    }
    
    #[test]
    fn test_continual_learning() {
        let mut learner = real_implementation::RealContinualLearner::new(10, 5, 3);
        
        // Create two simple tasks
        let task1_data: Vec<(Array1<f32>, Array1<f32>)> = (0..100)
            .map(|i| {
                let input = Array1::from_vec(vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]);
                let mut target = Array1::zeros(3);
                target[i % 3] = 1.0;
                (input, target)
            })
            .collect();
        
        let task2_data: Vec<(Array1<f32>, Array1<f32>)> = (0..100)
            .map(|i| {
                let input = Array1::from_vec(vec![-1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0]);
                let mut target = Array1::zeros(3);
                target[(i + 1) % 3] = 1.0;
                (input, target)
            })
            .collect();
        
        // Train on task 1
        let acc1 = learner.train_task("Task1", &task1_data, 5);
        println!("Task 1 accuracy: {:.1}%", acc1 * 100.0);
        
        // Train on task 2
        let acc2 = learner.train_task("Task2", &task2_data, 5);
        println!("Task 2 accuracy: {:.1}%", acc2 * 100.0);
        
        // Test retention on task 1
        let retention = learner.test_task(&task1_data);
        println!("Task 1 retention: {:.1}%", retention * 100.0);
        
        // Should retain some knowledge (better than random)
        assert!(retention > 0.33, "Should retain better than random chance (33%)");
        
        println!("✅ Continual learning test passed!");
    }
    
    #[test]
    fn test_sparsity() {
        let network = real_implementation::RealTrinaryNetwork::new(vec![100, 50, 10]);
        let sparsity = network.sparsity();
        
        println!("Network sparsity: {:.1}%", sparsity * 100.0);
        assert!(sparsity > 0.2, "Should have significant sparsity due to zero initialization");
        
        println!("✅ Sparsity test passed!");
    }
}