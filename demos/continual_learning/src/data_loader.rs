// Real data loading - no fake data
use anyhow::Result;
use mnist::{Mnist, MnistBuilder};
use ndarray::Array1;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Load real MNIST dataset
pub fn load_real_mnist() -> Result<(Vec<(Array1<f32>, Array1<f32>)>, Vec<(Array1<f32>, Array1<f32>)>)> {
    println!("📥 Loading real MNIST dataset...");
    
    // Ensure MNIST data is downloaded first
    download_mnist()?;
    
    // Load MNIST using the mnist crate with custom data path
    let Mnist {
        trn_img, trn_lbl,
        tst_img, tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .base_path("data/")
        .finalize();
    
    // Convert to our format
    let train_data: Vec<(Array1<f32>, Array1<f32>)> = trn_img
        .chunks(784)
        .zip(trn_lbl.iter())
        .map(|(img, &label)| {
            // Normalize pixels to [-1, 0, 1] for trinary
            let input = Array1::from_vec(
                img.iter().map(|&x| {
                    let normalized = x as f32 / 255.0;
                    if normalized < 0.33 { -1.0 }
                    else if normalized < 0.66 { 0.0 }
                    else { 1.0 }
                }).collect()
            );
            
            // One-hot encode label
            let mut target = Array1::zeros(10);
            target[label as usize] = 1.0;
            
            (input, target)
        })
        .collect();
    
    let test_data: Vec<(Array1<f32>, Array1<f32>)> = tst_img
        .chunks(784)
        .zip(tst_lbl.iter())
        .map(|(img, &label)| {
            let input = Array1::from_vec(
                img.iter().map(|&x| {
                    let normalized = x as f32 / 255.0;
                    if normalized < 0.33 { -1.0 }
                    else if normalized < 0.66 { 0.0 }
                    else { 1.0 }
                }).collect()
            );
            
            let mut target = Array1::zeros(10);
            target[label as usize] = 1.0;
            
            (input, target)
        })
        .collect();
    
    println!("✅ Loaded {} training and {} test samples", train_data.len(), test_data.len());
    Ok((train_data, test_data))
}

/// Load Fashion-MNIST (structure identical to MNIST)
pub fn load_fashion_mnist() -> Result<(Vec<(Array1<f32>, Array1<f32>)>, Vec<(Array1<f32>, Array1<f32>)>)> {
    anyhow::bail!("Fashion-MNIST not implemented - please use real MNIST dataset for now")
}

/// Load CIFAR-10 (32x32x3 = 3072 dimensions)
pub fn load_cifar10() -> Result<(Vec<(Array1<f32>, Array1<f32>)>, Vec<(Array1<f32>, Array1<f32>)>)> {
    anyhow::bail!("CIFAR-10 not implemented - please use real MNIST dataset for now")
}

/// Download MNIST if not present
fn download_mnist() -> Result<()> {
    println!("📥 Downloading MNIST dataset...");
    
    std::fs::create_dir_all("data")?;
    
    let files = vec![
        ("train-images-idx3-ubyte.gz", "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"),
        ("train-labels-idx1-ubyte.gz", "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"),
        ("t10k-images-idx3-ubyte.gz", "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"),
        ("t10k-labels-idx1-ubyte.gz", "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"),
    ];
    
    for (filename, url) in files {
        let path = format!("data/{}", filename);
        if !Path::new(&path).exists() {
            println!("  Downloading {}...", filename);
            
            // Use curl to download
            let output = std::process::Command::new("curl")
                .args(&["-L", "-o", &path, url])
                .output()?;
            
            if !output.status.success() {
                anyhow::bail!("Failed to download {}", filename);
            }
            
            // Decompress
            let gz_path = path.clone();
            let out_path = path.replace(".gz", "");
            
            let output = std::process::Command::new("gunzip")
                .args(&["-c", &gz_path])
                .output()?;
            
            if output.status.success() {
                std::fs::write(&out_path, output.stdout)?;
                println!("  Decompressed {}", filename);
            } else {
                anyhow::bail!("Failed to decompress {}", filename);
            }
        }
    }
    
    println!("✅ MNIST dataset ready");
    Ok(())
}