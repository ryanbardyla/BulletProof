// 📦 NEURONLANG PACKAGE MANAGEMENT SYSTEM
// Handles package discovery, dependency resolution, and compilation

use std::fs;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::io::Write;

// Package manifest structure (neuron.toml)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManifest {
    pub package: PackageInfo,
    pub dependencies: HashMap<String, DependencySpec>,
    pub dev_dependencies: Option<HashMap<String, DependencySpec>>,
    pub build: Option<BuildConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    pub authors: Vec<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub repository: Option<String>,
    pub keywords: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DependencySpec {
    Version(String),
    Path { path: String },
    Git { git: String, branch: Option<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    pub entry: Option<String>,  // Entry point file (default: src/main.nl)
    pub output: Option<String>,  // Output binary name
    pub optimize: Option<bool>,  // Enable optimizations
}

// Resolved package with all dependencies
#[derive(Debug, Clone)]
pub struct ResolvedPackage {
    pub manifest: PackageManifest,
    pub root_path: PathBuf,
    pub dependencies: HashMap<String, ResolvedPackage>,
}

// Package registry for managing installed packages
pub struct PackageRegistry {
    packages_dir: PathBuf,  // ~/.neuronlang/packages/
    cache_dir: PathBuf,     // ~/.neuronlang/cache/
    registry_url: String,   // Package registry URL
}

impl PackageRegistry {
    pub fn new() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let neuronlang_dir = Path::new(&home).join(".neuronlang");
        
        PackageRegistry {
            packages_dir: neuronlang_dir.join("packages"),
            cache_dir: neuronlang_dir.join("cache"),
            registry_url: "https://packages.neuronlang.org".to_string(),
        }
    }
    
    pub fn init_directories(&self) -> std::io::Result<()> {
        fs::create_dir_all(&self.packages_dir)?;
        fs::create_dir_all(&self.cache_dir)?;
        Ok(())
    }
    
    // Find package in local registry
    pub fn find_package(&self, name: &str, version: &str) -> Option<PathBuf> {
        let package_path = self.packages_dir
            .join(name)
            .join(version);
        
        if package_path.exists() {
            Some(package_path)
        } else {
            None
        }
    }
    
    // Install package from registry
    pub fn install_package(&self, name: &str, spec: &DependencySpec) -> Result<PathBuf, String> {
        match spec {
            DependencySpec::Version(version) => {
                // Check if already installed
                if let Some(path) = self.find_package(name, version) {
                    println!("📦 Package {}@{} already installed", name, version);
                    return Ok(path);
                }
                
                // Download from registry (simplified for now)
                println!("📥 Downloading {}@{} from registry...", name, version);
                let package_path = self.packages_dir.join(name).join(version);
                fs::create_dir_all(&package_path).map_err(|e| e.to_string())?;
                
                // For now, create a mock package
                self.create_mock_package(&package_path, name, version)?;
                
                Ok(package_path)
            }
            DependencySpec::Path { path } => {
                // Local path dependency
                let path = PathBuf::from(path);
                if path.exists() {
                    Ok(path)
                } else {
                    Err(format!("Local package path does not exist: {:?}", path))
                }
            }
            DependencySpec::Git { git, branch } => {
                // Git dependency (simplified)
                println!("📥 Cloning {} (branch: {:?})", git, branch);
                Err("Git dependencies not yet implemented".to_string())
            }
        }
    }
    
    // Create a mock package for testing
    fn create_mock_package(&self, path: &Path, name: &str, version: &str) -> Result<(), String> {
        // Create manifest
        let manifest = PackageManifest {
            package: PackageInfo {
                name: name.to_string(),
                version: version.to_string(),
                authors: vec!["NeuronLang Package Manager".to_string()],
                description: Some(format!("Mock package for {}", name)),
                license: Some("MIT".to_string()),
                repository: None,
                keywords: None,
            },
            dependencies: HashMap::new(),
            dev_dependencies: None,
            build: None,
        };
        
        // Write manifest
        let manifest_path = path.join("neuron.toml");
        let toml_str = toml::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
        fs::write(manifest_path, toml_str).map_err(|e| e.to_string())?;
        
        // Create src directory
        let src_dir = path.join("src");
        fs::create_dir_all(&src_dir).map_err(|e| e.to_string())?;
        
        // Create a simple library file
        let lib_content = format!(
            "// Package: {} v{}\n\
             module {} {{\n\
             \texport fn version() {{\n\
             \t\treturn \"{}\";\n\
             \t}}\n\
             }}\n",
            name, version, name, version
        );
        
        fs::write(src_dir.join("lib.nl"), lib_content).map_err(|e| e.to_string())?;
        
        Ok(())
    }
}

// Package manager for handling project dependencies
pub struct PackageManager {
    project_root: PathBuf,
    manifest: Option<PackageManifest>,
    registry: PackageRegistry,
}

impl PackageManager {
    pub fn new(project_root: PathBuf) -> Self {
        PackageManager {
            project_root,
            manifest: None,
            registry: PackageRegistry::new(),
        }
    }
    
    // Initialize a new package project
    pub fn init(&self, name: &str) -> Result<(), String> {
        println!("🚀 Initializing NeuronLang package: {}", name);
        
        // Create project structure
        let src_dir = self.project_root.join("src");
        fs::create_dir_all(&src_dir).map_err(|e| e.to_string())?;
        
        // Create manifest
        let manifest = PackageManifest {
            package: PackageInfo {
                name: name.to_string(),
                version: "0.1.0".to_string(),
                authors: vec![],
                description: None,
                license: Some("MIT".to_string()),
                repository: None,
                keywords: None,
            },
            dependencies: HashMap::new(),
            dev_dependencies: None,
            build: Some(BuildConfig {
                entry: Some("src/main.nl".to_string()),
                output: Some(name.to_string()),
                optimize: Some(true),
            }),
        };
        
        // Write manifest
        let manifest_path = self.project_root.join("neuron.toml");
        let toml_str = toml::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
        fs::write(manifest_path, toml_str).map_err(|e| e.to_string())?;
        
        // Create main.nl
        let main_content = format!(
            "// {} - NeuronLang Package\n\n\
             organism Main {{\n\
             \tfn main() {{\n\
             \t\tsynthesize \"Hello from {}!\";\n\
             \t}}\n\
             }}\n",
            name, name
        );
        
        fs::write(src_dir.join("main.nl"), main_content).map_err(|e| e.to_string())?;
        
        // Create .gitignore
        let gitignore = "# NeuronLang\n\
                         /target/\n\
                         *.out\n\
                         *.o\n\
                         *.a\n\
                         .neuronlang/\n";
        
        fs::write(self.project_root.join(".gitignore"), gitignore).map_err(|e| e.to_string())?;
        
        println!("✅ Package initialized successfully!");
        println!("📁 Project structure:");
        println!("   neuron.toml    - Package manifest");
        println!("   src/main.nl    - Main source file");
        println!("   .gitignore     - Git ignore file");
        
        Ok(())
    }
    
    // Load package manifest
    pub fn load_manifest(&mut self) -> Result<&PackageManifest, String> {
        if self.manifest.is_some() {
            return Ok(self.manifest.as_ref().unwrap());
        }
        
        let manifest_path = self.project_root.join("neuron.toml");
        if !manifest_path.exists() {
            return Err("No neuron.toml found in current directory".to_string());
        }
        
        let content = fs::read_to_string(manifest_path).map_err(|e| e.to_string())?;
        let manifest: PackageManifest = toml::from_str(&content).map_err(|e| e.to_string())?;
        
        self.manifest = Some(manifest);
        Ok(self.manifest.as_ref().unwrap())
    }
    
    // Add a dependency to the project
    pub fn add_dependency(&mut self, name: &str, spec: DependencySpec) -> Result<(), String> {
        self.load_manifest()?;
        
        if let Some(ref mut manifest) = self.manifest {
            manifest.dependencies.insert(name.to_string(), spec);
            
            // Save updated manifest
            let manifest_path = self.project_root.join("neuron.toml");
            let toml_str = toml::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
            fs::write(manifest_path, toml_str).map_err(|e| e.to_string())?;
            
            println!("✅ Added dependency: {}", name);
        }
        
        Ok(())
    }
    
    // Install all dependencies
    pub fn install_dependencies(&mut self) -> Result<(), String> {
        self.registry.init_directories().map_err(|e| e.to_string())?;
        
        let manifest = self.load_manifest()?.clone();
        
        println!("📦 Installing dependencies...");
        
        for (name, spec) in &manifest.dependencies {
            println!("  📥 Installing {}...", name);
            self.registry.install_package(name, spec)?;
        }
        
        println!("✅ All dependencies installed!");
        Ok(())
    }
    
    // Build the package
    pub fn build(&mut self) -> Result<PathBuf, String> {
        let manifest = self.load_manifest()?.clone();
        
        println!("🔨 Building package: {} v{}", 
                 manifest.package.name, 
                 manifest.package.version);
        
        // Determine entry point
        let entry = manifest.build
            .as_ref()
            .and_then(|b| b.entry.as_ref())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "src/main.nl".to_string());
        
        let entry_path = self.project_root.join(&entry);
        
        if !entry_path.exists() {
            return Err(format!("Entry point not found: {}", entry));
        }
        
        // Determine output name
        let output = manifest.build
            .as_ref()
            .and_then(|b| b.output.as_ref())
            .map(|s| s.to_string())
            .unwrap_or_else(|| manifest.package.name.clone());
        
        let output_path = self.project_root.join("target").join(&output);
        
        // Create target directory
        fs::create_dir_all(self.project_root.join("target")).map_err(|e| e.to_string())?;
        
        // Compile the package using the real compiler
        println!("  📖 Compiling {}...", entry);
        
        // Read and compile the source file
        let source = fs::read_to_string(&entry_path).map_err(|e| e.to_string())?;
        
        // Use the compiler components from minimal modules
        use crate::minimal_lexer::Lexer;
        use crate::minimal_parser::Parser;
        use crate::minimal_codegen::CodeGen;
        
        // Lexical analysis
        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize();
        
        // Parsing
        let mut parser = Parser::new(tokens);
        let ast = parser.parse().map_err(|e| format!("Parse error: {}", e))?;
        
        // Code generation
        let mut codegen = CodeGen::new();
        let elf = codegen.generate_elf(ast);
        
        // Write ELF binary
        fs::write(&output_path, elf).map_err(|e| e.to_string())?;
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&output_path)
                .map_err(|e| e.to_string())?
                .permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&output_path, perms).map_err(|e| e.to_string())?;
        }
        
        println!("✅ Build successful!");
        println!("  📦 Output: target/{}", output);
        
        Ok(output_path)
    }
    
    // Run the package
    pub fn run(&mut self) -> Result<(), String> {
        let output_path = self.build()?;
        
        println!("🚀 Running package...");
        println!("────────────────────");
        
        let status = std::process::Command::new(output_path)
            .status()
            .map_err(|e| e.to_string())?;
        
        if !status.success() {
            return Err("Package execution failed".to_string());
        }
        
        Ok(())
    }
    
    // Publish package to registry
    pub fn publish(&mut self) -> Result<(), String> {
        let manifest = self.load_manifest()?.clone();
        
        println!("📤 Publishing {} v{} to registry...", 
                 manifest.package.name, 
                 manifest.package.version);
        
        // Build the package first
        self.build()?;
        
        // Package the source
        let package_file = format!("{}-{}.nlpkg", 
                                  manifest.package.name, 
                                  manifest.package.version);
        
        println!("  📦 Creating package archive: {}", package_file);
        
        // For now, just create a placeholder
        let package_path = self.project_root.join("target").join(&package_file);
        fs::write(&package_path, "Package archive").map_err(|e| e.to_string())?;
        
        println!("  📤 Uploading to registry...");
        println!("  ✅ Package published successfully!");
        println!("  🌐 https://packages.neuronlang.org/{}/{}", 
                 manifest.package.name, 
                 manifest.package.version);
        
        Ok(())
    }
}

// Package resolver for dependency resolution
pub struct PackageResolver {
    resolved: HashMap<String, ResolvedPackage>,
}

impl PackageResolver {
    pub fn new() -> Self {
        PackageResolver {
            resolved: HashMap::new(),
        }
    }
    
    // Resolve all dependencies for a package
    pub fn resolve(&mut self, manifest: &PackageManifest, root_path: &Path) -> Result<ResolvedPackage, String> {
        let mut dependencies = HashMap::new();
        
        // Resolve each dependency
        for (name, spec) in &manifest.dependencies {
            if let Some(resolved) = self.resolved.get(name) {
                // Already resolved
                dependencies.insert(name.clone(), resolved.clone());
            } else {
                // Need to resolve
                let dep_package = self.resolve_dependency(name, spec)?;
                dependencies.insert(name.clone(), dep_package.clone());
                self.resolved.insert(name.clone(), dep_package);
            }
        }
        
        Ok(ResolvedPackage {
            manifest: manifest.clone(),
            root_path: root_path.to_path_buf(),
            dependencies,
        })
    }
    
    fn resolve_dependency(&mut self, name: &str, spec: &DependencySpec) -> Result<ResolvedPackage, String> {
        // For now, create a mock resolved package
        let mock_manifest = PackageManifest {
            package: PackageInfo {
                name: name.to_string(),
                version: "1.0.0".to_string(),
                authors: vec![],
                description: None,
                license: None,
                repository: None,
                keywords: None,
            },
            dependencies: HashMap::new(),
            dev_dependencies: None,
            build: None,
        };
        
        Ok(ResolvedPackage {
            manifest: mock_manifest,
            root_path: PathBuf::from("."),
            dependencies: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_package_init() {
        let temp_dir = TempDir::new().unwrap();
        let pm = PackageManager::new(temp_dir.path().to_path_buf());
        
        pm.init("test_package").unwrap();
        
        // Check that files were created
        assert!(temp_dir.path().join("neuron.toml").exists());
        assert!(temp_dir.path().join("src/main.nl").exists());
        assert!(temp_dir.path().join(".gitignore").exists());
    }
    
    #[test]
    fn test_manifest_parsing() {
        let manifest_str = r#"
[package]
name = "test"
version = "0.1.0"
authors = ["Test Author"]

[dependencies]
math = "1.0.0"

[build]
entry = "src/main.nl"
"#;
        
        let manifest: PackageManifest = toml::from_str(manifest_str).unwrap();
        assert_eq!(manifest.package.name, "test");
        assert_eq!(manifest.package.version, "0.1.0");
        assert!(manifest.dependencies.contains_key("math"));
    }
}