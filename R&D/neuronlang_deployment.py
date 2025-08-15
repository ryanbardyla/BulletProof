#!/usr/bin/env python3
"""
NeuronLang Survey System - Production Deployment Manager
Handles environment setup, dependency installation, and system initialization
Author: NeuronLang Development Team
Version: 1.0.0
"""

import os
import sys
import subprocess
import json
import logging
import argparse
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import venv
import sqlite3
from datetime import datetime
import hashlib
import secrets
import yaml


class DeploymentManager:
    """Manages the deployment and setup of the NeuronLang Survey System"""
    
    def __init__(self, project_root: Path = Path.cwd()):
        self.project_root = project_root
        self.venv_path = project_root / "venv"
        self.config_path = project_root / "config"
        self.data_path = project_root / "data"
        self.logs_path = project_root / "logs"
        self.backups_path = project_root / "backups"
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # System requirements
        self.min_python_version = (3, 9)
        self.required_space_gb = 5  # Minimum disk space in GB
        
    def setup_logging(self):
        """Configure logging for deployment"""
        log_file = self.project_root / "deployment.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def check_system_requirements(self) -> Tuple[bool, List[str]]:
        """Check if system meets requirements"""
        issues = []
        
        # Check Python version
        current_version = sys.version_info[:2]
        if current_version < self.min_python_version:
            issues.append(f"Python {self.min_python_version[0]}.{self.min_python_version[1]}+ required, found {current_version[0]}.{current_version[1]}")
        
        # Check available disk space
        stat = shutil.disk_usage(self.project_root)
        free_gb = stat.free / (1024 ** 3)
        if free_gb < self.required_space_gb:
            issues.append(f"Insufficient disk space: {free_gb:.2f}GB available, {self.required_space_gb}GB required")
        
        # Check for required system commands
        required_commands = ['git', 'pip']
        for cmd in required_commands:
            if not shutil.which(cmd):
                issues.append(f"Required command '{cmd}' not found in PATH")
        
        # Check internet connectivity
        try:
            import urllib.request
            urllib.request.urlopen('https://pypi.org', timeout=5)
        except:
            issues.append("No internet connection detected - required for package installation")
        
        return len(issues) == 0, issues
    
    def create_directory_structure(self):
        """Create project directory structure"""
        directories = [
            self.config_path,
            self.data_path,
            self.logs_path,
            self.backups_path,
            self.data_path / "responses",
            self.data_path / "analysis",
            self.data_path / "reports",
            self.config_path / "models",
            self.config_path / "prompts"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    def setup_virtual_environment(self) -> bool:
        """Create and setup Python virtual environment"""
        try:
            self.logger.info("Creating virtual environment...")
            
            # Create venv
            venv.create(self.venv_path, with_pip=True)
            
            # Get pip path
            if platform.system() == "Windows":
                pip_path = self.venv_path / "Scripts" / "pip.exe"
                python_path = self.venv_path / "Scripts" / "python.exe"
            else:
                pip_path = self.venv_path / "bin" / "pip"
                python_path = self.venv_path / "bin" / "python"
            
            # Upgrade pip
            subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            # Install requirements
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                self.logger.info("Installing dependencies...")
                subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], check=True)
                
                # Download spaCy model
                subprocess.run([str(python_path), "-m", "spacy", "download", "en_core_web_sm"], check=True)
            else:
                self.logger.warning("requirements.txt not found - skipping package installation")
            
            self.logger.info("Virtual environment setup complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup virtual environment: {e}")
            return False
    
    def create_configuration_files(self):
        """Create configuration files with secure defaults"""
        
        # Main configuration
        main_config = {
            "environment": "production",
            "api_keys": {
                "openai": os.getenv("OPENAI_API_KEY", ""),
                "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
                "google": os.getenv("GOOGLE_API_KEY", ""),
                "replicate": os.getenv("REPLICATE_API_KEY", ""),
                "cohere": os.getenv("COHERE_API_KEY", "")
            },
            "models_to_survey": [
                "gpt-4",
                "claude-3-opus",
                "gemini-pro"
            ],
            "survey_settings": {
                "max_concurrent_requests": 5,
                "temperature": 0.7,
                "max_retries": 3,
                "timeout_seconds": 30,
                "save_raw_responses": True,
                "batch_size": 10
            },
            "database": {
                "path": str(self.data_path / "neuronlang_survey.db"),
                "backup_enabled": True,
                "backup_frequency_hours": 24
            },
            "logging": {
                "level": "INFO",
                "max_file_size_mb": 100,
                "max_backup_count": 10,
                "log_directory": str(self.logs_path)
            },
            "security": {
                "api_key_encryption": True,
                "ssl_verify": True,
                "data_encryption_at_rest": False
            },
            "monitoring": {
                "enabled": True,
                "metrics_port": 9090,
                "health_check_interval_seconds": 60
            }
        }
        
        config_file = self.project_root / "config.json"
        with open(config_file, 'w') as f:
            json.dump(main_config, f, indent=2)
        self.logger.info(f"Created configuration file: {config_file}")
        
        # Environment variables template
        env_template = """# NeuronLang Survey System Environment Variables
# Copy this file to .env and fill in your actual API keys

# API Keys (Required for surveying respective models)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
REPLICATE_API_KEY=your_replicate_api_key_here
COHERE_API_KEY=your_cohere_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///data/neuronlang_survey.db
DATABASE_BACKUP_ENABLED=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE_MAX_SIZE_MB=100
LOG_BACKUP_COUNT=10

# Survey Configuration
MAX_CONCURRENT_REQUESTS=5
DEFAULT_TEMPERATURE=0.7
REQUEST_TIMEOUT_SECONDS=30

# Security Configuration
ENCRYPT_API_KEYS=true
SSL_VERIFY=true

# Monitoring Configuration
MONITORING_ENABLED=true
METRICS_PORT=9090
HEALTH_CHECK_PORT=8080

# Development/Production Mode
ENVIRONMENT=production
DEBUG=false
"""
        
        env_file = self.project_root / ".env.template"
        with open(env_file, 'w') as f:
            f.write(env_template)
        self.logger.info(f"Created environment template: {env_file}")
        
        # Docker configuration
        dockerfile_content = """FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config /app/backups

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import sys; sys.exit(0)"

# Run the application
CMD ["python", "neuronlang_survey_orchestrator.py"]
"""
        
        dockerfile = self.project_root / "Dockerfile"
        with open(dockerfile, 'w') as f:
            f.write(dockerfile_content)
        self.logger.info(f"Created Dockerfile: {dockerfile}")
        
        # Docker Compose configuration
        docker_compose_content = """version: '3.8'

services:
  neuronlang_survey:
    build: .
    container_name: neuronlang_survey_system
    restart: unless-stopped
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
      - ./backups:/app/backups
    env_file:
      - .env
    ports:
      - "9090:9090"  # Metrics port
      - "8080:8080"  # Health check port
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - neuronlang_network
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  # Optional: Prometheus for monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: neuronlang_prometheus
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9091:9090"
    networks:
      - neuronlang_network
    depends_on:
      - neuronlang_survey

  # Optional: Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: neuronlang_grafana
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    networks:
      - neuronlang_network
    depends_on:
      - prometheus
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

networks:
  neuronlang_network:
    driver: bridge

volumes:
  prometheus_data:
  grafana_data:
"""
        
        docker_compose_file = self.project_root / "docker-compose.yml"
        with open(docker_compose_file, 'w') as f:
            f.write(docker_compose_content)
        self.logger.info(f"Created Docker Compose file: {docker_compose_file}")
    
    def initialize_database(self):
        """Initialize the survey database"""
        db_path = self.data_path / "neuronlang_survey.db"
        
        # Create database connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Additional tables for production use
        
        # API usage tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_provider TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_used INTEGER,
                cost_usd REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Survey sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS survey_sessions (
                session_id TEXT PRIMARY KEY,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                models_surveyed TEXT,
                questions_asked TEXT,
                total_responses INTEGER,
                status TEXT DEFAULT 'in_progress',
                error_log TEXT
            )
        ''')
        
        # Model performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                avg_response_time_ms REAL,
                success_rate REAL,
                avg_token_count INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Feature extraction results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extracted_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT NOT NULL,
                feature_description TEXT,
                frequency INTEGER,
                models_suggesting TEXT,
                category TEXT,
                implementation_difficulty TEXT,
                priority_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Database initialized at: {db_path}")
    
    def create_monitoring_configuration(self):
        """Create Prometheus monitoring configuration"""
        prometheus_config = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'neuronlang_survey'
    static_configs:
      - targets: ['neuronlang_survey:9090']
    metrics_path: '/metrics'
    
  - job_name: 'node_exporter'
    static_configs:
      - targets: ['localhost:9100']
"""
        
        prometheus_file = self.config_path / "prometheus.yml"
        with open(prometheus_file, 'w') as f:
            f.write(prometheus_config)
        self.logger.info(f"Created Prometheus configuration: {prometheus_file}")
    
    def create_backup_script(self):
        """Create automated backup script"""
        backup_script = """#!/usr/bin/env python3
'''
Automated backup script for NeuronLang Survey System
Runs periodically to backup database and configuration
'''

import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
import json
import gzip
import hashlib

class BackupManager:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.backup_dir = self.project_root / "backups"
        self.data_dir = self.project_root / "data"
        self.config_dir = self.project_root / "config"
        
    def create_backup(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / timestamp
        backup_subdir.mkdir(parents=True, exist_ok=True)
        
        # Backup database
        db_source = self.data_dir / "neuronlang_survey.db"
        if db_source.exists():
            db_backup = backup_subdir / "neuronlang_survey.db.gz"
            with open(db_source, 'rb') as f_in:
                with gzip.open(db_backup, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Database backed up to: {db_backup}")
        
        # Backup configuration
        config_backup = backup_subdir / "config"
        if self.config_dir.exists():
            shutil.copytree(self.config_dir, config_backup, dirs_exist_ok=True)
            print(f"Configuration backed up to: {config_backup}")
        
        # Create backup manifest
        manifest = {
            "timestamp": timestamp,
            "files_backed_up": list(backup_subdir.glob("**/*")),
            "total_size_mb": sum(f.stat().st_size for f in backup_subdir.glob("**/*") if f.is_file()) / (1024 * 1024)
        }
        
        manifest_file = backup_subdir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        # Clean old backups (keep last 7 days)
        self.cleanup_old_backups(days_to_keep=7)
        
        return backup_subdir
    
    def cleanup_old_backups(self, days_to_keep=7):
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                if backup_dir.stat().st_mtime < cutoff_date:
                    shutil.rmtree(backup_dir)
                    print(f"Removed old backup: {backup_dir}")
    
    def restore_backup(self, backup_timestamp):
        backup_subdir = self.backup_dir / backup_timestamp
        
        if not backup_subdir.exists():
            raise ValueError(f"Backup not found: {backup_timestamp}")
        
        # Restore database
        db_backup = backup_subdir / "neuronlang_survey.db.gz"
        if db_backup.exists():
            db_target = self.data_dir / "neuronlang_survey.db"
            
            # Create backup of current database
            if db_target.exists():
                shutil.copy2(db_target, str(db_target) + ".before_restore")
            
            with gzip.open(db_backup, 'rb') as f_in:
                with open(db_target, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Database restored from: {db_backup}")
        
        # Restore configuration
        config_backup = backup_subdir / "config"
        if config_backup.exists():
            # Create backup of current config
            if self.config_dir.exists():
                shutil.copytree(self.config_dir, str(self.config_dir) + ".before_restore", dirs_exist_ok=True)
            
            shutil.copytree(config_backup, self.config_dir, dirs_exist_ok=True)
            print(f"Configuration restored from: {config_backup}")
        
        print(f"Backup {backup_timestamp} restored successfully")

if __name__ == "__main__":
    import sys
    
    manager = BackupManager()
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        if len(sys.argv) > 2:
            manager.restore_backup(sys.argv[2])
        else:
            print("Usage: backup.py restore <timestamp>")
    else:
        backup_path = manager.create_backup()
        print(f"Backup completed: {backup_path}")
"""
        
        backup_file = self.project_root / "backup.py"
        with open(backup_file, 'w') as f:
            f.write(backup_script)
        
        # Make executable on Unix systems
        if platform.system() != "Windows":
            os.chmod(backup_file, 0o755)
        
        self.logger.info(f"Created backup script: {backup_file}")
    
    def create_systemd_service(self):
        """Create systemd service file for Linux systems"""
        if platform.system() != "Linux":
            self.logger.info("Skipping systemd service creation (not on Linux)")
            return
        
        service_content = f"""[Unit]
Description=NeuronLang AI Survey System
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'neuronlang')}
WorkingDirectory={self.project_root}
Environment="PATH={self.venv_path}/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart={self.venv_path}/bin/python {self.project_root}/neuronlang_survey_orchestrator.py
Restart=on-failure
RestartSec=10
StandardOutput=append:{self.logs_path}/neuronlang.log
StandardError=append:{self.logs_path}/neuronlang_error.log

[Install]
WantedBy=multi-user.target
"""
        
        service_file = self.project_root / "neuronlang.service"
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        self.logger.info(f"Created systemd service file: {service_file}")
        self.logger.info("To install: sudo cp neuronlang.service /etc/systemd/system/ && sudo systemctl enable neuronlang")
    
    def run_health_checks(self) -> Dict[str, bool]:
        """Run system health checks"""
        health_status = {}
        
        # Check virtual environment
        health_status['virtual_environment'] = self.venv_path.exists()
        
        # Check database
        db_path = self.data_path / "neuronlang_survey.db"
        health_status['database'] = db_path.exists()
        
        # Check configuration
        config_file = self.project_root / "config.json"
        health_status['configuration'] = config_file.exists()
        
        # Check API keys configuration
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                api_keys_configured = any(
                    key and key != f"YOUR_{provider.upper()}_KEY"
                    for provider, key in config.get('api_keys', {}).items()
                )
                health_status['api_keys_configured'] = api_keys_configured
        else:
            health_status['api_keys_configured'] = False
        
        # Check disk space
        stat = shutil.disk_usage(self.project_root)
        free_gb = stat.free / (1024 ** 3)
        health_status['sufficient_disk_space'] = free_gb >= 1  # At least 1GB free
        
        # Check network connectivity
        try:
            import urllib.request
            urllib.request.urlopen('https://api.openai.com', timeout=5)
            health_status['network_connectivity'] = True
        except:
            health_status['network_connectivity'] = False
        
        return health_status
    
    def deploy(self, skip_venv: bool = False, skip_docker: bool = False):
        """Main deployment function"""
        print("""
        ╔══════════════════════════════════════════════════════════════╗
        ║          NeuronLang Survey System Deployment Manager        ║
        ║                    Production Setup v1.0.0                  ║
        ╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Check system requirements
        self.logger.info("Checking system requirements...")
        requirements_met, issues = self.check_system_requirements()
        
        if not requirements_met:
            self.logger.error("System requirements not met:")
            for issue in issues:
                self.logger.error(f"  - {issue}")
            return False
        
        self.logger.info("System requirements satisfied")
        
        # Create directory structure
        self.logger.info("Creating directory structure...")
        self.create_directory_structure()
        
        # Setup virtual environment
        if not skip_venv:
            self.logger.info("Setting up virtual environment...")
            if not self.setup_virtual_environment():
                self.logger.error("Failed to setup virtual environment")
                return False
        
        # Create configuration files
        self.logger.info("Creating configuration files...")
        self.create_configuration_files()
        
        # Initialize database
        self.logger.info("Initializing database...")
        self.initialize_database()
        
        # Create monitoring configuration
        self.logger.info("Setting up monitoring...")
        self.create_monitoring_configuration()
        
        # Create backup script
        self.logger.info("Creating backup script...")
        self.create_backup_script()
        
        # Create systemd service
        self.create_systemd_service()
        
        # Run health checks
        self.logger.info("Running health checks...")
        health_status = self.run_health_checks()
        
        print("\n" + "="*60)
        print("DEPLOYMENT SUMMARY")
        print("="*60)
        
        print("\nHealth Check Results:")
        for check, status in health_status.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {check.replace('_', ' ').title()}")
        
        print("\nNext Steps:")
        print("1. Copy .env.template to .env and add your API keys")
        print("2. Review and adjust config.json as needed")
        print("3. Run the survey: python neuronlang_survey_orchestrator.py")
        print("4. (Optional) Deploy with Docker: docker-compose up -d")
        print("5. (Optional) Setup automated backups: crontab -e")
        print("   Add: 0 2 * * * /path/to/backup.py")
        
        if platform.system() == "Linux":
            print("6. (Optional) Install as systemd service:")
            print("   sudo cp neuronlang.service /etc/systemd/system/")
            print("   sudo systemctl enable neuronlang")
            print("   sudo systemctl start neuronlang")
        
        print("\n" + "="*60)
        print("Deployment completed successfully!")
        print("="*60 + "\n")
        
        return True


def main():
    """Main entry point for deployment script"""
    parser = argparse.ArgumentParser(description="NeuronLang Survey System Deployment Manager")
    parser.add_argument('--skip-venv', action='store_true', help='Skip virtual environment setup')
    parser.add_argument('--skip-docker', action='store_true', help='Skip Docker configuration')
    parser.add_argument('--health-check', action='store_true', help='Run health checks only')
    
    args = parser.parse_args()
    
    manager = DeploymentManager()
    
    if args.health_check:
        health_status = manager.run_health_checks()
        print("\nHealth Check Results:")
        for check, status in health_status.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {check.replace('_', ' ').title()}")
    else:
        success = manager.deploy(skip_venv=args.skip_venv, skip_docker=args.skip_docker)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
