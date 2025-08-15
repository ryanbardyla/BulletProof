#!/usr/bin/env python3

"""
SECURE GITHUB ASSIMILATOR - With Verification & Safety Checks
Only assimilates VERIFIED, SAFE, HIGH-QUALITY code
Protects the neural network from poisoned data!
"""

import os
import git
import requests
import json
import time
import hashlib
import redis
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

class SecureGitHubAssimilator:
    def __init__(self):
        self.redis_client = redis.Redis(host='192.168.1.30', port=6379, decode_responses=True)
        self.repos_dir = Path("/home/ryan/repo/Fenrisa/BULLETPROOF_PROJECT/assimilated_repos")
        self.repos_dir.mkdir(exist_ok=True)
        
        # Security settings
        self.verified_repos = self.load_verified_repos()
        self.blacklisted_repos = self.load_blacklisted_repos()
        self.suspicious_patterns = self.load_suspicious_patterns()
        
        # Quality thresholds
        self.min_stars = 100  # Minimum GitHub stars
        self.min_age_days = 90  # Repository must be at least 90 days old
        self.max_file_size_mb = 10  # Skip files larger than 10MB
        
        # Tracking
        self.scan_results = []
        self.rejected_repos = []
        self.quarantine_dir = Path("/home/ryan/repo/Fenrisa/BULLETPROOF_PROJECT/quarantine")
        self.quarantine_dir.mkdir(exist_ok=True)
        
    def load_verified_repos(self) -> Dict[str, Dict]:
        """Load list of verified safe repositories"""
        return {
            # Official language repos (VERIFIED)
            "https://github.com/python/cpython": {
                "category": "language",
                "trust_level": 10,
                "description": "Official Python implementation"
            },
            "https://github.com/rust-lang/rust": {
                "category": "language", 
                "trust_level": 10,
                "description": "Official Rust implementation"
            },
            "https://github.com/golang/go": {
                "category": "language",
                "trust_level": 10,
                "description": "Official Go implementation"
            },
            
            # Established frameworks (VERIFIED)
            "https://github.com/django/django": {
                "category": "framework",
                "trust_level": 9,
                "description": "Django web framework"
            },
            "https://github.com/facebook/react": {
                "category": "framework",
                "trust_level": 9,
                "description": "React framework (Meta official)"
            },
            "https://github.com/tensorflow/tensorflow": {
                "category": "ml",
                "trust_level": 9,
                "description": "TensorFlow (Google official)"
            },
            "https://github.com/pytorch/pytorch": {
                "category": "ml",
                "trust_level": 9,
                "description": "PyTorch (Meta official)"
            },
            
            # Trusted systems (VERIFIED)
            "https://github.com/torvalds/linux": {
                "category": "os",
                "trust_level": 10,
                "description": "Linux kernel (Linus Torvalds)"
            },
            "https://github.com/redis/redis": {
                "category": "database",
                "trust_level": 9,
                "description": "Redis database"
            },
            "https://github.com/postgres/postgres": {
                "category": "database",
                "trust_level": 9,
                "description": "PostgreSQL database"
            },
            
            # Blockchain (VERIFIED - but careful with crypto!)
            "https://github.com/bitcoin/bitcoin": {
                "category": "blockchain",
                "trust_level": 8,
                "description": "Bitcoin Core"
            },
            "https://github.com/ethereum/go-ethereum": {
                "category": "blockchain",
                "trust_level": 8,
                "description": "Official Ethereum Go implementation"
            }
        }
    
    def load_blacklisted_repos(self) -> set:
        """Repos known to contain malicious or low-quality code"""
        return {
            # Add any known bad repos here
            "https://github.com/*/malware*",
            "https://github.com/*/hack*", 
            "https://github.com/*/crack*",
            "https://github.com/*/exploit*",
        }
    
    def load_suspicious_patterns(self) -> List[str]:
        """Patterns that indicate potentially malicious code"""
        return [
            # Obfuscation patterns
            "eval(base64",
            "exec(decode",
            "\\x00\\x00\\x00",  # Null bytes
            
            # Suspicious network activity
            "requests.get('http://evil",
            "urllib.urlopen('http://malware",
            "socket.connect(('0.0.0.0'",
            
            # Crypto miners
            "stratum+tcp://",
            "monero",
            "cryptonight",
            
            # System compromise
            "os.system('rm -rf",
            "subprocess.call(['format",
            "__import__('os').system",
            
            # Data exfiltration
            "upload_to_pastebin",
            "send_to_c2_server",
            "steal_credentials",
            
            # Common malware signatures
            "ransomware",
            "keylogger",
            "backdoor",
            "rootkit",
        ]
    
    def verify_repository(self, repo_url: str) -> Dict[str, Any]:
        """Comprehensive repository verification"""
        print(f"🔍 Verifying repository: {repo_url}")
        
        verification = {
            "url": repo_url,
            "safe": False,
            "trust_score": 0,
            "warnings": [],
            "metadata": {}
        }
        
        # Check if whitelisted
        if repo_url in self.verified_repos:
            verification["safe"] = True
            verification["trust_score"] = self.verified_repos[repo_url]["trust_level"]
            verification["metadata"] = self.verified_repos[repo_url]
            print(f"  ✅ Verified repository (trust: {verification['trust_score']}/10)")
            return verification
        
        # Check if blacklisted
        for blacklist_pattern in self.blacklisted_repos:
            if self.matches_pattern(repo_url, blacklist_pattern):
                verification["warnings"].append(f"Matches blacklist pattern: {blacklist_pattern}")
                print(f"  ❌ BLACKLISTED repository!")
                return verification
        
        # Get repository metadata from GitHub API
        try:
            api_url = repo_url.replace("github.com", "api.github.com/repos")
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                repo_data = response.json()
                
                # Check stars
                stars = repo_data.get('stargazers_count', 0)
                if stars < self.min_stars:
                    verification["warnings"].append(f"Low stars: {stars} < {self.min_stars}")
                else:
                    verification["trust_score"] += 2
                
                # Check age
                created_at = datetime.strptime(repo_data['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                age_days = (datetime.now() - created_at).days
                if age_days < self.min_age_days:
                    verification["warnings"].append(f"Too new: {age_days} days old")
                else:
                    verification["trust_score"] += 2
                
                # Check if archived or disabled
                if repo_data.get('archived', False):
                    verification["warnings"].append("Repository is archived")
                
                # Check license
                if repo_data.get('license'):
                    verification["trust_score"] += 1
                else:
                    verification["warnings"].append("No license")
                
                # Check for security policy
                if repo_data.get('has_issues', False):
                    verification["trust_score"] += 1
                
                # Store metadata
                verification["metadata"] = {
                    "stars": stars,
                    "forks": repo_data.get('forks_count', 0),
                    "age_days": age_days,
                    "language": repo_data.get('language', 'unknown'),
                    "size_kb": repo_data.get('size', 0),
                    "open_issues": repo_data.get('open_issues_count', 0)
                }
                
                # Determine if safe
                if verification["trust_score"] >= 5 and len(verification["warnings"]) == 0:
                    verification["safe"] = True
                    print(f"  ✅ Repository verified (trust: {verification['trust_score']}/10)")
                else:
                    print(f"  ⚠️  Repository has warnings: {verification['warnings']}")
                    
        except Exception as e:
            verification["warnings"].append(f"API verification failed: {e}")
            print(f"  ❌ Could not verify via GitHub API: {e}")
        
        return verification
    
    def scan_for_malicious_code(self, file_path: Path) -> List[str]:
        """Scan file for suspicious patterns"""
        warnings = []
        
        try:
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                warnings.append(f"File too large: {file_size_mb:.2f}MB")
                return warnings
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check for suspicious patterns
            for pattern in self.suspicious_patterns:
                if pattern.lower() in content.lower():
                    warnings.append(f"Suspicious pattern found: {pattern}")
            
            # Check for binary content in text file
            if '\x00' in content and file_path.suffix in ['.py', '.js', '.rs', '.cpp']:
                warnings.append("Binary content in source file")
            
            # Check for extremely long lines (obfuscation)
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if len(line) > 500:
                    warnings.append(f"Suspicious long line at {i+1}: {len(line)} chars")
            
            # Check for encoded/obfuscated content
            if 'base64' in content and 'decode' in content:
                warnings.append("Possible base64 obfuscation")
            
            if 'eval(' in content or 'exec(' in content:
                warnings.append("Dynamic code execution detected")
            
        except Exception as e:
            warnings.append(f"Error scanning file: {e}")
        
        return warnings
    
    def quarantine_repository(self, repo_path: Path, reason: str):
        """Move suspicious repository to quarantine"""
        print(f"🔒 Quarantining repository: {reason}")
        
        quarantine_path = self.quarantine_dir / repo_path.name
        if repo_path.exists():
            repo_path.rename(quarantine_path)
        
        # Log quarantine
        with open(self.quarantine_dir / "quarantine_log.txt", 'a') as f:
            f.write(f"{datetime.now()}: {repo_path.name} - {reason}\n")
    
    def safe_assimilate_repo(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """Safely assimilate repository with verification"""
        
        # Step 1: Verify repository
        verification = self.verify_repository(repo_url)
        
        if not verification["safe"]:
            print(f"❌ Repository failed verification")
            self.rejected_repos.append({
                "url": repo_url,
                "reason": verification["warnings"],
                "timestamp": datetime.now().isoformat()
            })
            return None
        
        # Step 2: Clone to temporary location first
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        temp_path = self.quarantine_dir / f"temp_{repo_name}"
        
        try:
            print(f"📥 Cloning to temporary location for scanning...")
            if temp_path.exists():
                import shutil
                shutil.rmtree(temp_path)
            
            repo = git.Repo.clone_from(repo_url, temp_path, depth=1)  # Shallow clone first
            
            # Step 3: Scan all files for malicious code
            print(f"🔍 Scanning for malicious patterns...")
            total_warnings = []
            files_scanned = 0
            
            for file_path in temp_path.rglob('*'):
                if file_path.is_file() and self.is_code_file(file_path):
                    warnings = self.scan_for_malicious_code(file_path)
                    if warnings:
                        total_warnings.extend(warnings)
                    files_scanned += 1
                    
                    # Stop if too many warnings
                    if len(total_warnings) > 10:
                        break
            
            print(f"  Scanned {files_scanned} files")
            
            # Step 4: Decision based on scan
            if total_warnings:
                print(f"⚠️  Found {len(total_warnings)} warnings:")
                for warning in total_warnings[:5]:  # Show first 5
                    print(f"    - {warning}")
                
                # Ask for confirmation if warnings found
                if verification["trust_score"] >= 8:
                    print("  Repository is highly trusted despite warnings")
                    proceed = True
                else:
                    proceed = False
                    self.quarantine_repository(temp_path, f"Warnings: {total_warnings[:3]}")
            else:
                proceed = True
                print(f"✅ No malicious patterns found")
            
            # Step 5: Move to proper location if safe
            if proceed:
                final_path = self.repos_dir / repo_name
                if final_path.exists():
                    import shutil
                    shutil.rmtree(final_path)
                temp_path.rename(final_path)
                
                # Step 6: Process the repository
                stats = self.process_repository(final_path)
                stats["verification"] = verification
                
                # Step 7: Feed to neural network
                self.feed_to_neural_network(stats)
                
                return stats
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error during assimilation: {e}")
            if temp_path.exists():
                self.quarantine_repository(temp_path, str(e))
            return None
    
    def is_code_file(self, file_path: Path) -> bool:
        """Check if file is source code"""
        code_extensions = {
            '.py', '.rs', '.cpp', '.c', '.h', '.hpp', '.js', '.ts', 
            '.go', '.java', '.rb', '.php', '.cs', '.swift'
        }
        return file_path.suffix.lower() in code_extensions
    
    def process_repository(self, repo_path: Path) -> Dict[str, Any]:
        """Process verified repository"""
        stats = {
            'total_files': 0,
            'languages': {},
            'patterns': [],
            'safe_files': 0,
            'skipped_files': 0
        }
        
        for file_path in repo_path.rglob('*'):
            if file_path.is_file() and self.is_code_file(file_path):
                # Double-check each file
                warnings = self.scan_for_malicious_code(file_path)
                
                if not warnings:
                    # Process safe file
                    stats['safe_files'] += 1
                    stats['total_files'] += 1
                    
                    # Learn from it
                    self.learn_from_file(file_path)
                else:
                    stats['skipped_files'] += 1
                    print(f"    Skipped: {file_path.name} ({warnings[0]})")
        
        print(f"✅ Processed {stats['safe_files']} safe files")
        print(f"⚠️  Skipped {stats['skipped_files']} suspicious files")
        
        return stats
    
    def learn_from_file(self, file_path: Path):
        """Learn from verified safe file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Store in Redis for neural network
            key = f"safe_code:{file_path.stem}"
            self.redis_client.set(key, content[:10000])  # Limit size
            
        except Exception as e:
            print(f"    Error learning from {file_path}: {e}")
    
    def feed_to_neural_network(self, stats: Dict[str, Any]):
        """Feed verified data to neural network"""
        print("🧠 Feeding verified data to neural network...")
        
        # Only feed data with high trust score
        if stats.get("verification", {}).get("trust_score", 0) >= 5:
            self.redis_client.publish('neuronlang:verified_training', json.dumps(stats))
            print("  ✅ High-quality data fed to neural network")
        else:
            print("  ⚠️  Data not high-quality enough for training")
    
    def matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches pattern (with wildcards)"""
        import fnmatch
        return fnmatch.fnmatch(text.lower(), pattern.lower())
    
    def get_curated_repo_list(self) -> List[str]:
        """Get list of curated, verified repositories"""
        return [
            # Core languages (all verified)
            "https://github.com/python/cpython",
            "https://github.com/rust-lang/rust",
            "https://github.com/golang/go",
            
            # Web frameworks (established)
            "https://github.com/django/django",
            "https://github.com/expressjs/express",
            "https://github.com/rails/rails",
            
            # Databases (trusted)
            "https://github.com/redis/redis",
            "https://github.com/postgres/postgres",
            "https://github.com/mongodb/mongo",
            
            # ML frameworks (official repos only)
            "https://github.com/tensorflow/tensorflow",
            "https://github.com/pytorch/pytorch",
            "https://github.com/scikit-learn/scikit-learn",
            
            # Tools (verified maintainers)
            "https://github.com/git/git",
            "https://github.com/vim/vim",
            "https://github.com/tmux/tmux",
        ]
    
    def assimilate_curated_repos(self):
        """Assimilate only curated, verified repositories"""
        repos = self.get_curated_repo_list()
        
        print("🛡️ SECURE ASSIMILATION MODE")
        print(f"   Processing {len(repos)} verified repositories")
        print("")
        
        successful = 0
        failed = 0
        
        for repo_url in repos:
            print(f"\n{'='*60}")
            result = self.safe_assimilate_repo(repo_url)
            
            if result:
                successful += 1
            else:
                failed += 1
            
            # Rate limiting
            time.sleep(2)
        
        print(f"\n{'='*60}")
        print(f"📊 ASSIMILATION COMPLETE")
        print(f"   Successful: {successful}")
        print(f"   Rejected: {failed}")
        print(f"   Quarantined: {len(list(self.quarantine_dir.glob('*')))}")
    
    def view_rejected_repos(self):
        """View list of rejected repositories"""
        print("\n❌ REJECTED REPOSITORIES:")
        for rejected in self.rejected_repos:
            print(f"\n  {rejected['url']}")
            print(f"  Timestamp: {rejected['timestamp']}")
            print(f"  Reasons:")
            for reason in rejected['reason']:
                print(f"    - {reason}")
    
    def view_quarantine(self):
        """View quarantined repositories"""
        print("\n🔒 QUARANTINED REPOSITORIES:")
        
        log_file = self.quarantine_dir / "quarantine_log.txt"
        if log_file.exists():
            with open(log_file, 'r') as f:
                print(f.read())
        else:
            print("  No repositories in quarantine")

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║       🛡️  SECURE GITHUB ASSIMILATOR FOR NEURONLANG 🛡️        ║
║                                                               ║
║  Only assimilates VERIFIED, SAFE, HIGH-QUALITY code          ║
║  Protects the neural network from malicious data!            ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    assimilator = SecureGitHubAssimilator()
    
    while True:
        print("\n🔒 SECURE OPTIONS:")
        print("1. Assimilate curated repos only (SAFE)")
        print("2. Verify specific repo (CHECK FIRST)")
        print("3. View rejected repos")
        print("4. View quarantine")
        print("5. Set security thresholds")
        print("6. Exit")
        
        choice = input("\nChoice: ")
        
        if choice == "1":
            print("\n⚠️  This will only process verified, curated repositories")
            confirm = input("Proceed? (yes/no): ")
            if confirm.lower() == "yes":
                assimilator.assimilate_curated_repos()
        
        elif choice == "2":
            repo_url = input("Enter GitHub repo URL to verify: ")
            verification = assimilator.verify_repository(repo_url)
            
            print(f"\n📊 Verification Results:")
            print(f"   Safe: {verification['safe']}")
            print(f"   Trust Score: {verification['trust_score']}/10")
            print(f"   Warnings: {verification['warnings']}")
            print(f"   Metadata: {json.dumps(verification['metadata'], indent=2)}")
            
            if verification['safe']:
                proceed = input("\nAssimilate this repository? (yes/no): ")
                if proceed.lower() == "yes":
                    assimilator.safe_assimilate_repo(repo_url)
        
        elif choice == "3":
            assimilator.view_rejected_repos()
        
        elif choice == "4":
            assimilator.view_quarantine()
        
        elif choice == "5":
            print("\nCurrent thresholds:")
            print(f"   Min stars: {assimilator.min_stars}")
            print(f"   Min age: {assimilator.min_age_days} days")
            print(f"   Max file size: {assimilator.max_file_size_mb} MB")
            
            new_stars = input(f"New min stars [{assimilator.min_stars}]: ")
            if new_stars:
                assimilator.min_stars = int(new_stars)
            
            new_age = input(f"New min age days [{assimilator.min_age_days}]: ")
            if new_age:
                assimilator.min_age_days = int(new_age)
        
        elif choice == "6":
            print("\n🛡️ Secure assimilation session ended")
            break

if __name__ == "__main__":
    main()