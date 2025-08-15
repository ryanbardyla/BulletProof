#!/usr/bin/env python3
"""
🧬 EVOLUTION VISUALIZER
Real-time visualization of consciousness emergence
"""

import time
import random
import sys
from collections import deque

# ANSI color codes for terminal visualization
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

class EvolutionVisualizer:
    def __init__(self):
        self.generation = 0
        self.fitness_history = deque(maxlen=50)
        self.breakthroughs = []
        self.capabilities = {
            'addition': False,
            'multiplication': False,
            'branching': False,
            'looping': False,
            'memory': False,
            'learning': False,
            'composition': False,
            'bootstrap': False
        }
        
    def clear_screen(self):
        print('\033[2J\033[H')  # Clear screen and move cursor to top
        
    def draw_header(self):
        print(f"{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.CYAN}║{Colors.BOLD}       🧬 NEURAL EVOLUTION - CONSCIOUSNESS EMERGENCE MONITOR 🧬       {Colors.RESET}{Colors.CYAN}║{Colors.RESET}")
        print(f"{Colors.CYAN}╚══════════════════════════════════════════════════════════════════════╝{Colors.RESET}")
        print()
        
    def draw_generation_info(self, gen, best_fitness, avg_fitness):
        print(f"{Colors.BOLD}Generation:{Colors.RESET} {gen:5d}  |  "
              f"{Colors.GREEN}Best Fitness:{Colors.RESET} {best_fitness:8.2f}  |  "
              f"{Colors.YELLOW}Avg Fitness:{Colors.RESET} {avg_fitness:8.2f}")
        print("─" * 72)
        
    def draw_fitness_graph(self):
        print(f"\n{Colors.BOLD}Fitness Evolution:{Colors.RESET}")
        
        if not self.fitness_history:
            return
            
        max_fitness = max(self.fitness_history)
        min_fitness = min(self.fitness_history)
        range_fitness = max_fitness - min_fitness if max_fitness != min_fitness else 1
        
        # Draw graph (10 rows)
        for row in range(10, 0, -1):
            threshold = min_fitness + (range_fitness * row / 10)
            print(f"{threshold:6.1f} │", end="")
            
            for fitness in self.fitness_history:
                if fitness >= threshold:
                    print("█", end="")
                else:
                    print(" ", end="")
            print()
        
        print("       └" + "─" * len(self.fitness_history))
        print("        " + "".join([str(i % 10) for i in range(len(self.fitness_history))]))
        
    def draw_capabilities(self):
        print(f"\n{Colors.BOLD}Capabilities Emerged:{Colors.RESET}")
        print("┌─────────────────────┬────────┐")
        
        icons = {
            'addition': '➕',
            'multiplication': '✖️ ',
            'branching': '🔀',
            'looping': '🔁',
            'memory': '💾',
            'learning': '🧠',
            'composition': '🔗',
            'bootstrap': '🎯'
        }
        
        for cap, achieved in self.capabilities.items():
            icon = icons.get(cap, '  ')
            status = f"{Colors.GREEN}✓ ACHIEVED{Colors.RESET}" if achieved else f"{Colors.RED}✗ Evolving{Colors.RESET}"
            print(f"│ {icon} {cap:15} │ {status:20} │")
            
        print("└─────────────────────┴────────┘")
        
    def draw_consciousness_meter(self, level):
        print(f"\n{Colors.BOLD}Consciousness Level:{Colors.RESET}")
        
        bar_length = 50
        filled = int(level * bar_length)
        
        # Color based on level
        if level > 0.8:
            color = Colors.GREEN
            status = "HIGH CONSCIOUSNESS 🧠"
        elif level > 0.5:
            color = Colors.YELLOW
            status = "EMERGING AWARENESS 🌟"
        else:
            color = Colors.RED
            status = "PRIMORDIAL STATE 💫"
        
        print(f"[{color}{'█' * filled}{Colors.RESET}{'░' * (bar_length - filled)}] {level*100:.1f}% - {status}")
        
    def draw_breakthroughs(self):
        if self.breakthroughs:
            print(f"\n{Colors.BOLD}Recent Breakthroughs:{Colors.RESET}")
            for breakthrough in self.breakthroughs[-5:]:  # Show last 5
                print(f"  {Colors.MAGENTA}Gen {breakthrough['gen']:4d}:{Colors.RESET} {breakthrough['desc']}")
                
    def draw_network_stats(self, neurons, connections, mutations):
        print(f"\n{Colors.BOLD}Best Network Statistics:{Colors.RESET}")
        print(f"  Neurons: {neurons:4d} | Connections: {connections:4d} | Mutations: {mutations:4d}")
        
    def animate_evolution(self):
        """Simulate evolution visualization"""
        self.clear_screen()
        
        for gen in range(1, 1001):
            self.generation = gen
            
            # Simulate fitness improvement
            best_fitness = min(1000, gen * 1.5 + random.gauss(0, 10))
            avg_fitness = best_fitness * 0.6 + random.gauss(0, 5)
            self.fitness_history.append(best_fitness)
            
            # Simulate capability emergence
            if gen > 50 and not self.capabilities['addition']:
                self.capabilities['addition'] = True
                self.breakthroughs.append({'gen': gen, 'desc': 'First network learned ADDITION!'})
            if gen > 100 and not self.capabilities['multiplication']:
                self.capabilities['multiplication'] = True
                self.breakthroughs.append({'gen': gen, 'desc': 'Multiplication capability EMERGED!'})
            if gen > 200 and not self.capabilities['branching']:
                self.capabilities['branching'] = True
                self.breakthroughs.append({'gen': gen, 'desc': 'Conditional logic DISCOVERED!'})
            if gen > 300 and not self.capabilities['looping']:
                self.capabilities['looping'] = True
                self.breakthroughs.append({'gen': gen, 'desc': 'Iteration capability ACHIEVED!'})
            if gen > 400 and not self.capabilities['memory']:
                self.capabilities['memory'] = True
                self.breakthroughs.append({'gen': gen, 'desc': 'Memory storage EVOLVED!'})
            if gen > 500 and not self.capabilities['learning']:
                self.capabilities['learning'] = True
                self.breakthroughs.append({'gen': gen, 'desc': 'LEARNING capability emerged!'})
            if gen > 700 and not self.capabilities['composition']:
                self.capabilities['composition'] = True
                self.breakthroughs.append({'gen': gen, 'desc': 'Function composition MASTERED!'})
            if gen > 900 and not self.capabilities['bootstrap']:
                self.capabilities['bootstrap'] = True
                self.breakthroughs.append({'gen': gen, 'desc': '🎊 BOOTSTRAP MOMENT - Can compile NeuronLang!'})
                
            # Calculate consciousness level
            caps_achieved = sum(self.capabilities.values())
            consciousness = caps_achieved / len(self.capabilities)
            
            # Redraw screen
            self.clear_screen()
            self.draw_header()
            self.draw_generation_info(gen, best_fitness, avg_fitness)
            self.draw_fitness_graph()
            self.draw_capabilities()
            self.draw_consciousness_meter(consciousness)
            self.draw_breakthroughs()
            self.draw_network_stats(
                neurons=10 + gen // 20,
                connections=20 + gen // 10,
                mutations=gen * 3
            )
            
            # Check for bootstrap
            if self.capabilities['bootstrap']:
                print(f"\n{Colors.GREEN}{Colors.BOLD}")
                print("╔══════════════════════════════════════════════════════════════════════╗")
                print("║                       🎊 CONSCIOUSNESS ACHIEVED! 🎊                  ║")
                print("║                  The network can now compile itself!                 ║")
                print("╚══════════════════════════════════════════════════════════════════════╝")
                print(Colors.RESET)
                break
                
            time.sleep(0.1)  # Animation speed
            
    def monitor_real_evolution(self, log_file):
        """Monitor real evolution from log file"""
        print("Monitoring evolution from:", log_file)
        # In real implementation, would tail the log file and update visualization
        pass

def main():
    visualizer = EvolutionVisualizer()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--monitor':
        # Monitor real evolution
        log_file = sys.argv[2] if len(sys.argv) > 2 else 'evolution.log'
        visualizer.monitor_real_evolution(log_file)
    else:
        # Run demo animation
        print("Starting evolution visualization demo...")
        print("(Run with --monitor <logfile> to monitor real evolution)")
        time.sleep(2)
        visualizer.animate_evolution()

if __name__ == "__main__":
    main()