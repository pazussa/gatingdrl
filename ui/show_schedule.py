import argparse
import os
import glob
import sys

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.definitions import OUT_DIR





def display_schedule_log(log_file=None, by_link=True):
    """
    Display the contents of a scheduling log file.
    
    Args:
        log_file: Path to the log file. If None, displays the most recent log file.
        by_link: If True, show logs organized by link, otherwise by flow.
    """
    if log_file is None:
        # Find the most recent schedule_res log file
        pattern = 'schedule_res_by_link_*.log' if by_link else 'schedule_res_*.log'
        log_files = glob.glob(os.path.join(OUT_DIR, pattern))
        if not log_files:
            # Try the other format if no logs found
            alt_pattern = 'schedule_res_*.log' if by_link else 'schedule_res_by_link_*.log'
            log_files = glob.glob(os.path.join(OUT_DIR, alt_pattern))
            if not log_files:
                print("No scheduling log files found in", OUT_DIR)
                return
        log_file = max(log_files, key=os.path.getmtime)
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        print("\n" + "="*80)
        print(f"SCHEDULING DETAILS FROM: {os.path.basename(log_file)}")
        print("="*80)
        print(content)
        print("="*80 + "\n")
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file}")
    except Exception as e:
        print(f"Error reading log file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display scheduling log file contents")
    parser.add_argument('--file', type=str, help='Path to specific log file (optional)')
    parser.add_argument('--by-flow', action='store_true', help='Show logs organized by flow instead of by link')
    
    args = parser.parse_args()
    display_schedule_log(args.file, not args.by_flow)
