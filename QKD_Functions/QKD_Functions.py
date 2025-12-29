import sys
import os

# Add the project root to the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import everything from the new src.qkd package to maintain backward compatibility
from src.qkd.physics import *
from src.qkd.utils import *
from src.qkd.model import *