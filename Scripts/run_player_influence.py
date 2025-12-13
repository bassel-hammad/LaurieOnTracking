#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Player Influence Analysis Runner

Convenient entry point to run the player influence analysis from the Scripts folder.

Usage:
    python run_player_influence.py
"""

import sys
import os

# Add the Scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Change to the project root directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from player_influence.main import main

if __name__ == "__main__":
    main()
