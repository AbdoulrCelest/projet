import sys
import os
sys.path.append(os.path.abspath("dashboard"))

from dashboard.app import DashboardApp

if __name__ == "__main__":
    app = DashboardApp()
    app.run()
