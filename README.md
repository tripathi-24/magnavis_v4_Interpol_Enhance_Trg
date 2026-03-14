# Python Application, Magnavis

This repository contains a Python application developed by lab members of Prof. Saikat Ghosh, Department of Physics, IIT Kanpur. The application is designed to provide insights to data analytics and visualisation in context of magnetic fields, anamoly fields and other advanced magnetic simulations.

## Prerequisites

Before running the application, ensure you have the following installed:
- **Python 3.12+** (recommended)
- **pip** (Python package manager)
- **Git** (optional, for cloning the repository)

## Setup Instructions

Follow these steps to set up and run the application on your local machine:

### 1. Clone the Repository
Clone this repository to your local machine using:
```bash
git clone https://gitlab.com/joy.b/magnavis.git
cd magnavis
```
### 2. Install python modules
set up and activate virtual environment (optional)

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
powershell(Windows)
python -m venv venv
.\venv\Scripts\Activate.ps1

 and install the libraries:
~~~~
pip install -r requirements_feb_2025.txt
~~~~

### 3. run the application
using relative file name
```
python src/application.py
```
or absolute filename
~~~~
python c:/Users/DELL/Desktop/Projects/quantum/magnavis/src/application.py
~~~~

For the multi-sensor DB/CSV workflow (pre-trained GRU, anomaly direction, triangulation):
```
python src/application_temp.py
```

### 4. Logs for offline analysis
When you run the application, all messages that appear in the on-screen log are also written to a session log file:
- **Application log:** `src/sessions/<session_id>/app.log`  
  One line per message: timestamp, level (Info/Warning/Error), and message text. The session id is shown in the log at startup (e.g. `Session id "abc-123-..."`).
- **Predictor logs (per sensor):** `src/sessions/<session_id>/<sensor_id>/predict_stdout.log` and `predict_stderr.log`  
  Stdout/stderr from the GRU predictor subprocess.

Use these files for offline analysis and debugging.