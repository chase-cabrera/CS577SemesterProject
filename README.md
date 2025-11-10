# CS577 Semester Project: FEC Campaign Finance Data Analysis

Analysis of Federal Election Commission (FEC) individual contribution data for CS577 Data Science Course.

## Setup

1. **Create virtual environment and install dependencies:**
   
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   

2. **Configure database:**
     
    Use .env.example as a template
     

3. **Initialize database:**
   
   python src/setup_database.py


## Data Files

Place FEC data files in `data/`:
- `cn.txt` - Candidate master file
- `cm.txt` - Committee master file  
- `ccl.txt` - Candidate-committee linkages
- `itcont*.txt` - Individual contribution files
- more files to come
   

## Usage

**Import campaign data** (committees, candidates):

python src/import_campaign_data.py


**Process contributions**:

python src/parallel_processor.py --workers 4


**Calculate totals:**

python src/calculate_totals.py



