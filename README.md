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

   Read only or (select) Access to the DB
   BAbramowitz
   OELh9Jy903Rb

## Data Files

**Download FEC data files from:** [Google Drive](https://drive.google.com/drive/u/2/folders/10v6GdV33jA6hiG1BwzqyzFupMZBinKe0)

Place downloaded files in `data/` directory:
- `cn.txt` - Candidate master file
- `cm.txt` - Committee master file  
- `ccl.txt` - Candidate-committee linkages
- `itcont*.txt` - Individual contribution files
- more files to come

   
**1. Initialize database**:

python src/setup_database.py

**2. Import campaign data** (committees, candidates, linkages):

python src/import_campaign_data.py

**3. Import raw contributions** (parses names on insert, marks status=. Needs to be processed, normalized, and have totals aggregated):

python src/import_raw_contributions.py

**4. Process raw contributions** (parallel, updates contributors + contributions tables, marks status=1):

python src/process_raw_parallel.py --workers 8 --batch-size 1000

This populates:
- contributors (aggregated donor features for ML/EDA)
- contributions (normalized transaction records for detailed analysis)


**5. Exploratory Data Analysis (EDA)** - generates visualizations from contributors table:

python src/eda_analysis.py

