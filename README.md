# Optimal Batting Order Selection and Bowling Strategy Optimization in T20 Cricket

## Project Overview

This project develops a machine learning-based system to simulate cricket matches and optimize team strategies, focusing on the Indian Premier League (IPL) T20 format. By leveraging historical match data, the system processes JSON files to create datasets, trains predictive models, and simulates match outcomes to recommend optimal batting and bowling orders. The project aims to enhance strategic decision-making in cricket by predicting the best players to deploy based on match context, such as phase of play and remaining balls.

The system consists of four main components:
1. **Data Processing**: Extracts and organizes IPL match data into structured datasets.
2. **Batting Order Prediction**: Uses a Random Forest Regressor to predict the next batter based on expected runs.
3. **Bowling Order Prediction**: Employs a Random Forest Classifier to select the next bowler based on wickets and runs conceded.
4. **Match Simulation**: Simulates full T20 matches, providing detailed commentary and comparing user-defined and model-predicted strategies.

## Repository Contents

- **Team_26.ipynb**: The main Jupyter Notebook containing all code for data processing, model training, and match simulation.
- **datasets/**: Directory containing generated CSV files:
  - `batter_bowler_runs.csv`: Runs scored by batters against bowlers per delivery, including phase.
  - `bowler_batter_outcome.csv`: Outcomes (runs or wickets) of deliveries, categorized by phase.
  - `batter_balls_faced.csv`: Average balls faced per innings by each batter.
  - `bowler_balls_per_match.csv`: Average balls bowled per match by each bowler.
- **ipl_json/**: Directory containing raw IPL match data in JSON format (not included in repository; must be provided).
- **batting_order_model.pkl**, **bowler_order_model.pkl**, **le_*.pkl**: Trained models and label encoders saved using joblib.
- **README.md**: This file, providing an overview and instructions for the project.

## Prerequisites

To run the project, ensure the following dependencies are installed:

- Python 3.11.7
- Libraries: `pandas`, `numpy`, `sklearn`, `joblib`, `json`, `os`, `collections`, `random`, `math`
- Jupyter Notebook for executing the `.ipynb` file
- IPL JSON match data (place in `ipl_json/` directory)

Install dependencies using:
```bash
pip install pandas numpy scikit-learn joblib
```

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Prepare IPL JSON Data**:
   - Obtain IPL match JSON files (e.g., from a dataset like Cricsheet).
   - Place them in the `ipl_json/` directory within the project folder.

3. **Install Dependencies**:
   Run the pip command above to install required Python libraries.

4. **Run the Jupyter Notebook**:
   - Start Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open `Team_26.ipynb` and execute the cells sequentially.

## Project Workflow

### 1. Data Processing (Cell 1)
- **Input**: JSON files in `ipl_json/` containing IPL match details (deliveries, players, outcomes).
- **Process**:
  - Reads JSON files and extracts relevant data (batters, bowlers, runs, wickets, overs).
  - Defines match phases: Powerplay (overs 1-6), Middleover (overs 7-15), Deathover (overs 16-20).
  - Creates four datasets:
    - **batter_bowler_runs.csv**: Tracks runs scored per delivery, with batter/bowler IDs and names, and phase.
    - **bowler_batter_outcome.csv**: Records delivery outcomes (runs or 'W' for wicket), with phase.
    - **batter_balls_faced.csv**: Calculates average balls faced per innings for each batter.
    - **bowler_balls_per_match.csv**: Computes average balls bowled per match for each bowler.
- **Output**: CSV files saved in `datasets/` directory.

### 2. Batting Order Prediction (Cell 2)
- **Objective**: Predict the next batter to maximize runs.
- **Input**: `batter_bowler_runs.csv`, `batter_balls_faced.csv`.
- **Process**:
  - Merges datasets to include only batters with balls-faced data.
  - Encodes categorical features (batter, bowler, phase) using LabelEncoder.
  - Trains a Random Forest Regressor to predict runs scored per delivery.
  - Saves the model and encoders as `.pkl` files.
  - Implements `predict_next_batter` function:
    - Takes remaining batters, current bowler, and remaining balls as input.
    - Predicts runs for each batter over their average balls faced (adjusted for remaining balls).
    - Returns a sorted DataFrame of predicted runs and balls faced.
- **Output**:
  - Mean Squared Error (MSE) of the model (e.g., 3.0053).
  - Example prediction for batters like SA Yadav, recommending the highest-scoring batter.

### 3. Bowling Order Prediction (Cell 3)
- **Objective**: Select the next bowler to maximize wickets and minimize runs.
- **Input**: `bowler_batter_outcome.csv`, `bowler_balls_per_match.csv`.
- **Process**:
  - Merges datasets to include only bowlers with balls-per-match data.
  - Encodes features (bowler, batter, phase, outcome) using LabelEncoder.
  - Trains a Random Forest Classifier to predict delivery outcomes (0, 1, 2, 3, 4, 6, or 'W').
  - Saves the model and encoders as `.pkl` files.
  - Implements `predict_next_bowler` function:
    - Takes remaining bowlers, current batsman, balls bowled, and remaining balls as input.
    - Simulates one over (6 balls) per bowler, predicting wickets and runs.
    - Returns a sorted DataFrame prioritizing wickets, then runs conceded.
- **Output**:
  - Model accuracy (e.g., 0.3805).
  - Example prediction for bowlers like JJ Bumrah, recommending the most effective bowler.

### 4. Match Simulation (Cell 4)
- **Objective**: Simulate a full T20 match with ball-by-ball commentary.
- **Input**: Team squads, batting/bowling orders, datasets, and trained models.
- **Process**:
  - Defines functions to:
    - Determine match phase based on remaining balls.
    - Predict ball outcomes using bowling predictions and phase-based probabilities.
    - Select next batter/bowler using trained models (for model-predicted simulation).
    - Simulate innings, tracking runs, wickets, and player stats.
  - Runs two simulations:
    - **User-Defined**: Uses provided batting and bowling orders.
    - **Model-Predicted**: Dynamically selects batters and bowlers using predictive models.
  - Generates detailed commentary for each ball, over, and innings.
  - Compares innings scores to determine the match result (win, loss, or tie).
- **Output**:
  - Ball-by-ball commentary for both innings.
  - Batter and bowler statistics (runs, balls faced, wickets, etc.).
  - Match result (e.g., "The match is a tie!").

### 5. Empty Cell (Cell 5)
- Reserved for future extensions or debugging.

## Usage Example

The notebook includes example usage in each cell. For the match simulation (Cell 4), two teams are defined:
- **Team 1**: Q de Kock, KL Rahul, DJ Hooda, KH Pandya, A Badoni, MP Stoinis, JO Holder, PVD Chameera, Avesh Khan, Mohsin Khan, Ravi Bishnoi.
- **Team 2**: B Indrajith, AJ Finch, SS Iyer, N Rana, RK Singh, AD Russell, SP Narine, AS Roy, Shivam Mavi, TG Southee, Harshit Rana.

The simulation runs with Team 1 batting first, producing detailed commentary and statistics. The example output shows a tied match (154/5 vs. 154/2).

To simulate a different match:
1. Update the `team1_squad`, `team2_squad`, `team1_batting_order`, `team2_batting_order`, `team1_bowling_order`, and `team2_bowling_order` in Cell 4.
2. Set `batting_team` to 'Team 1' or 'Team 2'.
3. Run the cell to generate new simulation results.

## Demo and Report

- **YouTube Demo**: A video presentation of the project is available at [https://youtu.be/iEI-OaNv9jc](https://youtu.be/iEI-OaNv9jc). It showcases the notebook execution, dataset generation, model predictions, and match simulation.
- **Project Report**: A detailed report is accessible via [Google Drive](https://drive.google.com/drive/folders/1ivfQFJGLBm-YRcgtlSeBPGMu1oYMXKGm?usp=drive_link). It covers the methodology, implementation, results, and analysis.

## Limitations and Future Work

- **Data Dependency**: The system relies on IPL JSON data, which must be sourced separately.
- **Model Accuracy**: The bowling model's accuracy (0.3805) suggests room for improvement, possibly by incorporating more features (e.g., pitch conditions, player form).
- **Simulation Realism**: Outcome probabilities are simplified; future versions could integrate advanced statistical models or real-time data.
- **Scalability**: The system is tailored for T20; adapting to other formats (e.g., ODIs) requires adjustments to phase definitions and ball counts.
- **Potential Enhancements**:
  - Add real-time player performance updates via web scraping.
  - Incorporate team strategies (e.g., aggressive vs. defensive play).
  - Develop a user interface for interactive match simulations.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of changes.

Please ensure code follows PEP 8 style guidelines and includes comments for clarity.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, contact the project team via GitHub Issues or the email provided in the project report.

---

*Developed by Team 26 as part of a machine learning project for cricket strategy optimization.*
