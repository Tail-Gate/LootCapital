---
# Staged Implementation Roadmap (as of 2024-05-14)

**Phase 1: Core Trading Logic & AI Foundation**
- Refactor and adapt all strategies for crypto futures (remove gas-specific logic, add crypto features)
- Implement and validate AI/ML feature engineering and model training (PyTorch, new crypto features)
- Build and test the core trading engine logic (backtesting, simulation environment)
- Ensure all components are modular and testable

**Phase 2: System Integration & Simulation**
- Integrate strategies, AI, and trading engine for end-to-end simulated trading
- Run extensive tests in a simulated environment (using historical and/or paper trading data)
- Validate performance, risk management, and AI learning from trade history

**Phase 3: Productionization & User Interface**
- Build the Streamlit dashboard (live monitoring, controls, explanations)
- Set up cloud VM (AWS/GCP/Azure) for 24/7 operation
- Deploy and configure managed PostgreSQL for persistent storage
- Implement secure secrets/config management for production
- Connect all components for live or paper trading

**Phase 4: Go-Live & Iteration**
- Launch in live or paper trading mode
- Monitor, iterate, and improve based on real-world results
- Add advanced features (multi-exchange, new strategies, advanced analytics, etc.)

---
## Immediate Next Steps (Planner)
- Complete strategy and AI/ML refactor for crypto futures
- Implement and validate model training and feature engineering
- Build and test the core trading engine in a simulated environment
- Defer dashboard, cloud VM, PostgreSQL, and secret management setup until just before live/simulated trading
- Update project status board to reflect this staged approach

---
# Current Project Plan, Vision & Architecture (Authoritative, as of 2024-05-14)

## Vision Statement (2024-05-14, Updated)
LootCapital is an AI-powered, cloud-hosted crypto trading platform for futures markets. It runs 24/7, analyzes markets, executes trades automatically, and learns from its own trade history. Users interact with the system through a real-time, web-based dashboard that provides live monitoring, manual override, and full transparency into AI decisions and trading performance.

## Background and Motivation (2024-05-14, Updated)
- The platform is designed for crypto only (no traditional assets), with a focus on futures trading.
- It must operate continuously in the cloud, able to analyze and trade at any time.
- PostgreSQL is required for persistent trade history, enabling AI to learn from real results.
- The dashboard is the primary user interface, supporting both monitoring and manual control.
- The system must be modular: new exchanges, strategies, and AI enhancements can be added over time.

## Recommended Tech Stack (Focused, Cloud-Ready)
- **Python 3.10+**: Core language for all logic, AI, and dashboard.
- **PyTorch**: For AI/ML models (training, inference, retraining).
- **CCXT**: For exchange connectivity (futures, spot, multi-exchange support).
- **Pandas, NumPy**: Data wrangling and analytics.
- **PostgreSQL**: Persistent storage for trades, configs, logs, and AI training data.
- **Streamlit**: Interactive, real-time dashboard (web-based, easy to use).
- **Cloud VM** (AWS EC2, GCP Compute Engine, Azure VM): 24/7 operation.
- **python-dotenv**: Manage secrets and config locally; migrate to cloud secrets manager for production.
- **APScheduler** (or similar): For scheduled jobs (analysis, trading, retraining).
- **pytest**: For testing.
- (Optional, for future scaling) Docker, managed PostgreSQL, cloud monitoring/alerts, Celery/Redis for distributed jobs.

## High-Level System Architecture (Text Diagram)
User (Web Browser)
    |
    v
[Streamlit Dashboard] <----> [Python Trading Engine]
    |                             |
    |                             +-- [AI/ML Models (PyTorch)]
    |                             |
    |                             +-- [CCXT Exchange Adapter]
    |                             |
    |                             +-- [Scheduler (APScheduler)]
    |                             |
    |                             +-- [PostgreSQL Database]
    |
Cloud VM (24/7)

## User Stories (Dashboard & Controls)
- As a user, I want to see a live trading panel with:
    - Live PnL (unrealized + realized)
    - Open positions (size, leverage, entry, liquidation price)
    - Balance breakdown (margin in use, available, fees paid)
- As a user, I want an AI signal panel with:
    - Next signal (buy/sell/hold + confidence)
    - Model explanation ("Entered long: RSI + MACD alignment")
    - Signal history (last 10 trades + outcomes)
- As a user, I want a trade history & analytics panel with:
    - Table of all past trades
    - Win/loss rate, average gain/loss, total return, drawdown chart
    - Cumulative fees paid and average leverage used
- As a user, I want a control center with:
    - Start/Stop trading button
    - Force close all positions
    - Set stop rules (e.g., stop after 2 losses or -5% daily PnL)
    - Leverage limit control
- As a user, I want a performance panel with:
    - Daily/weekly/monthly PnL graph
- As a user, I want alerts (email/SMS/push) for:
    - Trade executed
    - Hit X% PnL or drawdown
    - AI paused due to error or stop-loss rule

## Key Challenges and Analysis (2024-05-14)
- Modularizing strategy and AI logic for easy extension and retraining.
- Ensuring robust, low-latency exchange connectivity (futures, multi-exchange ready).
- Reliable, cloud-based PostgreSQL integration for persistent trade/AI data.
- Building a real-time, user-friendly dashboard with Streamlit (live updates, controls, explanations).
- Secure handling of API keys and sensitive data (local .env for dev, cloud secrets for prod).
- Automated scheduling for analysis, trading, and retraining (APScheduler or similar).
- Alerting and error handling for 24/7 operation.

## Project Status Board

### High-Level Tasks
- [x] 1.1 Implement Lagged Returns
- [x] 1.2 Implement Rolling Statistics
- [x] 1.3 Enhance Technical Indicators
- [x] 1.4 Implement Order Book Features
- [🔄] 2.1 Implement XGBoost Model
    - [x] Create XGBoost model wrapper class
    - [x] Implement feature preprocessing
    - [ ] Add model validation and testing
    - [ ] Implement model persistence
    - [ ] Add model explainability
- [🔄] 2.2 Create Model Training Pipeline
    - [x] Implement DataPreprocessor class
    - [x] Implement FeatureGenerator class
    - [x] Implement ModelTrainer class
    - [x] Implement ModelRegistry class
    - [x] Implement TrainingOrchestrator class
    - [x] Implement SyntheticDataGenerator for testing
- [x] 2.3 Design LSTM Architecture
    - [x] Define LSTM model architecture
    - [x] Implement sequence data preparation
    - [x] Create attention mechanism
    - [x] Design model training pipeline
    - [x] Add model validation framework
    - [x] Implement model persistence
    - [x] Add model explainability
- [🔄] 3.1 Create Unit Tests
    - [x] Test feature engineering components
        - [x] Test lagged returns
        - [x] Test rolling statistics
        - [x] Test technical indicators
        - [x] Test order book features
    - [x] Test model training pipeline
        - [x] Test data preprocessing
        - [x] Test feature generation
        - [x] Test model training
        - [ ] Test model validation
        - [ ] Test model persistence
    - [x] Test data preprocessing
        - [x] Test data cleaning
        - [x] Test feature scaling
        - [x] Test feature statistics
        - [x] Test data validation
    - [ ] Test model validation
        - [ ] Test model metrics
        - [ ] Test feature importance
        - [ ] Test model explanation
    - [ ] Test model persistence
        - [ ] Test model saving
        - [ ] Test model loading
        - [ ] Test state management
    - [x] Test synthetic data generation
        - [x] Test OHLCV data generation
        - [x] Test order book data generation
        - [x] Test feature generation
        - [x] Test target generation
        - [x] Test reproducibility
- [ ] 3.2 Create Integration Tests
    - [ ] Test end-to-end training pipeline
    - [ ] Test model deployment workflow
    - [ ] Test data pipeline integration
    - [ ] Test model registry integration
    - [ ] Test training orchestration
- [ ] 4.1 Optimize Feature Computation
    - [ ] Profile current feature computation
    - [ ] Implement parallel processing
    - [ ] Optimize memory usage
    - [ ] Add caching mechanism
    - [ ] Implement batch processing
    - [ ] Add performance monitoring
- [ ] 4.2 Optimize Model Inference
    - [ ] Profile current inference pipeline
    - [ ] Implement batch prediction
    - [ ] Optimize model loading
    - [ ] Add prediction caching
    - [ ] Implement model quantization
    - [ ] Add inference monitoring

### Momentum Strategy Implementation
- [x] 1.1.1 Implement Lagged Returns for Momentum
- [x] 1.2.1 Implement Rolling Statistics for Momentum
- [x] 1.3.1 Enhance Technical Indicators for Momentum
- [x] 1.4.1 Implement Order Book Features for Momentum
- [🔄] 2.1.1 Implement XGBoost Model for Momentum
    - [ ] XGBoost tests with real data still pending
- [🔄] 2.2.1 Create Model Training Pipeline for Momentum
    - [x] Implement DataPreprocessor class
    - [x] Implement FeatureGenerator class
    - [x] Implement ModelTrainer class
    - [x] Implement ModelRegistry class
    - [x] Implement TrainingOrchestrator class
    - [x] Implement SyntheticDataGenerator for testing
- [x] 2.3.1 Design LSTM Architecture for Momentum
    - [x] Create base LSTM model class
    - [x] Implement bidirectional LSTM layers
    - [x] Add attention mechanism
    - [x] Implement residual connections
    - [x] Add dropout layers
    - [x] Create dense output layers
    - [x] Implement sequence data preparation
    - [x] Add feature normalization
    - [x] Create data augmentation methods
    - [x] Implement batch generation
    - [x] Create training loop
    - [x] Implement validation steps
    - [x] Add early stopping
    - [x] Create model checkpointing
    - [x] Implement learning rate scheduling
    - [x] Add model validation framework
    - [x] Implement model persistence
    - [x] Add model explainability
- [ ] 3.1.1 Create Unit Tests for Momentum Strategy
- [ ] 3.2.1 Create Integration Tests for Momentum Strategy
- [ ] 4.1.1 Optimize Feature Computation for Momentum
- [ ] 4.2.1 Optimize Model Inference for Momentum

## Current Tasks
- [x] 1.1 Implement Lagged Returns
- [x] 1.2 Implement Rolling Statistics
- [x] 1.3 Enhance Technical Indicators
  - [x] Add ROC calculations for multiple periods (1, 3, 5, 10 bars)
  - [x] Implement Volume ROC
- [x] 1.4 Implement Order Book Features
  - [x] Add order book imbalance detection
  - [x] Implement volume spike detection
  - [x] Add market depth analysis
  - [x] Implement depth imbalance calculation
  - [x] Add cumulative volume delta tracking
  - [x] Implement volume pressure calculation
  - [x] Add significant imbalance detection
- [ ] 1.5 Implement Volatility Features
- [ ] 1.6 Implement Trend Features
- [ ] 1.7 Implement Volume Profile Features
- [ ] 1.8 Implement Market Microstructure Features
- [ ] 1.9 Implement Time-Based Features
- [ ] 1.10 Implement Sentiment Features

## Completed Tasks
- [x] 1.1 Implement Lagged Returns
  - Added lagged returns for 1, 2, and 3 bars
  - Verified calculations with unit tests
- [x] 1.2 Implement Rolling Statistics
  - Added rolling mean and standard deviation calculations
  - Verified calculations with unit tests
- [x] 1.3 Enhance Technical Indicators
  - Added ROC calculations for multiple periods (1, 3, 5, 10 bars)
  - Implemented Volume ROC with normalization
  - Verified all features with unit tests
- [x] 1.4 Implement Order Book Features
  - Added order book imbalance detection with threshold-based significance
  - Implemented volume spike detection using rolling statistics
  - Added market depth analysis and depth imbalance calculation
  - Implemented cumulative volume delta tracking
  - Added volume pressure calculation
  - Verified all features with unit tests

## Next Steps
1. Implement Volatility Features (Task 1.5)
   - Add historical volatility calculation
   - Implement realized volatility estimation
   - Add volatility ratio indicators
   - Implement volatility regime detection

---

## Executor's Feedback or Assistance Requests

- **Audit of gas-specific features in AI/ML code is underway.**
- Initial findings confirm extensive use of seasonality, EIA, storage, and contract logic in feature engineering and model inputs across market_analysis, risk_management, and strategies.
- Next step: Begin systematic removal/refactoring of these features, starting with market_analysis/.

---

## Step 4: Detailed Plan for Enhanced Dynamic Mean Reversion (XGBoost Day Trading)

### Vision Statement (Day Trading, XGBoost Variant)
Deliver a high-performance, interpretable, and robust intraday mean reversion strategy for crypto, leveraging XGBoost for signal generation, with all features derived from standard exchange data (OHLCV, order book, volume). The system should be modular, testable, and ready for both backtesting and live trading.

### User-Visible Outcomes
- Users can select the XGBoost mean reversion strategy for intraday trading.
- The system fetches live or historical data, computes advanced features, and generates trade signals with probability/confidence scores.
- All trades are executed with dynamic position sizing and risk controls.
- Users can view model explanations (feature importances, signal breakdowns).

### High-Level Task Breakdown: XGBoost Day Trading Mean Reversion

#### 1. Model Integration & Interface
1.1. Add XGBoost as a dependency and utility wrapper (train, predict, save/load). ✅
1.2. Design a new `XGBoostMeanReversionStrategy` class, inheriting from `TechnicalStrategy`. ✅
1.3. Implement model training, prediction, and probability thresholding logic. ✅
1.4. Add model explainability (feature importances, SHAP values if possible). ✅

#### 2. Feature Engineering Pipeline
2.1. Define and implement all required features: ✅
  - Bollinger %B, Z-Score, VWAP Deviation, RSI, Order Flow Imbalance
  - DMI+ADX, Volume Profile, Market Depth Imbalance, Technical Divergence
2.2. Ensure features are efficiently calculated and reusable (extend `utils/technical_indicators.py` and `market_analysis/signal_enhancer.py` as needed).
2.3. Add robust feature validation and missing data handling.

#### 3. Entry/Exit Logic
3.1. Implement entry/exit rules based on XGBoost model probability thresholds.
3.2. Add secondary indicator confirmation (e.g., require agreement from ADX, VWAP, or order flow).
3.3. Integrate with dynamic stop loss/take profit logic.

#### 4. Position Sizing & Risk Management
4.1. Integrate with DQNParameterOptimizer for volatility-adjusted sizing.
4.2. Add risk checks for max position, max risk per trade, and portfolio heat.

#### 5. Backtesting & Testing
5.1. Add unit and integration tests for all new components.
5.2. Implement backtest hooks for the new strategy (if not present).
5.3. Validate performance and edge cases with historical data.

#### 6. Documentation & Interface Contracts
6.1. Document the new strategy class, feature pipeline, and model interface.
6.2. Update system architecture and navigation guides.

---

### Current Status / Progress Tracking
- ✅ Task 1.1: XGBoost dependency and utility wrapper implemented
- ✅ Task 1.2: XGBoostMeanReversionStrategy class designed and implemented
- ✅ Task 1.3: Model training, prediction, and probability thresholding implemented
- ✅ Task 1.4: Model explainability with SHAP values and feature importance added
- ✅ Task 2.1: Feature engineering pipeline implemented with all required features
- 🔄 Task 2.2: Feature calculation optimization in progress
- ⏳ Task 2.3: Feature validation and missing data handling pending
- ⏳ Task 3.1: Entry/exit rules implementation pending
- ⏳ Task 3.2: Secondary indicator confirmation pending
- ⏳ Task 3.3: Dynamic stop loss/take profit integration pending
- ⏳ Task 4.1: DQNParameterOptimizer integration pending
- ⏳ Task 4.2: Risk management checks pending
- ⏳ Task 5.1: Unit and integration tests pending
- ⏳ Task 5.2: Backtest hooks implementation pending
- ⏳ Task 5.3: Performance validation pending
- ⏳ Task 6.1: Documentation pending
- ⏳ Task 6.2: Architecture updates pending

### Next Steps
1. Optimize feature calculations in `utils/technical_indicators.py` and `market_analysis/signal_enhancer.py`
2. Implement robust feature validation and missing data handling
3. Design and implement entry/exit rules based on XGBoost model probabilities

### Executor's Feedback or Assistance Requests
- Need to verify if all required features are being calculated efficiently
- May need assistance with optimizing feature calculations for large datasets
- Should consider adding feature importance visualization tools

### Lessons
- XGBoost model should be trained with early stopping to prevent overfitting
- Feature importance can help identify which indicators are most predictive
- SHAP values provide valuable insights into model predictions
- Model predictions should be combined with traditional technical indicators for better signal validation

---

## Step 5: Detailed Plan for Enhanced Dynamic Mean Reversion (LSTM Swing Trading)

### Vision Statement (Swing Trading, LSTM Variant)
Deliver a robust, interpretable, and high-performance swing trading strategy for crypto, leveraging a Bidirectional LSTM with attention for mean reversion signal generation. All features are derived from standard exchange data (OHLCV, order book, volume). The system should be modular, testable, and ready for both backtesting and live trading.

### User-Visible Outcomes
- Users can select the LSTM mean reversion strategy for multi-day swing trading.
- The system fetches historical data, computes advanced sequential features, and generates trade signals with predicted reversion magnitude and confidence.
- Trades are executed with dynamic position sizing and robust risk controls.
- Users can view model explanations (feature attributions, attention maps).
- **Users can run both day trading (XGBoost) and swing trading (LSTM) strategies in parallel, with independent or combined risk management and execution.**

### High-Level Task Breakdown: LSTM Swing Trading Mean Reversion

#### 1. Model Integration & Interface
1.1. Add PyTorch as a dependency and utility wrapper (train, predict, save/load).
1.2. Design a new `LSTMSwingMeanReversionStrategy` class, inheriting from `TechnicalStrategy`.
1.3. Implement Bidirectional LSTM with attention for sequence modeling.
1.4. Add model explainability (attention visualization, feature attributions).

#### 2. Feature Engineering Pipeline
2.1. Define and implement all required sequential features:
  - Price reversion speed (rolling mean reversion metrics)
  - Order flow sequences (multi-step order book/volume features)
  - RSI trajectories (multi-step RSI, momentum)
  - Volatility and liquidity patterns
2.2. Ensure features are efficiently calculated and reusable (extend `utils/technical_indicators.py` and `market_analysis/signal_enhancer.py` as needed).
2.3. Add robust feature validation and missing data handling.

#### 3. Entry/Exit Logic
3.1. Implement entry/exit rules based on predicted reversion magnitude and model confidence.
3.2. Add confirmation filters (e.g., volatility regime, order flow agreement).
3.3. Integrate with dynamic stop loss/take profit logic.

#### 4. Position Sizing & Risk Management
4.1. Integrate with DQNParameterOptimizer or volatility-based sizing.
4.2. Add risk checks for max position, max risk per trade, and portfolio heat.

#### 5. Backtesting & Testing
5.1. Add unit and integration tests for all new components.
5.2. Implement backtest hooks for the new strategy (if not present).
5.3. Validate performance and edge cases with historical data.

#### 6. Documentation & Interface Contracts
6.1. Document the new strategy class, feature pipeline, and model interface.
6.2. Update system architecture and navigation guides.

---

### Interface/Architecture Notes
- **Multi-Strategy Support:** The system is designed to run both day trading (XGBoost) and swing trading (LSTM) strategies in parallel, with clear configuration for enabling/disabling each and managing their risk/execution independently or in aggregate.
- **Model Interface:** All ML models (XGBoost, LSTM, Bayesian NN) should implement a common interface for train/predict/explain.
- **Feature Store:** Centralize feature engineering to avoid duplication and ensure consistency.
- **Strategy Factory:** Register the new strategy for easy instantiation and configuration.
- **Exchange Data:** Ensure all features can be derived from OHLCV, order book, and volume data (no external/sentiment sources).

---

### Project Status Board (LSTM Swing Trading)
- [x] Add PyTorch dependency and utility wrapper
- [x] Implement LSTMSwingMeanReversionStrategy class
- [x] Build sequential feature pipeline for all required indicators
- [x] Integrate model training, prediction, and thresholding
- [x] Add model explainability (attention/feature attributions)
- [x] Implement entry/exit and confirmation logic
- [ ] Integrate with risk management and position sizing
- [x] Add unit/integration tests
- [ ] Document new components and update navigation guides

---

## LSTM Swing Mean Reversion Strategy — Documentation & Navigation Guide

### Purpose and Usage
- Implements a swing trading mean reversion strategy for crypto futures using a Bidirectional LSTM with attention.
- Designed for multi-day trades, leveraging sequential features from OHLCV, order book, and volume data.
- Modular, testable, and ready for backtesting, simulation, and (eventually) live trading.

### Configuration
- `LSTMSwingMeanReversionConfig` allows customization of:
  - `model_path`: Where to save/load the trained model
  - `sequence_length`: Number of timesteps in each input sequence (default: 14)
  - `feature_list`: List of sequential features to use (auto-populated if None)
  - `probability_threshold`: Minimum model output for a trade signal (default: 0.6)

### Feature Pipeline
- Features are engineered in `prepare_features`:
  - Price reversion speed (z-score)
  - Order flow imbalance and bid/ask volume sequences
  - RSI sequence and momentum
  - Volatility, volume profile, VWAP deviation, ADX, technical divergence
  - Funding rate and on-chain metrics (placeholders for future expansion)
- Output is a DataFrame ready for LSTM input (samples × timesteps × features).

### Model Training and Prediction
- `train`: Trains the LSTM model using a PyTorch DataLoader, criterion, and optimizer.
- `predict`: Runs inference on new data, returning predictions and attention weights.
- `save_model`/`load_model`: Persist and restore model weights.

### Explainability
- `explain`: Returns attention weights and a summary for any input, supporting downstream visualization and interpretability.
- Useful for understanding which timesteps/features the model focused on for each prediction.

### Signal Logic
- `calculate_signals`: Combines model prediction, confidence, and secondary indicator confirmation (ADX, RSI) to generate robust entry/exit signals.
- Only issues a trade if both the model and confirmation logic agree, and confidence is above a minimum threshold.

### How to Run Tests
- Tests are in `tests/test_lstm_swing_mean_reversion.py`.
- Covers feature pipeline, signal logic, explainability, and end-to-end workflow.
- Run with: `pytest tests/test_lstm_swing_mean_reversion.py`

### System Integration
- This strategy is registered in the strategy factory and can be selected for swing trading in the overall trading engine.
- Designed to be compatible with the modular, cloud-ready architecture (see project vision and architecture sections above).

--- 

## Executor's Feedback or Assistance Requests
- The sequential feature pipeline for LSTM swing mean reversion is implemented in prepare_features of LSTMSwingMeanReversionConfig (strategies/lstm_swing_mean_reversion.py) and is ready for use in model training.
- Model training, prediction, and thresholding are now implemented in LSTMSwingMeanReversionStrategy, enabling end-to-end model training and inference.
- Model explainability is now implemented: the explain method returns attention weights and a summary for any input, supporting downstream visualization and interpretability.
- Entry/exit and confirmation logic is now implemented in calculate_signals, using model prediction, confidence, and secondary indicator confirmation (ADX, RSI).
- Unit and integration tests are implemented in tests/test_lstm_swing_mean_reversion.py, covering features, signal logic, explainability, and end-to-end workflow.
- **Next step:** Document new components and update navigation guides for the LSTM swing mean reversion strategy.

---

## Current State & Immediate Next Step (as of [14/05/2025])
- The XGBoost mean reversion strategy's feature pipeline is complete and tested.
- All core logic for model training, prediction, and explainability is implemented.
- **Next actionable step:** Train the XGBoost model on prepared features and historical data, then validate outputs and document results.

---

# Phase 1: Core Trading Logic & AI Foundation — Detailed Breakdown

## 1. Refactor/Adapt All Strategies for Crypto Futures
- [x] Audit all strategies for gas-specific logic and dependencies
- [x] Remove or replace all non-crypto logic (seasonality, EIA, contract size, etc.)
- [x] Add crypto-specific features (funding rates, perpetual swaps, exchange events)
- [x] Modularize strategy interfaces for easy extension
- [x] Validate strategies on crypto data (signals generated, no gas-specific artifacts)

**Success Criteria:**
- All strategies run on crypto data, produce plausible signals, and are free of legacy logic

## 2. Implement/Validate AI/ML Feature Engineering and Model Training
- [x] Define and implement feature engineering pipelines for LSTM and XGBoost (OHLCV, order book, volume, etc.)
- [ ] Validate feature sets on real crypto data (no missing/NaN, correct shapes, meaningful values)
- [ ] Prepare and clean historical data for model training (train/test split, normalization, etc.)
- [ ] Implement model training routines for LSTM and XGBoost (with config options)
- [ ] Add model evaluation metrics (accuracy, precision, recall, PnL, etc.)
- [ ] Implement model explainability (feature importances for XGBoost, attention/attribution for LSTM)
- [ ] Save/load trained models and feature pipelines

**Success Criteria:**
- Models can be trained end-to-end on historical crypto data
- Feature pipelines are robust and reusable
- Model outputs and explanations are available for inspection

## 3. Build/Test Core Trading Engine Logic (Simulation/Backtesting)
- [ ] Integrate strategies and AI models into the trading engine
- [ ] Implement order simulation (fills, slippage, fees, leverage, liquidation)
- [ ] Add position management, PnL calculation, and risk controls
- [ ] Build a backtesting loop (run over historical data, log trades, compute metrics)
- [ ] Add logging and debugging output for all major events
- [ ] Validate engine with known scenarios (edge cases, error handling)

**Success Criteria:**
- Trading engine can run full backtests with any strategy/model
- All trades, positions, and metrics are logged and available for review
- Engine handles edge cases and errors gracefully

## 4. Ensure All Components Are Modular and Testable
- [ ] Write unit tests for feature engineering, model training, and trading logic
- [ ] Add integration tests for end-to-end workflows
- [ ] Document interfaces and expected inputs/outputs for each module
- [ ] Refactor code for clarity, separation of concerns, and extensibility

**Success Criteria:**
- All major modules have passing unit/integration tests
- Codebase is documented and easy to extend

---

# AI-Powered Order Flow Strategy Integration

## Current Status / Progress Tracking

### Completed Components
1. Core Strategy Implementation
   - ✅ Created `AIOrderFlowStrategy` class
   - ✅ Implemented LSTM integration for day trading
   - ✅ Implemented XGBoost integration for swing trading
   - ✅ Added feature preparation for both models
   - ✅ Implemented signal calculation and explanation
   - ✅ Created comprehensive test suite

2. Utility Functions
   - ✅ Created `xgboost_utils.py` with model training and prediction functions
   - ✅ Implemented SHAP-based model explanation
   - ✅ Added model persistence (save/load) functionality

### Pending Requirements

1. Data Requirements
   - Historical Data:
     - High-frequency order book data (1-min intervals)
     - Multi-hour to multi-day aggregated data
     - On-chain metrics (e.g., transaction volume, active addresses)
     - Exchange reserve data
   - Real-time Data:
     - Order book snapshots
     - Trade flow data
     - Market depth information

2. Exchange Integration
   - API Access Requirements:
     - Real-time order book data
     - Trade execution capabilities
     - Account management
     - Market data streaming
   - Exchange Client Implementation:
     - Authentication and security
     - Rate limiting and error handling
     - WebSocket connections for real-time data
     - REST API integration for historical data

3. Model Training & Validation
   - Data Pipeline:
     - Data collection and storage
     - Feature engineering pipeline
     - Data validation and cleaning
   - Model Training:
     - LSTM model training with high-frequency data
     - XGBoost model training with aggregated features
     - Cross-validation and performance metrics
   - Model Validation:
     - Backtesting framework
     - Performance metrics calculation
     - Risk analysis

4. Deployment & Testing
   - Paper Trading Environment:
     - Simulated order execution
     - Performance monitoring
     - Risk management controls
   - Production Environment:
     - Monitoring and alerting
     - Error handling and recovery
     - Performance optimization

## Next Steps

1. Data Collection
   - Identify and select data providers
   - Set up data collection pipeline
   - Implement data storage solution

2. Exchange Integration
   - Select and set up exchange API access
   - Implement exchange client
   - Test API connectivity and data flow

3. Model Training
   - Prepare historical data
   - Train and validate models
   - Implement model versioning

4. Testing & Deployment
   - Set up paper trading environment
   - Implement monitoring and logging
   - Deploy to production

## Lessons Learned
- Always validate data availability before implementing complex features
- Consider rate limits and API restrictions when designing real-time systems
- Implement proper error handling and recovery mechanisms
- Use paper trading environment for initial testing
- Document all API integrations and data sources

## Technical Debt
- Need to implement proper data validation and cleaning
- Add comprehensive error handling for API calls
- Implement proper logging and monitoring
- Add performance optimization for real-time data processing

## Success Metrics
1. Model Performance:
   - LSTM accuracy > 70% for day trading
   - XGBoost accuracy > 65% for swing trading
   - False positive rate < 20%

2. Trading Performance:
   - Sharpe ratio > 2.0
   - Maximum drawdown < 15%
   - Win rate > 55%

3. System Performance:
   - API latency < 100ms
   - Signal generation time < 50ms
   - System uptime > 99.9%

---

# Momentum Strategy Implementation Plan

## Background and Motivation
We're enhancing our existing momentum strategy to better handle 15-minute bar data and incorporate more sophisticated technical indicators and machine learning models. The goal is to improve the strategy's ability to identify and capitalize on price momentum patterns in financial markets.

## Key Challenges and Analysis
1. Data Granularity: Transitioning to 15-minute bars requires careful handling of higher-frequency data and potential noise
2. Feature Engineering: Need to balance computational efficiency with feature richness
3. Model Integration: Ensuring smooth integration of ML models with existing technical analysis
4. Testing Framework: Need comprehensive testing for both technical indicators and ML components

## High-level Task Breakdown

### 1. Feature Engineering Enhancement (1.0)
1.1. Implement Lagged Returns (1.1)
- Calculate returns for 1, 2, and 3 bars
- Success Criteria: Unit tests passing for all lag periods
- Dependencies: None

1.2. Rolling Statistics (1.2)
- Implement rolling mean and std for 5-bar window
- Success Criteria: Unit tests passing for both statistics
- Dependencies: None

1.3. Technical Indicators Enhancement (1.3)
- Add ROC for multiple periods (1, 3, 5, 10 bars)
- Implement Volume ROC
- Success Criteria: Unit tests passing for all new indicators
- Dependencies: None

1.4. Order Book Features (1.4)
- Implement order book imbalance detection
- Add volume spike detection
- Success Criteria: Unit tests passing for both features
- Dependencies: None

### 2. Model Integration (2.0)
2.1. XGBoost Implementation (2.1)
- Create XGBoost model wrapper class
- Implement feature preprocessing
- Success Criteria: Model training and prediction working
- Dependencies: 1.0

2.2. Model Training Pipeline (2.2)
- Implement cross-validation
- Add hyperparameter tuning
- Success Criteria: Model performance metrics meeting targets
- Dependencies: 2.1

2.3. LSTM Preparation (2.3)
- Design LSTM architecture
- Create data preparation pipeline
- Success Criteria: Architecture design complete
- Dependencies: 2.1

### 3. Testing Framework (3.0)
3.1. Unit Tests (3.1)
- Create tests for all new features
- Add model validation tests
- Success Criteria: All tests passing
- Dependencies: 1.0, 2.0

3.2. Integration Tests (3.2)
- Test feature pipeline integration
- Test model integration
- Success Criteria: All integration tests passing
- Dependencies: 3.1

### 4. Performance Optimization (4.0)
4.1. Feature Computation Optimization (4.1)
- Optimize rolling calculations
- Implement parallel processing where possible
- Success Criteria: Performance benchmarks met
- Dependencies: 1.0

4.2. Model Inference Optimization (4.2)
- Optimize model prediction pipeline
- Implement batch processing
- Success Criteria: Latency targets met
- Dependencies: 2.0

## Project Status Board
- [x] 1.1 Implement Lagged Returns
- [x] 1.2 Implement Rolling Statistics
- [x] 1.3 Enhance Technical Indicators
- [x] 1.4 Implement Order Book Features
- [🔄] 2.1 Implement XGBoost Model
    - [x] Create XGBoost model wrapper class
    - [x] Implement feature preprocessing
    - [ ] Add model validation and testing
    - [ ] Implement model persistence
    - [ ] Add model explainability
- [🔄] 2.2 Create Model Training Pipeline
    - [x] Implement DataPreprocessor class
    - [x] Implement FeatureGenerator class
    - [x] Implement ModelTrainer class
    - [x] Implement ModelRegistry class
    - [x] Implement TrainingOrchestrator class
    - [x] Implement SyntheticDataGenerator for testing
- [x] 2.3 Design LSTM Architecture
    - [x] Define LSTM model architecture
    - [x] Implement sequence data preparation
    - [x] Create attention mechanism
    - [x] Design model training pipeline
    - [x] Add model validation framework
    - [x] Implement model persistence
    - [x] Add model explainability
- [🔄] 3.1 Create Unit Tests
    - [x] Test feature engineering components
        - [x] Test lagged returns
        - [x] Test rolling statistics
        - [x] Test technical indicators
        - [x] Test order book features
    - [x] Test model training pipeline
        - [x] Test data preprocessing
        - [x] Test feature generation
        - [x] Test model training
        - [ ] Test model validation
        - [ ] Test model persistence
    - [x] Test data preprocessing
        - [x] Test data cleaning
        - [x] Test feature scaling
        - [x] Test feature statistics
        - [x] Test data validation
    - [ ] Test model validation
        - [ ] Test model metrics
        - [ ] Test feature importance
        - [ ] Test model explanation
    - [ ] Test model persistence
        - [ ] Test model saving
        - [ ] Test model loading
        - [ ] Test state management
    - [x] Test synthetic data generation
        - [x] Test OHLCV data generation
        - [x] Test order book data generation
        - [x] Test feature generation
        - [x] Test target generation
        - [x] Test reproducibility
- [ ] 3.2 Create Integration Tests
    - [ ] Test end-to-end training pipeline
    - [ ] Test model deployment workflow
    - [ ] Test data pipeline integration
    - [ ] Test model registry integration
    - [ ] Test training orchestration
- [ ] 4.1 Optimize Feature Computation
    - [ ] Profile current feature computation
    - [ ] Implement parallel processing
    - [ ] Optimize memory usage
    - [ ] Add caching mechanism
    - [ ] Implement batch processing
    - [ ] Add performance monitoring
- [ ] 4.2 Optimize Model Inference
    - [ ] Profile current inference pipeline
    - [ ] Implement batch prediction
    - [ ] Optimize model loading
    - [ ] Add prediction caching
    - [ ] Implement model quantization
    - [ ] Add inference monitoring

## Architecture Decision Records
1. Feature Engineering
   - Decision: Use pandas for feature computation
   - Rationale: Efficient for time series operations
   - Alternatives: NumPy (less convenient for time series)

2. Model Selection
   - Decision: Start with XGBoost
   - Rationale: Good balance of performance and interpretability
   - Alternatives: Random Forest (less powerful), Neural Networks (more complex)

## Dependencies
- pandas
- numpy
- xgboost
- scikit-learn
- tensorflow (for future LSTM implementation)

## Component Interfaces
1. Feature Engineering Interface
```python
class FeatureEngine:
    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame
    def validate_features(self, features: pd.DataFrame) -> bool
```

2. Model Interface
```python
class MomentumModel:
    def train(self, features: pd.DataFrame, targets: pd.Series) -> None
    def predict(self, features: pd.DataFrame) -> np.ndarray
    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]
```

## Technical Debt
- Need to implement proper logging
- Consider adding feature importance analysis
- Plan for model versioning system
- [ ] Complete XGBoost testing for momentum strategy
  - Need to implement comprehensive tests for the XGBoost model in momentum strategy
  - Current tests are failing due to missing features and data handling issues
  - Priority: High - This is critical for ensuring model reliability

## Test Coverage
- Unit tests for all new features
- Integration tests for model pipeline
- Performance benchmarks for feature computation

## Performance Benchmarks
- Feature computation: < 100ms per 1000 bars
- Model prediction: < 50ms per batch
- Memory usage: < 1GB for typical dataset

## Vision Statement
Create a robust momentum trading strategy that effectively identifies and capitalizes on price momentum patterns using both traditional technical analysis and modern machine learning approaches.

## User Stories
1. As a trader, I want to identify strong momentum patterns in 15-minute bars
2. As a trader, I want to understand why the model made a particular prediction
3. As a developer, I want to easily add new features to the strategy

## Success Metrics
1. Model accuracy > 60% on out-of-sample data
2. Feature computation time < 100ms per 1000 bars
3. Strategy Sharpe ratio > 1.5 in backtesting

## Priorities
1. Feature engineering implementation
2. XGBoost model integration
3. Testing framework
4. Performance optimization

## Executor's Feedback or Assistance Requests
- Completed implementation of lagged returns and rolling statistics
- Enhanced technical indicators with multiple ROC periods (1, 3, 5, 10) and Volume ROC
- All features have been tested and verified with unit tests
- Ready to proceed with implementing Order Book Features (Task 1.4)

## Lessons
None yet - will be populated during implementation.

---

## Task 2.2: Create Model Training Pipeline (Planning)

### Background and Motivation
A robust model training pipeline is essential for reproducible research, backtesting, and production retraining. The pipeline should automate data ingestion, feature engineering, data splitting, model training, and evaluation for all supported models (XGBoost, LSTM, etc.), ensuring consistency and scalability.

### Key Challenges and Analysis
- Handling large, potentially messy historical data efficiently
- Ensuring feature engineering is consistent between research and production
- Supporting multiple model types (XGBoost, LSTM, etc.) with a unified interface
- Automating data splits (train/validation/test) and cross-validation
- Logging and tracking model parameters, metrics, and artifacts
- Making the pipeline modular and extensible for future models/features

### User-Visible Outcome
- Users (and developers) can run a single command or script to train any supported model on historical data, with all steps (feature engineering, splitting, training, evaluation) handled automatically.
- The pipeline outputs trained model files, evaluation metrics, and logs for reproducibility.

### High-Level Task Breakdown: Model Training Pipeline (Task 2.2)

#### 1. Data Preparation Pipeline
1.1. Create DataPreprocessor class
    - Implement data cleaning and validation
    - Add feature scaling/normalization
    - Handle missing values and outliers
    - Add data versioning and caching

1.2. Implement Feature Engineering Pipeline
    - Create FeatureGenerator class
    - Add feature selection and importance tracking
    - Implement feature interaction detection
    - Add feature versioning

1.3. Create Data Splitting and Validation
    - Implement time-based cross-validation
    - Add stratified sampling for imbalanced data
    - Create validation set generation
    - Add data leakage prevention

#### 2. Model Training Pipeline
2.1. Create ModelTrainer class
    - Implement hyperparameter optimization
    - Add early stopping and model checkpointing
    - Create training progress tracking
    - Add model versioning

2.2. Implement Model Evaluation
    - Add comprehensive metrics calculation
    - Create performance visualization
    - Implement model comparison tools
    - Add cross-validation results tracking

2.3. Create Model Registry
    - Implement model storage and versioning
    - Add model metadata tracking
    - Create model deployment pipeline
    - Add model rollback capability

#### 3. Training Orchestration
3.1. Create TrainingOrchestrator class
    - Implement training job scheduling
    - Add resource monitoring
    - Create training logs and metrics storage
    - Add error handling and recovery

3.2. Implement Training Monitoring
    - Add real-time training metrics
    - Create training visualization
    - Implement alert system
    - Add performance tracking

#### 4. Testing and Validation
4.1. Create Pipeline Tests
    - Add unit tests for each component
    - Implement integration tests
    - Create end-to-end tests
    - Add performance benchmarks

4.2. Implement Validation Framework
    - Add data validation tests
    - Create model validation tests
    - Implement pipeline validation
    - Add security and compliance checks

Success Criteria:
1. All components are modular and independently testable
2. Pipeline can handle both XGBoost and LSTM models
3. Training process is reproducible and versioned
4. Performance metrics are tracked and visualized
5. Pipeline can be run both locally and in cloud environment
6. All components have comprehensive test coverage
7. Documentation is complete and up-to-date

---

## LSTM Architecture Design for Momentum Strategy

### Vision Statement (LSTM Momentum)
Implement a robust LSTM-based momentum strategy that can capture complex temporal patterns in price movements, order flow, and market microstructure. The model should be able to learn both short-term and long-term dependencies in the data, with attention mechanisms to focus on the most relevant features at each time step.

### Architecture Overview
1. Model Architecture
   - Bidirectional LSTM layers for capturing both forward and backward dependencies
   - Multi-head attention mechanism for feature importance weighting
   - Residual connections for better gradient flow
   - Dropout layers for regularization
   - Dense layers for final prediction

2. Input Features
   - Price-based features (returns, momentum, volatility)
   - Volume-based features (volume profile, order flow)
   - Technical indicators (RSI, MACD, ADX)
   - Market microstructure features (order book imbalance, depth)
   - Time-based features (time of day, day of week)

3. Training Pipeline
   - Sequence preparation with sliding windows
   - Feature normalization and scaling
   - Data augmentation for robustness
   - Cross-validation with time-based splits
   - Early stopping and model checkpointing

4. Validation Framework
   - Performance metrics (accuracy, precision, recall, F1)
   - Trading metrics (sharpe ratio, sortino ratio, max drawdown)
   - Statistical tests (stationarity, autocorrelation)
   - Out-of-sample testing
   - Walk-forward analysis

5. Model Persistence
   - Save/load model architecture and weights
   - Version control for model artifacts
   - Model registry for tracking experiments
   - A/B testing framework

6. Explainability
   - Attention weight visualization
   - Feature importance analysis
   - SHAP value computation
   - Decision path analysis
   - Performance attribution

### Implementation Plan
1. Define LSTM model architecture
   - [ ] Create base LSTM model class
   - [ ] Implement bidirectional LSTM layers
   - [ ] Add attention mechanism
   - [ ] Implement residual connections
   - [ ] Add dropout layers
   - [ ] Create dense output layers

2. Implement sequence data preparation
   - [ ] Create sequence generator
   - [ ] Implement sliding window logic
   - [ ] Add feature normalization
   - [ ] Create data augmentation methods
   - [ ] Implement batch generation

3. Create attention mechanism
   - [ ] Implement multi-head attention
   - [ ] Add attention weight computation
   - [ ] Create attention visualization
   - [ ] Add attention-based feature importance

4. Design model training pipeline
   - [ ] Create training loop
   - [ ] Implement validation steps
   - [ ] Add early stopping
   - [ ] Create model checkpointing
   - [ ] Implement learning rate scheduling

5. Add model validation framework
   - [ ] Create performance metrics
   - [ ] Implement trading metrics
   - [ ] Add statistical tests
   - [ ] Create walk-forward analysis
   - [ ] Implement cross-validation

6. Implement model persistence
   - [ ] Create save/load methods
   - [ ] Implement version control
   - [ ] Add model registry
   - [ ] Create A/B testing framework

7. Add model explainability
   - [ ] Implement attention visualization
   - [ ] Add feature importance analysis
   - [ ] Create SHAP value computation
   - [ ] Implement decision path analysis
   - [ ] Add performance attribution

### Success Criteria
1. Model Performance
   - Outperforms baseline momentum strategy
   - Shows consistent performance across different market regimes
   - Demonstrates robustness to market noise
   - Maintains stable performance in live trading

2. Technical Requirements
   - Modular and maintainable code
   - Comprehensive test coverage
   - Efficient training and inference
   - Clear documentation
   - Easy to extend and modify

3. Business Value
   - Improved trading performance
   - Better risk management
   - More interpretable decisions
   - Faster adaptation to market changes
   - Reduced false signals

---

