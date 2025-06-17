# MVP Backend Structure for Credit Scoring System
## "Improving Credit Scoring and Loan Recommendations with Hybrid Logistic Regression and Rule-Based Filtering"

```
C:.
│   .env
│   .gitignore
│   main.py
│   requirements.txt
│   docker-compose.yml (optional for MVP)
│
├───app
│   ├───ai
│   │   │   __init__.py
│   │   ├───ml
│   │   │   │   __init__.py
│   │   │   │   model_loader.py          # Load penalized logistic regression model on startup
│   │   │   │   predictor.py             # Credit scoring with penalized logistic regression
│   │   │   │   explainer.py             # Model explainability (coefficients, feature importance)
│   │   │   │   preprocessor.py          # Feature engineering and preprocessing pipeline
│   │   │   └───models
│   │   │       │   __init__.py
│   │   │       └───credit_model.py      # Penalized logistic regression model wrapper
│   │   ├───rules
│   │   │   │   __init__.py
│   │   │   │   loan_filters.py          # Rule-based filtering for loan recommendations
│   │   │   │   business_rules.py        # Business logic rules (income, credit score thresholds)
│   │   │   │   risk_assessment.py       # Risk-based filtering rules
│   │   │   └───rule_engine.py           # Rule engine orchestrator
│   │   └───chatbot
│   │       │   __init__.py
│   │       │   openrouter_client.py     # OpenRouter API integration for explanations
│   │       │   explanation_generator.py # Generate human-readable explanations
│   │       └───prompts.py               # Prompt templates for ML explanations
│   │
│   ├───api
│   │   │   __init__.py
│   │   │   auth_routes.py               # JWT authentication endpoints (existing)
│   │   │   user_routes.py               # User profile and data management
│   │   │   prediction_routes.py         # Credit scoring prediction endpoints
│   │   │   loan_routes.py               # Loan recommendation endpoints with rule-based filtering
│   │   │   chat_routes.py               # Chatbot explanation endpoints
│   │   └───dependencies.py              # FastAPI dependency injection
│   │
│   ├───core
│   │   │   config.py                    # Application configuration and ML hyperparameters
│   │   │   security.py                  # JWT security and authentication logic
│   │   │   dependencies.py              # Global FastAPI dependencies
│   │   │   __init__.py
│   │   └───constants.py                 # ML thresholds, rule-based filtering constants
│   │
│   ├───database
│   │   │   connection.py                # MongoDB async connection setup
│   │   │   __init__.py
│   │   └───repositories
│   │       │   __init__.py
│   │       │   user_repository.py       # User CRUD operations
│   │       │   prediction_repository.py # Store predictions for model retraining
│   │       └───loan_repository.py       # Loan products and application storage
│   │
│   ├───exceptions
│   │   │   __init__.py
│   │   │   custom_exceptions.py         # API exceptions
│   │   └───ml_exceptions.py             # ML-specific exceptions
│   │
│   ├───models
│   │   │   __init__.py
│   │   │   user.py                      # User MongoDB document model
│   │   │   user_details.py              # Extended user profile information
│   │   │   prediction.py                # Credit prediction results storage
│   │   │   loan_application.py          # User loan application data
│   │   │   loan_product.py              # Available loan products with rules
│   │   │   chat_session.py              # Chat conversation history
│   │   └───feedback.py                  # User feedback for continuous learning
│   │
│   ├───schemas
│   │   │   __init__.py
│   │   │   user.py                      # User request/response Pydantic models
│   │   │   user_details.py              # User profile validation schemas
│   │   │   prediction.py                # ML prediction input/output schemas
│   │   │   loan.py                      # Loan recommendation request/response schemas
│   │   │   chat.py                      # Chatbot conversation schemas
│   │   └───ml.py                        # Machine learning data validation schemas
│   │
│   ├───services
│   │   │   __init__.py
│   │   │   auth_service.py              # JWT authentication business logic
│   │   │   prediction_service.py        # Credit scoring orchestration service
│   │   │   loan_service.py              # Hybrid recommendation service (ML + Rules)
│   │   │   chat_service.py              # Chatbot explanation service with OpenRouter
│   │   └───notification_service.py      # Email/SMS notifications (optional)
│   │
│   ├───utils
│   │   │   __init__.py
│   │   │   validators.py                # Input validation
│   │   │   formatters.py                # Response formatting
│   │   │   cache.py                     # Simple caching for model
│   │   └───decorators.py                # Custom decorators
│   │
│   └───workers
│       │   __init__.py
│       │   model_trainer.py             # Automated model retraining with new data
│       └───data_processor.py            # ETL pipeline for prediction feedback data
│
├───data
│   ├───raw
│   │   │   a_Dataset_CreditScoring.xlsx   # Original credit scoring dataset
│   │   └───feedback                       # Real-world prediction outcomes for retraining
│   ├───processed
│   │   │   training_data.csv              # Preprocessed training dataset
│   │   │   validation_data.csv            # Model validation dataset
│   │   └───features.json                  # Feature engineering configuration
│   ├───models
│   │   │   credit_model.pkl               # Trained penalized logistic regression model
│   │   │   scaler.pkl                     # Feature scaling pipeline
│   │   │   model_metadata.json            # Model performance metrics and info
│   │   └───versions                       # Model version control for A/B testing
│   │       ├───v1
│   │       └───v2
│   ├───rules
│   │   │   loan_eligibility_rules.json    # Rule-based filtering criteria
│   │   │   risk_assessment_rules.json     # Risk-based loan filtering
│   │   └───business_rules.json            # Business logic rules
│   └───loan_products
│       └───products.json                  # Available loan products with criteria
│
├───scripts
│   │   train_model.py                     # Initial penalized logistic regression training
│   │   retrain_model.py                   # Continuous learning model retraining
│   │   preprocess_data.py                 # Data preprocessing and feature engineering
│   │   seed_loans.py                      # Initialize loan products database
│   └───create_rules.py                    # Generate rule-based filtering configurations
│
├───tests                                  # Basic testing (optional for MVP)
│   │   test_predictions.py
│   │   test_auth.py
│   └───test_data.json
│
├───docs
│   │   README.md                          # Project overview and setup instructions
│   │   API_DOCUMENTATION.md               # Complete API endpoint documentation
│   │   MODEL_DOCUMENTATION.md             # ML model architecture and performance
│   │   THESIS_NOTES.md                    # Research methodology and findings
│   └───DEPLOYMENT_GUIDE.md                # Production deployment instructions
│
├───logs                                   # Simple logging
│   │   app.log
│   └───ml.log
│
└───storage                                # Your existing storage
    └───temp
```

## Development Phases (5-Week Timeline to 50% Working System)

### **Week 1: Core ML Foundation**
**Goal**: Get penalized logistic regression working with basic predictions
**Deliverables**: 
- ✅ Model training pipeline
- ✅ Basic prediction API
- ✅ Model loading and caching
- ✅ Feature preprocessing

**Key Files to Create**:
1. `app/ai/ml/model_loader.py` - Load and cache trained model
2. `app/ai/ml/predictor.py` - Core prediction logic
3. `app/ai/ml/preprocessor.py` - Feature engineering pipeline
4. `scripts/train_model.py` - Initial model training
5. `app/schemas/ml.py` - ML input/output validation
6. `app/api/prediction_routes.py` - Prediction endpoints

**Success Criteria**: Can input user data and get credit score + approval probability

### **Week 2: Rule-Based Loan Filtering**
**Goal**: Implement rule-based filtering for loan recommendations
**Deliverables**:
- ✅ Rule engine for loan filtering
- ✅ Business logic rules
- ✅ Loan recommendation API
- ✅ Hybrid system (ML + Rules)

**Key Files to Create**:
7. `app/ai/rules/loan_filters.py` - Rule-based filtering logic
8. `app/ai/rules/business_rules.py` - Business eligibility rules
9. `app/ai/rules/rule_engine.py` - Rule orchestration
10. `app/services/loan_service.py` - Hybrid recommendation service
11. `app/api/loan_routes.py` - Loan recommendation endpoints
12. `data/rules/loan_eligibility_rules.json` - Rule configurations

**Success Criteria**: Can recommend suitable loans based on credit score + rule filtering

### **Week 3: Model Explainability**
**Goal**: Make predictions explainable and transparent
**Deliverables**:
- ✅ Coefficient-based explanations
- ✅ Feature importance analysis
- ✅ Human-readable explanations
- ✅ Basic chatbot integration

**Key Files to Create**:
13. `app/ai/ml/explainer.py` - Model explanation logic
14. `app/ai/chatbot/openrouter_client.py` - LLM API integration
15. `app/ai/chatbot/explanation_generator.py` - Generate explanations
16. `app/services/chat_service.py` - Chatbot orchestration
17. `app/api/chat_routes.py` - Chat endpoints

**Success Criteria**: Users can ask "Why was I approved/rejected?" and get clear explanations

### **Week 4: Data Persistence & User Experience**
**Goal**: Complete user journey with data storage
**Deliverables**:
- ✅ Store all predictions
- ✅ User history tracking
- ✅ Improved API responses
- ✅ Error handling

**Key Files to Create**:
18. `app/database/repositories/prediction_repository.py` - Prediction data storage
19. `app/database/repositories/loan_repository.py` - Loan data management
20. `app/models/prediction.py` - Prediction document model
21. `app/models/loan_application.py` - Loan application model
22. `app/services/prediction_service.py` - Business logic orchestration
23. `app/exceptions/ml_exceptions.py` - ML-specific error handling

**Success Criteria**: Complete user flow from registration → prediction → recommendations → explanations

### **Week 5: Model Improvement & Polish**
**Goal**: Implement continuous learning and system optimization
**Deliverables**:
- ✅ Model retraining pipeline
- ✅ Performance monitoring
- ✅ API documentation
- ✅ System optimization

**Key Files to Create**:
24. `app/workers/model_trainer.py` - Automated retraining
25. `app/workers/data_processor.py` - Data pipeline for retraining
26. `scripts/retrain_model.py` - Model retraining script
27. `app/models/feedback.py` - Feedback collection model
28. `docs/API_DOCUMENTATION.md` - Complete API docs
29. `app/utils/cache.py` - Performance optimization

**Success Criteria**: 50% working system with continuous improvement capability

## File Priority Classification

### **🔴 Critical Priority (Weeks 1-2)**
- Core ML pipeline files
- Basic API endpoints
- Rule-based filtering
- Database models

### **🟡 High Priority (Weeks 3-4)**
- Model explainability
- Chatbot integration
- Data persistence
- User experience

### **🟢 Medium Priority (Week 5)**
- Model retraining
- Performance optimization
- Documentation
- Monitoring

### **🔵 Low Priority (Post-MVP)**
- Advanced analytics
- A/B testing
- Deployment automation
- Comprehensive testing

## Success Metrics for 50% Working System

1. **✅ Functional Credit Scoring**: Users can input data and receive credit scores
2. **✅ Loan Recommendations**: System provides filtered loan suggestions
3. **✅ Basic Explanations**: Users understand why they got certain results
4. **✅ Data Storage**: All interactions are stored for future improvement
5. **✅ API Completeness**: All major endpoints are functional
6. **✅ Model Performance**: Acceptable accuracy on validation data
7. **✅ Rule Integration**: Hybrid system works seamlessly
8. **✅ User Journey**: Complete flow from input to recommendations

