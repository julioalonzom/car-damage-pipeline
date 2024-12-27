# Car Damage Assessment Pipeline

An end-to-end ML pipeline for detecting car damage, estimating repair costs, and generating detailed reports.

## 🚀 Features

- **Damage Detection**: Fine-tuned Vision Transformer (ViT) model for identifying damaged car parts
- **Cost Estimation**: Intelligent cost estimation based on detected damage
- **Report Generation**: Automated detailed reports using LLM
- **API Interface**: RESTful API built with FastAPI
- **Frontend**: React-based UI using shadcn/ui components

## 🛠 Tech Stack

- **ML/Vision**: PyTorch, Transformers, Albumentations
- **Backend**: FastAPI, Pydantic
- **Frontend**: React, Tailwind CSS, shadcn/ui
- **Infrastructure**: Docker
- **Documentation**: OpenAPI (Swagger)

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/car-damage-pipeline.git
cd car-damage-pipeline
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

4. Create `.env` file:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## 🚀 Quick Start

1. Start the API server:
```bash
python run.py
```

2. Visit the API documentation:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## 📁 Project Structure

```
car-damage-pipeline/
├── src/
│   ├── api/           # FastAPI application
│   ├── vision/        # Computer vision components
│   ├── cost/          # Cost estimation logic
│   ├── llm/           # Report generation
│   └── utils/         # Shared utilities
├── tests/             # Test suite
├── frontend/          # React frontend
├── docker/            # Docker configuration
└── notebooks/         # Development notebooks
```

## 🧪 Testing

Run the test suite:
```bash
pytest
```

With coverage:
```bash
pytest --cov=src tests/
```

## 🐳 Docker

Build and run with Docker:
```bash
docker compose up --build
```

## 📈 Model Performance

Current model metrics:
- Validation Loss: 0.2513
- Early stopping triggered after 8 epochs
- Using test-time augmentation for improved robustness

## 📚 API Documentation

The API documentation is available at:
- `/api/docs` for Swagger UI
- `/api/redoc` for ReDoc

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- COCO car damage dataset
- Hugging Face Transformers library
- FastAPI framework
- shadcn/ui components