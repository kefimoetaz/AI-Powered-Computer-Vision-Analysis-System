# Image Analysis System

An AI-powered system for automatically detecting and counting people, vehicles, and traffic lights in images. Built with YOLOv8 for high accuracy and optimized for batch processing.

## Features

- **People Detection**: Counts all visible people, including partial views
- **Vehicle Detection**: Detects cars, trucks, buses, motorcycles, and bicycles  
- **Traffic Light Analysis**: Counts traffic lights and classifies their colors (red, green, yellow)
- **Batch Processing**: Process multiple images efficiently with parallel execution
- **Confidence Scoring**: Provides confidence scores for all detections
- **Structured Output**: Returns results in consistent JSON format
- **Logging**: Comprehensive logging for monitoring and debugging

## Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

The system will automatically download the YOLOv8 model on first use.

## Quick Start

### Single Image Analysis

```python
from image_analyzer import ImageAnalyzer

# Initialize analyzer
analyzer = ImageAnalyzer(confidence_threshold=0.5)

# Analyze single image
result = analyzer.analyze_image("path/to/your/image.jpg")

# Print results
print(f"People: {result.people_count}")
print(f"Vehicles: {result.vehicle_count}")
print(f"Traffic lights: {result.traffic_lights}")
```

### Batch Processing

```python
from batch_processor import BatchProcessor

# Initialize batch processor
processor = BatchProcessor(confidence_threshold=0.5, max_workers=4)

# Process all images in a directory
results = processor.process_directory(
    input_dir="path/to/images/",
    output_file="results.json"
)
```

### Command Line Usage

Process a directory of images:
```bash
python batch_processor.py /path/to/images/ -o results.json
```

Options:
- `-c, --confidence`: Set confidence threshold (default: 0.5)
- `-w, --workers`: Number of parallel workers (default: 4)
- `--sequential`: Process images sequentially instead of parallel

## Output Format

The system returns results in this JSON structure:

```json
{
  "people_count": 12,
  "vehicle_count": 4,
  "traffic_lights": {
    "total": 2,
    "red": 1,
    "green": 1,
    "yellow": 0
  },
  "confidence_scores": {
    "people": 0.85,
    "vehicles": 0.92,
    "traffic_lights": 0.78
  },
  "processing_time": 0.234,
  "image_path": "path/to/image.jpg",
  "timestamp": "2024-01-15T10:30:45"
}
```

## Architecture

- **ImageAnalyzer**: Core detection engine using YOLOv8
- **BatchProcessor**: Handles parallel processing and directory scanning
- **Traffic Light Classification**: HSV-based color analysis for accurate light state detection
- **Confidence Scoring**: Per-category confidence metrics
- **Logging**: Comprehensive logging with file and console output

## Performance Optimization

- **Parallel Processing**: Multi-threaded batch processing
- **Efficient Model**: Uses YOLOv8n (nano) for speed while maintaining accuracy
- **Smart Filtering**: Excludes mannequins, photos, and statues from people counts
- **Memory Management**: Processes images individually to handle large batches

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- 4GB+ RAM recommended for batch processing

## Extending the System

The modular design makes it easy to add new object categories:

1. Add new class IDs to the analyzer
2. Implement detection logic in `ImageAnalyzer`
3. Update the output format as needed

## Troubleshooting

- **Model Download Issues**: Ensure internet connection for initial YOLOv8 download
- **Memory Errors**: Reduce `max_workers` for large images or limited RAM
- **Low Accuracy**: Adjust `confidence_threshold` or ensure good image quality
- **Performance**: Use GPU acceleration by installing `torch` with CUDA support

## License

MIT License - see LICENSE file for details.