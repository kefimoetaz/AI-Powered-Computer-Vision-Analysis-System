"""
Image Analysis System for counting people, vehicles, and traffic lights.
Uses YOLOv8 for object detection with custom filtering and classification.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch

@dataclass
class DetectionResult:
    """Structure for individual detection results"""
    people_count: int
    vehicle_count: int
    traffic_lights: Dict[str, int]
    confidence_scores: Dict[str, float]
    processing_time: float
    image_path: str
    timestamp: str

class ImageAnalyzer:
    """Main class for image analysis and object detection"""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the analyzer with YOLO model
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_path)
        
        # COCO class mappings for our target objects
        self.person_classes = [0]  # person
        self.vehicle_classes = [2, 3, 5, 6, 7]  # car, motorcycle, bus, train, truck
        self.bicycle_classes = [1]  # bicycle
        self.traffic_light_classes = [9]  # traffic light
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the analyzer"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('image_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_image(self, image_path: str) -> DetectionResult:
        """
        Analyze a single image and return detection results
        
        Args:
            image_path: Path to the image file
            
        Returns:
            DetectionResult object with counts and metadata
        """
        start_time = datetime.now()
        
        try:
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Run YOLO detection
            results = self.model(image_path, conf=self.confidence_threshold)
            
            # Process detections
            people_count, people_conf = self._count_people(results[0])
            vehicle_count, vehicle_conf = self._count_vehicles(results[0])
            traffic_lights, traffic_conf = self._analyze_traffic_lights(results[0], image)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = DetectionResult(
                people_count=people_count,
                vehicle_count=vehicle_count,
                traffic_lights=traffic_lights,
                confidence_scores={
                    'people': people_conf,
                    'vehicles': vehicle_conf,
                    'traffic_lights': traffic_conf
                },
                processing_time=processing_time,
                image_path=image_path,
                timestamp=datetime.now().isoformat()
            )
            
            self.logger.info(f"Processed {image_path}: {people_count} people, {vehicle_count} vehicles")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            raise
    
    def _count_people(self, results) -> Tuple[int, float]:
        """Count people in detection results"""
        person_detections = []
        
        if results.boxes is not None:
            for box in results.boxes:
                if int(box.cls) in self.person_classes:
                    person_detections.append(float(box.conf))
        
        count = len(person_detections)
        avg_confidence = np.mean(person_detections) if person_detections else 0.0
        
        return count, avg_confidence
    
    def _count_vehicles(self, results) -> Tuple[int, float]:
        """Count vehicles (cars, trucks, buses, motorcycles, bicycles)"""
        vehicle_detections = []
        
        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls)
                if cls in self.vehicle_classes or cls in self.bicycle_classes:
                    vehicle_detections.append(float(box.conf))
        
        count = len(vehicle_detections)
        avg_confidence = np.mean(vehicle_detections) if vehicle_detections else 0.0
        
        return count, avg_confidence
    
    def _analyze_traffic_lights(self, results, image) -> Tuple[Dict[str, int], float]:
        """Analyze traffic lights and classify their colors"""
        traffic_lights = {"total": 0, "red": 0, "green": 0, "yellow": 0}
        confidences = []
        
        if results.boxes is not None:
            for box in results.boxes:
                if int(box.cls) in self.traffic_light_classes:
                    confidences.append(float(box.conf))
                    
                    # Extract traffic light region for color analysis
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    traffic_light_roi = image[y1:y2, x1:x2]
                    
                    # Classify color
                    color = self._classify_traffic_light_color(traffic_light_roi)
                    traffic_lights["total"] += 1
                    traffic_lights[color] += 1
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        return traffic_lights, avg_confidence
    
    def _classify_traffic_light_color(self, roi) -> str:
        """Classify traffic light color using HSV analysis"""
        if roi.size == 0:
            return "yellow"  # Default fallback
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        
        yellow_lower = np.array([20, 50, 50])
        yellow_upper = np.array([40, 255, 255])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = red_mask1 + red_mask2
        
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Count pixels for each color
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        
        # Return dominant color
        max_pixels = max(red_pixels, green_pixels, yellow_pixels)
        if max_pixels == red_pixels and red_pixels > 0:
            return "red"
        elif max_pixels == green_pixels and green_pixels > 0:
            return "green"
        elif max_pixels == yellow_pixels and yellow_pixels > 0:
            return "yellow"
        else:
            return "yellow"  # Default fallback
    
    def batch_process(self, image_paths: List[str], output_file: Optional[str] = None) -> List[DetectionResult]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            output_file: Optional JSON file to save results
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        
        self.logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        for image_path in image_paths:
            try:
                result = self.analyze_image(image_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {str(e)}")
                continue
        
        # Save results if output file specified
        if output_file:
            self.save_results(results, output_file)
        
        self.logger.info(f"Batch processing completed. Processed {len(results)} images successfully")
        return results
    
    def save_results(self, results: List[DetectionResult], output_file: str):
        """Save results to JSON file"""
        json_results = []
        
        for result in results:
            json_results.append({
                "people_count": result.people_count,
                "vehicle_count": result.vehicle_count,
                "traffic_lights": result.traffic_lights,
                "confidence_scores": result.confidence_scores,
                "processing_time": result.processing_time,
                "image_path": result.image_path,
                "timestamp": result.timestamp
            })
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_file}")

def main():
    """Example usage of the ImageAnalyzer"""
    analyzer = ImageAnalyzer(confidence_threshold=0.5)
    
    # Example single image analysis
    # result = analyzer.analyze_image("path/to/your/image.jpg")
    # print(json.dumps(result.__dict__, indent=2))
    
    # Example batch processing
    # image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    # results = analyzer.batch_process(image_paths, "analysis_results.json")
    
    print("ImageAnalyzer initialized successfully!")
    print("Use analyzer.analyze_image('path/to/image.jpg') to process a single image")
    print("Use analyzer.batch_process(['img1.jpg', 'img2.jpg']) for batch processing")

if __name__ == "__main__":
    main()