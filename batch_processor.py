"""
Batch processing utility for the Image Analysis System.
Handles directory scanning, parallel processing, and result aggregation.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

from image_analyzer import ImageAnalyzer, DetectionResult

class BatchProcessor:
    """Enhanced batch processing with parallel execution and progress tracking"""
    
    def __init__(self, confidence_threshold: float = 0.5, max_workers: int = 4):
        """
        Initialize batch processor
        
        Args:
            confidence_threshold: Detection confidence threshold
            max_workers: Maximum number of parallel workers
        """
        self.analyzer = ImageAnalyzer(confidence_threshold=confidence_threshold)
        self.max_workers = max_workers
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def find_images(self, directory: str) -> List[str]:
        """
        Recursively find all supported image files in directory
        
        Args:
            directory: Root directory to search
            
        Returns:
            List of image file paths
        """
        image_paths = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_paths.append(str(file_path))
        
        return sorted(image_paths)
    
    def process_directory(self, input_dir: str, output_file: str = None, 
                         parallel: bool = True) -> List[DetectionResult]:
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing images
            output_file: Optional output JSON file
            parallel: Whether to use parallel processing
            
        Returns:
            List of detection results
        """
        image_paths = self.find_images(input_dir)
        
        if not image_paths:
            print(f"No supported images found in {input_dir}")
            return []
        
        print(f"Found {len(image_paths)} images to process")
        
        if parallel and len(image_paths) > 1:
            results = self._process_parallel(image_paths)
        else:
            results = self._process_sequential(image_paths)
        
        if output_file:
            self.save_results_with_summary(results, output_file)
        
        return results
    
    def _process_parallel(self, image_paths: List[str]) -> List[DetectionResult]:
        """Process images in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.analyzer.analyze_image, path): path 
                for path in image_paths
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(image_paths), desc="Processing images") as pbar:
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Error processing {path}: {str(e)}")
                    finally:
                        pbar.update(1)
        
        return results
    
    def _process_sequential(self, image_paths: List[str]) -> List[DetectionResult]:
        """Process images sequentially"""
        results = []
        
        for path in tqdm(image_paths, desc="Processing images"):
            try:
                result = self.analyzer.analyze_image(path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
        
        return results
    
    def save_results_with_summary(self, results: List[DetectionResult], output_file: str):
        """Save results with summary statistics"""
        # Calculate summary statistics
        total_people = sum(r.people_count for r in results)
        total_vehicles = sum(r.vehicle_count for r in results)
        total_traffic_lights = sum(r.traffic_lights['total'] for r in results)
        avg_processing_time = sum(r.processing_time for r in results) / len(results) if results else 0
        
        # Prepare output data
        output_data = {
            "summary": {
                "total_images_processed": len(results),
                "total_people_detected": total_people,
                "total_vehicles_detected": total_vehicles,
                "total_traffic_lights_detected": total_traffic_lights,
                "average_processing_time_seconds": round(avg_processing_time, 3),
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "detailed_results": []
        }
        
        # Add detailed results
        for result in results:
            output_data["detailed_results"].append({
                "image_path": result.image_path,
                "people_count": result.people_count,
                "vehicle_count": result.vehicle_count,
                "traffic_lights": result.traffic_lights,
                "confidence_scores": result.confidence_scores,
                "processing_time": result.processing_time,
                "timestamp": result.timestamp
            })
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        print(f"Summary: {len(results)} images, {total_people} people, {total_vehicles} vehicles, {total_traffic_lights} traffic lights")

def main():
    """Command line interface for batch processing"""
    parser = argparse.ArgumentParser(description="Batch process images for object detection")
    parser.add_argument("input_dir", help="Directory containing images to process")
    parser.add_argument("-o", "--output", help="Output JSON file for results")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, 
                       help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("-w", "--workers", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    parser.add_argument("--sequential", action="store_true",
                       help="Process images sequentially instead of in parallel")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BatchProcessor(
        confidence_threshold=args.confidence,
        max_workers=args.workers
    )
    
    # Set default output file if not specified
    output_file = args.output or f"batch_results_{int(time.time())}.json"
    
    # Process directory
    try:
        results = processor.process_directory(
            input_dir=args.input_dir,
            output_file=output_file,
            parallel=not args.sequential
        )
        
        print(f"\nBatch processing completed successfully!")
        print(f"Processed {len(results)} images")
        
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())