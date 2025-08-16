"""
Test script for the Image Analysis System
Demonstrates usage and validates functionality
"""

import json
import os
from pathlib import Path
from image_analyzer import ImageAnalyzer
from batch_processor import BatchProcessor

def test_single_image():
    """Test single image analysis"""
    print("=== Testing Single Image Analysis ===")
    
    analyzer = ImageAnalyzer(confidence_threshold=0.5)
    
    # You can replace this with an actual image path for testing
    test_image = "test_image.jpg"
    
    if os.path.exists(test_image):
        try:
            result = analyzer.analyze_image(test_image)
            
            print(f"Image: {result.image_path}")
            print(f"People detected: {result.people_count}")
            print(f"Vehicles detected: {result.vehicle_count}")
            print(f"Traffic lights: {result.traffic_lights}")
            print(f"Processing time: {result.processing_time:.3f}s")
            print(f"Confidence scores: {result.confidence_scores}")
            
            # Save result as JSON
            result_dict = {
                "people_count": result.people_count,
                "vehicle_count": result.vehicle_count,
                "traffic_lights": result.traffic_lights,
                "confidence_scores": result.confidence_scores,
                "processing_time": result.processing_time,
                "image_path": result.image_path,
                "timestamp": result.timestamp
            }
            
            with open("single_test_result.json", "w") as f:
                json.dump(result_dict, f, indent=2)
            
            print("✓ Single image test completed successfully")
            
        except Exception as e:
            print(f"✗ Error testing single image: {str(e)}")
    else:
        print(f"Test image '{test_image}' not found. Place a test image and update the path.")

def test_batch_processing():
    """Test batch processing functionality"""
    print("\n=== Testing Batch Processing ===")
    
    # Create test directory structure
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        print(f"Test directory '{test_dir}' not found.")
        print("Create a 'test_images' directory with some images to test batch processing.")
        return
    
    processor = BatchProcessor(confidence_threshold=0.5, max_workers=2)
    
    try:
        # Find images in test directory
        image_paths = processor.find_images(test_dir)
        print(f"Found {len(image_paths)} images in {test_dir}")
        
        if image_paths:
            # Process images
            results = processor.process_directory(
                input_dir=test_dir,
                output_file="batch_test_results.json",
                parallel=True
            )
            
            print(f"✓ Batch processing completed: {len(results)} images processed")
            
            # Print summary
            total_people = sum(r.people_count for r in results)
            total_vehicles = sum(r.vehicle_count for r in results)
            total_lights = sum(r.traffic_lights['total'] for r in results)
            
            print(f"Summary: {total_people} people, {total_vehicles} vehicles, {total_lights} traffic lights")
        else:
            print("No images found in test directory")
            
    except Exception as e:
        print(f"✗ Error in batch processing: {str(e)}")

def validate_output_format():
    """Validate that output matches expected JSON format"""
    print("\n=== Validating Output Format ===")
    
    expected_keys = {
        "people_count", "vehicle_count", "traffic_lights", 
        "confidence_scores", "processing_time", "image_path", "timestamp"
    }
    
    expected_traffic_light_keys = {"total", "red", "green", "yellow"}
    expected_confidence_keys = {"people", "vehicles", "traffic_lights"}
    
    # Check if we have a test result file
    if os.path.exists("single_test_result.json"):
        with open("single_test_result.json", "r") as f:
            result = json.load(f)
        
        # Validate main structure
        result_keys = set(result.keys())
        if expected_keys.issubset(result_keys):
            print("✓ Main structure validation passed")
        else:
            missing = expected_keys - result_keys
            print(f"✗ Missing keys: {missing}")
        
        # Validate traffic lights structure
        if "traffic_lights" in result:
            tl_keys = set(result["traffic_lights"].keys())
            if expected_traffic_light_keys.issubset(tl_keys):
                print("✓ Traffic lights structure validation passed")
            else:
                missing = expected_traffic_light_keys - tl_keys
                print(f"✗ Missing traffic light keys: {missing}")
        
        # Validate confidence scores structure
        if "confidence_scores" in result:
            conf_keys = set(result["confidence_scores"].keys())
            if expected_confidence_keys.issubset(conf_keys):
                print("✓ Confidence scores structure validation passed")
            else:
                missing = expected_confidence_keys - conf_keys
                print(f"✗ Missing confidence score keys: {missing}")
        
        print("✓ Output format validation completed")
    else:
        print("No test result file found. Run single image test first.")

def main():
    """Run all tests"""
    print("Image Analysis System - Test Suite")
    print("=" * 50)
    
    # Test system initialization
    try:
        analyzer = ImageAnalyzer()
        print("✓ System initialization successful")
    except Exception as e:
        print(f"✗ System initialization failed: {str(e)}")
        return
    
    # Run tests
    test_single_image()
    test_batch_processing()
    validate_output_format()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("\nTo test with your own images:")
    print("1. Place test images in a 'test_images' directory")
    print("2. Or update the test_image path in test_single_image()")
    print("3. Run: python test_system.py")

if __name__ == "__main__":
    main()