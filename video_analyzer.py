"""
Video Analysis System for real-time processing of video feeds and files.
Extends the image analyzer to handle live streams and video files.
"""

import cv2
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from image_analyzer import ImageAnalyzer, DetectionResult

@dataclass
class VideoAnalysisResult:
    """Structure for video analysis results"""
    frame_number: int
    timestamp: float
    people_count: int
    vehicle_count: int
    traffic_lights: Dict[str, int]
    confidence_scores: Dict[str, float]
    processing_time: float

class VideoAnalyzer:
    """Video analysis system for live feeds and video files"""
    
    def __init__(self, confidence_threshold: float = 0.5, fps_limit: int = 10):
        """
        Initialize video analyzer
        
        Args:
            confidence_threshold: Detection confidence threshold
            fps_limit: Maximum FPS for processing (to control performance)
        """
        self.image_analyzer = ImageAnalyzer(confidence_threshold=confidence_threshold)
        self.fps_limit = fps_limit
        self.is_processing = False
        self.frame_skip = 1  # Process every nth frame
        
        # Results storage
        self.results_history = []
        self.max_history = 1000  # Keep last 1000 results
        
        # Callbacks for real-time updates
        self.frame_callback = None
        self.results_callback = None
        
    def set_callbacks(self, frame_callback: Callable = None, results_callback: Callable = None):
        """Set callbacks for real-time updates"""
        self.frame_callback = frame_callback
        self.results_callback = results_callback
    
    def analyze_webcam(self, camera_index: int = 0, display_window: bool = True):
        """
        Analyze live webcam feed
        
        Args:
            camera_index: Camera device index (0 for default camera)
            display_window: Whether to show live video window
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_processing = True
        frame_count = 0
        last_process_time = time.time()
        
        print(f"🎥 Starting webcam analysis (Camera {camera_index})")
        print("Press 'q' to quit, 's' to save current results")
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Control processing FPS
                if current_time - last_process_time >= (1.0 / self.fps_limit):
                    if frame_count % self.frame_skip == 0:
                        # Process frame
                        result = self._process_frame(frame, frame_count, current_time)
                        
                        # Add to history
                        self._add_to_history(result)
                        
                        # Call callbacks
                        if self.results_callback:
                            self.results_callback(result)
                    
                    last_process_time = current_time
                
                # Display frame with annotations
                if display_window:
                    annotated_frame = self._annotate_frame(frame, frame_count)
                    cv2.imshow('Street Vision AI - Live Analysis', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self.save_results_to_file(f"webcam_results_{int(time.time())}.json")
                
                # Call frame callback
                if self.frame_callback:
                    self.frame_callback(frame)
                    
        finally:
            cap.release()
            if display_window:
                cv2.destroyAllWindows()
            self.is_processing = False
            print("🛑 Webcam analysis stopped")
    
    def analyze_video_file(self, video_path: str, output_file: str = None, 
                          display_window: bool = True, save_frames: bool = False):
        """
        Analyze video file
        
        Args:
            video_path: Path to video file
            output_file: Optional output file for results
            display_window: Whether to show video playback
            save_frames: Whether to save annotated frames
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"🎬 Analyzing video: {Path(video_path).name}")
        print(f"📊 Total frames: {total_frames}, FPS: {fps:.1f}, Duration: {duration:.1f}s")
        
        self.is_processing = True
        frame_count = 0
        processed_count = 0
        
        # Setup frame saving if requested
        if save_frames:
            frames_dir = Path(f"analyzed_frames_{int(time.time())}")
            frames_dir.mkdir(exist_ok=True)
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every nth frame based on skip setting
                if frame_count % self.frame_skip == 0:
                    timestamp = frame_count / fps if fps > 0 else frame_count
                    result = self._process_frame(frame, frame_count, timestamp)
                    
                    self._add_to_history(result)
                    processed_count += 1
                    
                    # Call callbacks
                    if self.results_callback:
                        self.results_callback(result)
                    
                    # Save annotated frame if requested
                    if save_frames:
                        annotated_frame = self._annotate_frame(frame, frame_count)
                        frame_path = frames_dir / f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(str(frame_path), annotated_frame)
                
                # Display progress
                if frame_count % 30 == 0:  # Update every 30 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"📈 Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                
                # Display frame
                if display_window:
                    annotated_frame = self._annotate_frame(frame, frame_count)
                    cv2.imshow('Street Vision AI - Video Analysis', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self.save_results_to_file(f"video_results_{int(time.time())}.json")
                
                # Call frame callback
                if self.frame_callback:
                    self.frame_callback(frame)
                    
        finally:
            cap.release()
            if display_window:
                cv2.destroyAllWindows()
            self.is_processing = False
            
            print(f"✅ Video analysis complete!")
            print(f"📊 Processed {processed_count} frames out of {total_frames}")
            
            # Save results if output file specified
            if output_file:
                self.save_results_to_file(output_file)
    
    def analyze_rtsp_stream(self, rtsp_url: str, display_window: bool = True):
        """
        Analyze RTSP stream (IP cameras, etc.)
        
        Args:
            rtsp_url: RTSP stream URL
            display_window: Whether to show live video window
        """
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            raise ValueError(f"Could not connect to RTSP stream: {rtsp_url}")
        
        print(f"📡 Connected to RTSP stream: {rtsp_url}")
        
        self.is_processing = True
        frame_count = 0
        last_process_time = time.time()
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Lost connection to stream, attempting to reconnect...")
                    time.sleep(2)
                    cap.release()
                    cap = cv2.VideoCapture(rtsp_url)
                    continue
                
                frame_count += 1
                current_time = time.time()
                
                # Control processing FPS
                if current_time - last_process_time >= (1.0 / self.fps_limit):
                    if frame_count % self.frame_skip == 0:
                        result = self._process_frame(frame, frame_count, current_time)
                        self._add_to_history(result)
                        
                        if self.results_callback:
                            self.results_callback(result)
                    
                    last_process_time = current_time
                
                # Display frame
                if display_window:
                    annotated_frame = self._annotate_frame(frame, frame_count)
                    cv2.imshow('Street Vision AI - RTSP Stream', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self.save_results_to_file(f"rtsp_results_{int(time.time())}.json")
                
                if self.frame_callback:
                    self.frame_callback(frame)
                    
        finally:
            cap.release()
            if display_window:
                cv2.destroyAllWindows()
            self.is_processing = False
            print("🛑 RTSP stream analysis stopped")
    
    def _process_frame(self, frame, frame_number: int, timestamp: float) -> VideoAnalysisResult:
        """Process a single frame"""
        start_time = time.time()
        
        # Save frame temporarily for analysis
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        
        try:
            # Analyze using image analyzer
            result = self.image_analyzer.analyze_image(temp_path)
            
            processing_time = time.time() - start_time
            
            return VideoAnalysisResult(
                frame_number=frame_number,
                timestamp=timestamp,
                people_count=result.people_count,
                vehicle_count=result.vehicle_count,
                traffic_lights=result.traffic_lights,
                confidence_scores=result.confidence_scores,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"⚠️ Error processing frame {frame_number}: {str(e)}")
            return VideoAnalysisResult(
                frame_number=frame_number,
                timestamp=timestamp,
                people_count=0,
                vehicle_count=0,
                traffic_lights={"total": 0, "red": 0, "green": 0, "yellow": 0},
                confidence_scores={"people": 0.0, "vehicles": 0.0, "traffic_lights": 0.0},
                processing_time=0.0
            )
    
    def _annotate_frame(self, frame, frame_number: int):
        """Add annotations to frame for display"""
        annotated = frame.copy()
        
        # Get latest result if available
        if self.results_history:
            latest = self.results_history[-1]
            
            # Add text overlay
            overlay_text = [
                f"Frame: {frame_number}",
                f"People: {latest.people_count}",
                f"Vehicles: {latest.vehicle_count}",
                f"Traffic Lights: {latest.traffic_lights['total']}",
                f"FPS: {1.0/latest.processing_time:.1f}" if latest.processing_time > 0 else "FPS: --"
            ]
            
            # Draw background rectangle
            cv2.rectangle(annotated, (10, 10), (300, 150), (0, 0, 0), -1)
            cv2.rectangle(annotated, (10, 10), (300, 150), (255, 165, 0), 2)
            
            # Draw text
            for i, text in enumerate(overlay_text):
                y_pos = 35 + i * 25
                cv2.putText(annotated, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add title
        cv2.putText(annotated, "STREET VISION AI", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        return annotated
    
    def _add_to_history(self, result: VideoAnalysisResult):
        """Add result to history with size limit"""
        self.results_history.append(result)
        
        # Keep only recent results
        if len(self.results_history) > self.max_history:
            self.results_history = self.results_history[-self.max_history:]
    
    def get_statistics(self) -> Dict:
        """Get analysis statistics"""
        if not self.results_history:
            return {}
        
        people_counts = [r.people_count for r in self.results_history]
        vehicle_counts = [r.vehicle_count for r in self.results_history]
        processing_times = [r.processing_time for r in self.results_history]
        
        return {
            "total_frames_processed": len(self.results_history),
            "average_people": np.mean(people_counts),
            "max_people": max(people_counts),
            "average_vehicles": np.mean(vehicle_counts),
            "max_vehicles": max(vehicle_counts),
            "average_processing_time": np.mean(processing_times),
            "average_fps": 1.0 / np.mean(processing_times) if np.mean(processing_times) > 0 else 0
        }
    
    def save_results_to_file(self, filename: str):
        """Save analysis results to JSON file"""
        if not self.results_history:
            print("⚠️ No results to save")
            return
        
        # Prepare data for JSON serialization
        results_data = {
            "analysis_info": {
                "total_frames": len(self.results_history),
                "analysis_date": datetime.now().isoformat(),
                "statistics": self.get_statistics()
            },
            "frame_results": []
        }
        
        for result in self.results_history:
            results_data["frame_results"].append({
                "frame_number": result.frame_number,
                "timestamp": result.timestamp,
                "people_count": result.people_count,
                "vehicle_count": result.vehicle_count,
                "traffic_lights": result.traffic_lights,
                "confidence_scores": result.confidence_scores,
                "processing_time": result.processing_time
            })
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
    
    def stop_processing(self):
        """Stop video processing"""
        self.is_processing = False
    
    def set_fps_limit(self, fps: int):
        """Set FPS limit for processing"""
        self.fps_limit = max(1, min(fps, 30))  # Limit between 1-30 FPS
    
    def set_frame_skip(self, skip: int):
        """Set frame skip (process every nth frame)"""
        self.frame_skip = max(1, skip)

def main():
    """Example usage of VideoAnalyzer"""
    analyzer = VideoAnalyzer(confidence_threshold=0.5, fps_limit=5)
    
    print("🎥 Street Vision AI - Video Analysis System")
    print("=" * 50)
    print("Options:")
    print("1. Analyze webcam feed")
    print("2. Analyze video file")
    print("3. Analyze RTSP stream")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    try:
        if choice == "1":
            print("Starting webcam analysis...")
            analyzer.analyze_webcam(camera_index=0)
        
        elif choice == "2":
            video_path = input("Enter video file path: ").strip()
            print(f"Starting video file analysis: {video_path}")
            analyzer.analyze_video_file(video_path, output_file="video_analysis_results.json")
        
        elif choice == "3":
            rtsp_url = input("Enter RTSP URL: ").strip()
            print(f"Starting RTSP stream analysis: {rtsp_url}")
            analyzer.analyze_rtsp_stream(rtsp_url)
        
        else:
            print("Invalid choice")
            return
        
        # Print final statistics
        stats = analyzer.get_statistics()
        if stats:
            print("\n📊 Final Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")
    
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        analyzer.stop_processing()

if __name__ == "__main__":
    main()