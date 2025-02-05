import mmap
import ctypes
import time
import numpy as np
import subprocess
import signal
import sys
import os
from pathlib import Path
from audio_similarity import AudioSimilarityAnalyzer
from physical_system import chooseColour

# Define the shared memory structure to match the C++ code
class SharedMemory(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magnitudes", ctypes.c_float * 512),
        ("timestamp", ctypes.c_double),
        ("new_data_available", ctypes.c_bool)
    ]

def find_executable():
    # Get the script's directory
    script_dir = Path(__file__).parent.absolute()
    
    # Possible executable names
    exe_names = ['IML2.exe', 'iml_fftw_no_spectrogram.exe']
    
    # Possible relative paths to check
    relative_paths = [
        'x64/Debug',
        '../x64/Debug',
        '../../x64/Debug',
        'Debug',
        '../Debug',
        '../../Debug'
    ]
    
    # Check all combinations
    for exe_name in exe_names:
        for rel_path in relative_paths:
            full_path = script_dir / rel_path / exe_name
            if full_path.exists():
                return str(full_path)
    
    return None

def main():
    # Find the executable
    exe_path = find_executable()
    if not exe_path:
        print("Error: Could not find the C++ executable in any of the expected locations.")
        print("Please ensure the executable exists in one of the following locations relative to this script:")
        print("\n".join(["- x64/Debug", "- ../x64/Debug", "- ../../x64/Debug"]))
        return

    print(f"Found executable at: {exe_path}")
    
    try:
        # Initialize similarity analyzer
        similarity_analyzer = AudioSimilarityAnalyzer()
        
        # Start the C++ program
        cpp_process = subprocess.Popen([exe_path])
        print("Started C++ program, waiting for shared memory initialization...")
        
        # Give the C++ program time to initialize
        time.sleep(2)
        
        # Try to open the shared memory
        try:
            shm = mmap.mmap(
                -1,  # Create a new map
                ctypes.sizeof(SharedMemory),
                "Local\\SpectrogramData",
                mmap.ACCESS_READ
            )
        except Exception as e:
            print(f"Error connecting to shared memory: {e}")
            cpp_process.terminate()
            return
        
        print("\nConnected to shared memory. Reading frequency data...")
        print("Commands:")
        print("  r - Start recording reference pattern")
        print("  s - Stop recording reference pattern")
        print("  q - Quit")
        print("\nPress 'r' when you want to record a reference pattern.")
        
        last_timestamp = 0.0
        
        import msvcrt  # For Windows keyboard input
        
        while True:
            # Check for keyboard input
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                if key == 'r':
                    similarity_analyzer.start_recording_reference()
                elif key == 's':
                    similarity_analyzer.stop_recording_reference()
                elif key == 'q':
                    break
            
            # Read the shared memory
            shm.seek(0)
            raw_data = shm.read(ctypes.sizeof(SharedMemory))
            shared_data = SharedMemory.from_buffer_copy(raw_data)
            
            # Only process new data
            if shared_data.new_data_available and shared_data.timestamp != last_timestamp:
                # Convert magnitudes to numpy array
                magnitudes = np.array([shared_data.magnitudes[i] for i in range(512)])
                
                # Process the frame through similarity analyzer
                result = similarity_analyzer.process_frame(magnitudes)
                
                # Display appropriate message based on status
                if result["status"] == "recording_reference":
                    print("\rRecording reference... Press 's' to stop recording", end="")
                elif result["status"] == "analyzing":
                    similarity_score = result["similarity_score"]
                    is_similar = result["is_similar"]
                    print(f"\rSimilarity: {similarity_score:.2f} | Similar: {is_similar}", end="\n")
                elif result["status"] == "collecting_data":
                    print("\rCollecting data...", end="")
                
                # Use the chooseColour function
                colour = chooseColour(similarity_analyzer)
                print(f"Chosen colour: {colour}")
                
                last_timestamp = shared_data.timestamp
            
            time.sleep(0.016)  # ~60Hz update rate
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if 'shm' in locals():
            shm.close()
        if 'cpp_process' in locals():
            cpp_process.terminate()
            cpp_process.wait()

if __name__ == "__main__":
    main()