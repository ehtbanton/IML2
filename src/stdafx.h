#pragma once
#pragma once

// First define these to prevent Windows macro conflicts
#define _USE_MATH_DEFINES
#define NOMINMAX 
#define WIN32_LEAN_AND_MEAN

// 1. Include SFML headers BEFORE any Windows headers
#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

// 2. Now include Windows headers
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>

// 3. Standard library headers
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <utility>
#include <cmath>
#include <ctime>
#include <cstdint>

// 4. Third-party libraries
#include <fftw3.h>

// 5. Project-specific forward declarations and typedefs
// (Add these as needed)